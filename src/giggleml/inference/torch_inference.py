import os
import shutil
import socket
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import override

import filelock
import torch
import torch.distributed as dist
import zarr
from torch import FloatTensor, nn
from tqdm import tqdm

from giggleml.data.fasta import fasta_map
from giggleml.types import GenomicInterval
from giggleml.utils.file_utils import Pathish
from giggleml.utils.vram import VRAMCoeffs


class NucleotideModel[T](nn.Module, ABC):
    seq_max: int
    edim: int

    @abstractmethod
    def collate(self, sequences: Iterable[str]) -> T:
        """Tokenize a batch of nucleotide sequences for the forward pass."""
        ...

    @override
    @abstractmethod
    def forward(self, batch: T) -> FloatTensor: ...


def _infer_port() -> int:
    # Find a free port automatically
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _infer_backend() -> str:
    """Infer the optimal distributed backend based on available hardware.

    Returns:
        "nccl" for CUDA systems (best GPU communication performance)
        "gloo" for MPS or CPU systems (universal compatibility)
    """
    if torch.cuda.is_available():
        return "nccl"  # Best for CUDA GPUs
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "gloo"  # MPS doesn't support nccl
    else:
        return "gloo"  # CPU fallback


def _infer_device(rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    elif torch.backends.mps.is_built():
        return torch.device("mps")  # for mac use
    else:
        return torch.device("cpu")


def embed_intervals[T](
    model: NucleotideModel[T],
    fasta: dict[str, str],
    intervals: Sequence[Iterable[GenomicInterval]],
    out_paths: Sequence[Pathish],
    *,
    vram_coeffs: VRAMCoeffs,
    vram_cap: float,
    zarr_chunk_size: int = 256,
) -> None:
    """Embed interval sets across all GPUs using torchrun and save to Zarr v3.

    Uses file locks for work-stealing at the set level: processes race to claim
    entire interval sets, then process all batches within that set sequentially.

    Batch sizes are dynamically computed using a VRAM model: intervals are
    accumulated until adding another would exceed the estimated VRAM cap.
    This minimizes padding waste when intervals are sorted by size.

    Embeddings are cached in memory and written to zarr in chunk-aligned blocks
    to minimize I/O overhead.

    Args:
        model: The nucleotide embedding model.
        fasta: Dictionary mapping chromosome names to sequences.
        intervals: Sequence of interval sets to embed.
        out_paths: Output zarr array path for each interval set.
        vram_coeffs: VRAM model coefficients (from estimate_vram_coefficients.py).
        vram_cap: Target VRAM cap in bytes.
        zarr_chunk_size: Number of embeddings per zarr chunk (default 256).
    """
    assert len(intervals) == len(out_paths)

    # Sort by output path to ensure consistent ordering across all ranks.
    # Critical for distributed work-stealing: all ranks must iterate in the same order.
    sort_order = sorted(range(len(out_paths)), key=lambda i: Path(out_paths[i]).stem)
    intervals = [intervals[i] for i in sort_order]
    out_paths = [out_paths[i] for i in sort_order]

    # Distributed setup - only initialize if actually running in a distributed context
    # (e.g., via torchrun), detected by WORLD_SIZE > 1 or RANK being set
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    in_distributed = world_size > 1 or "RANK" in os.environ

    if in_distributed and dist.is_available():
        rank = int(os.environ.get("LOCAL_RANK", 0))
        os.environ["MASTER_PORT"] = str(os.environ.get("MASTER_PORT", _infer_port()))
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        backend = _infer_backend()
        # device_id only works with NCCL backend (CUDA)
        device_id = rank if backend == "nccl" else None
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            device_id=device_id,
            timeout=timedelta(minutes=5),  # Short timeout - all ranks sync closely
        )
        device = _infer_device(rank)
        print(f"Using {world_size} devices: {device}")
    else:
        rank = 0
        device = _infer_device(0)
        print("Using 1 device:", device)

    model = model.to(device)
    model.eval()

    n_sets = len(intervals)
    lock_dir = Path(out_paths[0]).parent / ".embed_locks"

    if rank == 0:
        lock_dir.mkdir(parents=True, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    # Pre-compute stems and paths for efficiency
    out_stems = [Path(out_paths[i]).stem for i in range(n_sets)]
    done_paths = [lock_dir / f"{stem}.done" for stem in out_stems]
    lock_paths = [lock_dir / f"{stem}.lock" for stem in out_stems]

    def count_done() -> int:
        return sum(1 for p in done_paths if p.exists())

    # Work-stealing with retry: each process races to claim interval sets,
    # retrying until all sets are complete (handles crashed ranks)
    retry_delay = 2.0  # seconds to wait before retrying when no work available
    with torch.inference_mode():
        pbar = tqdm(total=n_sets, disable=rank != 0, desc="Embedding sets")
        pbar.n = count_done()
        pbar.refresh()

        while True:
            processed_any = False

            for set_idx in range(n_sets):
                done_path = done_paths[set_idx]

                # Fast path: skip if already processed
                if done_path.exists():
                    continue

                lock = filelock.FileLock(lock_paths[set_idx])
                try:
                    lock.acquire(timeout=0)
                except filelock.Timeout:
                    # Another rank is working on this file, try next
                    continue

                try:
                    # Double-check after acquiring lock
                    if done_path.exists():
                        continue

                    # Materialize this interval set
                    all_intervals = list(intervals[set_idx])
                    n_intervals = len(all_intervals)

                    # Create output zarr array
                    out_path = Path(out_paths[set_idx])
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    chunk_size = min(zarr_chunk_size, n_intervals)
                    arr = zarr.create_array(
                        out_path,
                        shape=(n_intervals, model.edim),
                        chunks=(chunk_size, model.edim),
                        dtype="float32",
                        overwrite=True,
                    )

                    # Process batches with dynamic sizing based on VRAM model
                    # Cache embeddings and write in chunk-aligned blocks
                    i = 0
                    cached_embeddings: list[torch.Tensor] = []
                    cache_start = 0  # Start index for cached embeddings
                    vram_scale = 1.0  # Reduced on OOM to shrink future batches

                    while i < n_intervals:
                        batch_intervals: list[GenomicInterval] = []
                        batch_max_len = 0
                        effective_cap = vram_cap * vram_scale

                        # Accumulate intervals until adding another exceeds VRAM cap
                        while i < n_intervals:
                            interval = all_intervals[i]
                            interval_size = interval[2] - interval[1]
                            new_n = len(batch_intervals) + 1
                            new_max_len = max(batch_max_len, interval_size)

                            # Always include at least one interval per batch
                            if batch_intervals:
                                est_vram = vram_coeffs.estimate(new_n, new_max_len)
                                if est_vram > effective_cap:
                                    break

                            batch_intervals.append(interval)
                            batch_max_len = new_max_len
                            i += 1

                        # intervals -> sequences -> embeddings (with OOM retry)
                        while True:
                            try:
                                sequences = list(fasta_map(fasta, batch_intervals))
                                batch_input = model.collate(sequences)
                                if hasattr(batch_input, "to"):
                                    batch_input = batch_input.to(device)  # pyright: ignore[reportAttributeAccessIssue]

                                embeddings = model(batch_input).cpu()
                                cached_embeddings.append(embeddings)
                                break  # Success, exit retry loop
                            except RuntimeError as e:
                                # Only handle OOM errors, re-raise everything else
                                is_oom = (
                                    isinstance(e, torch.OutOfMemoryError)
                                    or "out of memory" in str(e).lower()
                                )
                                if not is_oom:
                                    raise

                                # Clear GPU memory
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                                if len(batch_intervals) == 1:
                                    # Can't reduce further, re-raise
                                    raise RuntimeError(
                                        f"OOM on single interval of size {batch_intervals[0][2] - batch_intervals[0][1]}"
                                    ) from e

                                # Halve the batch and put the rest back
                                split = len(batch_intervals) // 2
                                i -= (
                                    len(batch_intervals) - split
                                )  # Put back unused intervals
                                batch_intervals = batch_intervals[:split]
                                vram_scale *= 0.80  # Reduce future batch sizes
                                print(
                                    f"OOM: reducing batch to {split}, vram_scale={vram_scale:.2f}"
                                )

                        # Write when we've accumulated a full chunk or reached the end
                        cached_count = sum(e.shape[0] for e in cached_embeddings)
                        if cached_count >= chunk_size or i == n_intervals:
                            combined = torch.cat(cached_embeddings, dim=0)
                            arr[cache_start : cache_start + combined.shape[0]] = (
                                combined.numpy()
                            )
                            cache_start += combined.shape[0]
                            cached_embeddings = []

                    # Mark set as complete
                    done_path.touch()
                    processed_any = True
                    pbar.n = count_done()
                    pbar.refresh()
                finally:
                    lock.release()

            # Check if all done
            n_done = count_done()
            if n_done == n_sets:
                break

            # If we processed something this round, immediately try again
            if processed_any:
                continue

            # No work available but not all done - other ranks are working
            # Wait and retry (handles case where a rank crashes mid-processing)
            # Update progress bar to reflect work done by other ranks
            pbar.n = n_done
            pbar.refresh()
            time.sleep(retry_delay)

        pbar.close()

    # Final sync and cleanup
    # Safety: all ranks exit the work loop when count_done() == n_sets (shared FS state).
    # Maximum skew between ranks reaching this barrier is retry_delay (~2s), well within
    # the 5-minute timeout set at init_process_group.
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        shutil.rmtree(lock_dir)
