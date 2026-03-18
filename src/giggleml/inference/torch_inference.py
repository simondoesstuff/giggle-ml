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


def _setup_distributed() -> tuple[int, torch.device]:
    """Initialize distributed process group if running via torchrun.

    Returns:
        (rank, device) tuple. rank=0 for single-device execution.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    in_distributed = world_size > 1 or "RANK" in os.environ

    if in_distributed and dist.is_available():
        rank = int(os.environ.get("LOCAL_RANK", 0))
        os.environ["MASTER_PORT"] = str(os.environ.get("MASTER_PORT", _infer_port()))
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        backend = _infer_backend()
        device_id = rank if backend == "nccl" else None
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            device_id=device_id,
            timeout=timedelta(minutes=5),
        )
        device = _infer_device(rank)
        print(f"Using {world_size} devices: {device}")
    else:
        rank = 0
        device = _infer_device(0)
        print("Using 1 device:", device)

    return rank, device


def _accumulate_batch(
    all_intervals: list[GenomicInterval],
    start_idx: int,
    vram_coeffs: VRAMCoeffs,
    vram_cap: float,
) -> tuple[list[GenomicInterval], int]:
    """Accumulate intervals into a batch until VRAM estimate exceeds cap.

    Returns:
        (batch_intervals, next_idx) where next_idx is the index after the batch.
    """
    batch_intervals: list[GenomicInterval] = []
    batch_max_len = 0
    i = start_idx

    while i < len(all_intervals):
        interval = all_intervals[i]
        interval_size = interval[2] - interval[1]
        new_n = len(batch_intervals) + 1
        new_max_len = max(batch_max_len, interval_size)

        # Always include at least one interval per batch
        if batch_intervals:
            est_vram = vram_coeffs.estimate(new_n, new_max_len)
            if est_vram > vram_cap:
                break

        batch_intervals.append(interval)
        batch_max_len = new_max_len
        i += 1

    return batch_intervals, i


def _run_batch_with_oom_retry[T](
    model: NucleotideModel[T],
    fasta: dict[str, str],
    batch_intervals: list[GenomicInterval],
    device: torch.device,
    current_idx: int,
) -> tuple[torch.Tensor, int, float]:
    """Run model inference with OOM recovery.

    On OOM, halves the batch and retries. Returns the scale factor to apply
    to future VRAM estimates (< 1.0 if OOM occurred).

    Embeddings are returned on-device to allow overlapping computation with
    device-to-host transfer (caller moves to CPU at write time).

    Returns:
        (embeddings, new_idx, vram_scale_multiplier)
        - embeddings: Device tensor of shape (actual_batch_size, edim)
        - new_idx: Updated index (may be less than current_idx if batch was split)
        - vram_scale_multiplier: Multiply into vram_scale (1.0 if no OOM, <1.0 if OOM)
    """
    scale_multiplier = 1.0

    while True:
        try:
            sequences = list(fasta_map(fasta, batch_intervals))
            batch_input = model.collate(sequences)
            if hasattr(batch_input, "to"):
                batch_input = batch_input.to(device)  # pyright: ignore[reportAttributeAccessIssue]

            embeddings = model(batch_input)
            return embeddings, current_idx, scale_multiplier

        except RuntimeError as e:
            err_msg = str(e).lower()
            is_oom = (
                isinstance(e, torch.OutOfMemoryError)
                or "out of memory" in err_msg
                or "cufft" in err_msg  # cuFFT errors often indicate memory pressure
            )
            if not is_oom:
                raise

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if len(batch_intervals) == 1:
                raise RuntimeError(
                    f"OOM on single interval of size {batch_intervals[0][2] - batch_intervals[0][1]}"
                ) from e

            # Halve the batch and put the rest back
            split = len(batch_intervals) // 2
            current_idx -= len(batch_intervals) - split
            batch_intervals = batch_intervals[:split]
            scale_multiplier *= 0.80
            print(f"OOM: reducing batch to {split}")


def _process_interval_set[T](
    model: NucleotideModel[T],
    fasta: dict[str, str],
    all_intervals: list[GenomicInterval],
    out_path: Path,
    device: torch.device,
    vram_coeffs: VRAMCoeffs,
    vram_cap: float,
    zarr_chunk_size: int,
) -> None:
    """Embed all intervals in a set and write to a zarr array."""
    n_intervals = len(all_intervals)
    chunk_size = min(zarr_chunk_size, n_intervals)

    arr = zarr.create_array(
        out_path,
        shape=(n_intervals, model.edim),
        chunks=(chunk_size, model.edim),
        dtype="float32",
        overwrite=True,
    )

    i = 0
    cached_embeddings: list[torch.Tensor] = []
    cache_start = 0
    vram_scale = 1.0

    while i < n_intervals:
        # --- Batch accumulation ---
        effective_cap = vram_cap * vram_scale
        batch_intervals, i = _accumulate_batch(
            all_intervals, i, vram_coeffs, effective_cap
        )

        # --- Inference with OOM retry ---
        embeddings, i, scale_mult = _run_batch_with_oom_retry(
            model, fasta, batch_intervals, device, i
        )
        vram_scale *= scale_mult
        cached_embeddings.append(embeddings)

        # --- Chunk-aligned zarr writes (transfer to CPU here) ---
        cached_count = sum(e.shape[0] for e in cached_embeddings)
        if cached_count >= chunk_size or i == n_intervals:
            combined = torch.cat(cached_embeddings, dim=0).cpu()
            arr[cache_start : cache_start + combined.shape[0]] = combined.numpy()
            cache_start += combined.shape[0]
            cached_embeddings = []


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

    # -------------------------------------------------------------------------
    # 1. Sort inputs for consistent ordering across ranks
    # -------------------------------------------------------------------------
    sort_order = sorted(range(len(out_paths)), key=lambda i: Path(out_paths[i]).stem)
    intervals = [intervals[i] for i in sort_order]
    out_paths = [out_paths[i] for i in sort_order]

    # -------------------------------------------------------------------------
    # 2. Distributed setup
    # -------------------------------------------------------------------------
    rank, device = _setup_distributed()
    model = model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # 3. Work-stealing infrastructure (file locks)
    # -------------------------------------------------------------------------
    n_sets = len(intervals)
    lock_dir = Path(out_paths[0]).parent / ".embed_locks"

    if rank == 0:
        lock_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    out_stems = [Path(out_paths[i]).stem for i in range(n_sets)]
    done_paths = [lock_dir / f"{stem}.done" for stem in out_stems]
    lock_paths = [lock_dir / f"{stem}.lock" for stem in out_stems]

    def count_done() -> int:
        return sum(1 for p in done_paths if p.exists())

    # -------------------------------------------------------------------------
    # 4. Work-stealing loop
    # -------------------------------------------------------------------------
    retry_delay = 2.0
    with torch.inference_mode():
        pbar = tqdm(total=n_sets, disable=rank != 0, desc="Embedding sets")
        pbar.n = count_done()
        pbar.refresh()

        while True:
            processed_any = False

            for set_idx in range(n_sets):
                done_path = done_paths[set_idx]
                if done_path.exists():
                    continue

                # Try to acquire lock (non-blocking)
                lock = filelock.FileLock(lock_paths[set_idx])
                try:
                    lock.acquire(timeout=0)
                except filelock.Timeout:
                    continue

                try:
                    if done_path.exists():  # Double-check after lock
                        continue

                    # Process this interval set
                    out_path = Path(out_paths[set_idx])
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    _process_interval_set(
                        model=model,
                        fasta=fasta,
                        all_intervals=list(intervals[set_idx]),
                        out_path=out_path,
                        device=device,
                        vram_coeffs=vram_coeffs,
                        vram_cap=vram_cap,
                        zarr_chunk_size=zarr_chunk_size,
                    )

                    done_path.touch()
                    processed_any = True
                    pbar.n = count_done()
                    pbar.refresh()
                finally:
                    lock.release()

            # Check completion and retry logic
            n_done = count_done()
            if n_done == n_sets:
                break
            if processed_any:
                continue

            # Wait for other ranks to finish their work
            pbar.n = n_done
            pbar.refresh()
            time.sleep(retry_delay)

        pbar.close()

    # -------------------------------------------------------------------------
    # 5. Cleanup
    # -------------------------------------------------------------------------
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        shutil.rmtree(lock_dir)
