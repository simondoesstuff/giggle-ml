import os
import shutil
import socket
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
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
    batch_size: int = 256,
) -> None:
    """Embed interval sets across all GPUs using torchrun and save to Zarr v3.

    Uses file locks for work-stealing at the set level: processes race to claim
    entire interval sets, then process all batches within that set sequentially.

    Args:
        model: The nucleotide embedding model.
        fasta: Dictionary mapping chromosome names to sequences.
        intervals: Sequence of interval sets to embed.
        out_paths: Output zarr array path for each interval set.
        batch_size: Number of intervals to process per batch.
    """
    assert len(intervals) == len(out_paths)

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

    # Work-stealing at set level: each process races to claim entire interval sets
    with torch.inference_mode():
        for set_idx in tqdm(range(n_sets), disable=rank != 0):
            out_stem = Path(out_paths[set_idx]).stem
            done_path = lock_dir / f"{out_stem}.done"

            # Fast path: skip if already processed
            if done_path.exists():
                continue

            lock = filelock.FileLock(lock_dir / f"{out_stem}.lock")
            try:
                lock.acquire(timeout=0)
            except filelock.Timeout:
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
                arr = zarr.create_array(
                    out_path,
                    shape=(n_intervals, model.edim),
                    chunks=(min(batch_size, n_intervals), model.edim),
                    dtype="float32",
                    overwrite=True,
                )

                # Process all batches in this set
                for batch_start in range(0, n_intervals, batch_size):
                    batch_end = min(batch_start + batch_size, n_intervals)
                    batch_intervals = all_intervals[batch_start:batch_end]

                    # intervals -> sequences -> embeddings
                    sequences = list(fasta_map(fasta, batch_intervals))
                    batch_input = model.collate(sequences)
                    if hasattr(batch_input, "to"):
                        batch_input = batch_input.to(device)  # pyright: ignore[reportAttributeAccessIssue]

                    embeddings = model(batch_input)
                    arr[batch_start:batch_end] = embeddings.cpu().numpy()

                # Mark set as complete
                done_path.touch()
            finally:
                lock.release()

    # Final sync and cleanup
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        shutil.rmtree(lock_dir)
