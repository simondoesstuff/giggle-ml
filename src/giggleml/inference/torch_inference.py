import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import override

from tqdm import tqdm
import filelock
import torch
import torch.distributed as dist
import zarr
from torch import FloatTensor, nn

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

    # Distributed setup
    if dist.is_initialized():
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        for set_idx in tqdm(range(n_sets)):
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
