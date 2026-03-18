"""Memory-mapped similarity matrix for genomic interval comparisons."""

from __future__ import annotations

import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from giggleml.data.giggle import GiggleIndex, GiggleResult
from giggleml.utils.file_utils import Pathish

type MemMapMode = Literal["r", "r+", "w+", "c"]


def _query_bed_worker(
    bed: str,
    directory: Path,
    index_dir: Path,
    genome_size: int | None,
) -> tuple[str, list[GiggleResult]]:
    """Worker function for parallel query.

    Creates a fresh GiggleIndex in the subprocess (reuses existing index on disk)
    and queries directly by path to avoid redundant list_beds calls.
    """
    index = GiggleIndex(directory, index_dir=index_dir, genome_size=genome_size)
    query_path = directory / bed
    results = index.query(query_path)
    return bed, results


class SimilarityMatrix:
    """Memory-mapped fp16 similarity matrix.

    Provides a simple abstraction over a memory-mapped numpy array for storing
    pairwise similarity values. The matrix is stored as fp16 to reduce memory
    and disk usage.

    Args:
        path: Path to the memory-mapped file.
        n: Size of the square matrix (n x n).
        mode: File mode - 'r' for read-only, 'r+' for read-write existing,
              'w+' for create/overwrite.
    """

    _path: Path
    _n: int
    _mmap: np.memmap[tuple[int, int], np.dtype[np.float16]]

    def __init__(
        self,
        path: Pathish,
        n: int,
        *,
        mode: MemMapMode,
    ) -> None:
        self._path = Path(path)
        self._n = n

        if mode == "w+":
            # Create new file with zeros
            self._mmap = np.memmap(
                self._path,
                dtype=np.float16,
                mode="w+",
                shape=(n, n),
            )
            self._mmap.fill(0)
            self._mmap.flush()
        else:
            # Open existing file
            self._mmap = np.memmap(
                self._path,
                dtype=np.float16,
                mode=mode,
                shape=(n, n),
            )

    @property
    def path(self) -> Path:
        """Path to the memory-mapped file."""
        return self._path

    @property
    def n(self) -> int:
        """Size of the matrix (n x n)."""
        return self._n

    @property
    def array(self) -> NDArray[np.float16]:
        """The underlying numpy array (memory-mapped)."""
        return self._mmap

    def flush(self) -> None:
        """Flush changes to disk."""
        self._mmap.flush()

    def __getitem__(self, key: tuple[int, int] | int) -> np.floating:
        """Get item(s) from the matrix."""
        return self._mmap[key]

    def __setitem__(
        self, key: tuple[int, int] | int, value: float | np.floating
    ) -> None:
        """Set item(s) in the matrix."""
        self._mmap[key] = value

    @staticmethod
    def build_from_giggle(
        index: GiggleIndex,
        output_path: Pathish,
        *,
        symmetric: bool = True,
        n_jobs: int | None = None,
    ) -> "SimilarityMatrix":
        """Build a similarity matrix from a GiggleIndex using combo scores.

        Performs self_query for each BED file in the index and populates the
        similarity matrix with the combo_score values.

        Args:
            index: GiggleIndex to query.
            output_path: Path for the output similarity matrix file.
            symmetric: If True, enforce M[a,b] = M[b,a] by taking max on conflict.
            n_jobs: Number of parallel workers. Defaults to CPU count. Use 1 for
                sequential execution (useful for testing).

        Returns:
            The populated SimilarityMatrix.
        """
        # Ensure index is built before spawning workers
        index.build_index()

        beds = sorted(index.list_beds)
        n = len(beds)
        bed_to_idx = {bed: i for i, bed in enumerate(beds)}

        matrix = SimilarityMatrix(output_path, n, mode="w+")

        n_jobs = n_jobs or os.cpu_count() or 1

        def populate_row(bed: str, results: list[GiggleResult]) -> None:
            i = bed_to_idx[bed]
            for result in results:
                if result.file not in bed_to_idx:
                    continue
                j = bed_to_idx[result.file]
                matrix[i, j] = result.combo_score

        if n_jobs == 1:
            # Sequential mode - use self_query directly (no pickling needed)
            for bed in tqdm(beds, desc="Building similarity matrix"):
                results = index.self_query(bed)
                populate_row(bed, results)
        else:
            # Parallel mode - use worker function with Pool
            worker = partial(
                _query_bed_worker,
                directory=index.directory,
                index_dir=index.index_dir,
                genome_size=index.genome_size,
            )

            with Pool(processes=n_jobs) as pool:
                results_iter = pool.imap_unordered(worker, beds)
                for bed, results in tqdm(
                    results_iter,
                    total=n,
                    desc=f"Building similarity matrix ({n_jobs} workers)",
                ):
                    populate_row(bed, results)

        if symmetric:
            # Enforce symmetry by taking element-wise max with transpose
            arr = matrix.array
            np.maximum(arr, arr.T.copy(), out=arr)

        matrix.flush()
        return matrix
