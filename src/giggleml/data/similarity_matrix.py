"""Memory-mapped similarity matrix for genomic interval comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from giggleml.data.giggle import GiggleIndex
from giggleml.utils.file_utils import Pathish

type MemMapMode = Literal["r", "r+", "w+", "c"]


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
    ) -> "SimilarityMatrix":
        """Build a similarity matrix from a GiggleIndex using combo scores.

        Performs self_query for each BED file in the index and populates the
        similarity matrix with the combo_score values.

        Args:
            index: GiggleIndex to query.
            output_path: Path for the output similarity matrix file.
            symmetric: If True, enforce M[a,b] = M[b,a] by taking max on conflict.

        Returns:
            The populated SimilarityMatrix.
        """
        beds = sorted(index.list_beds)
        n = len(beds)
        bed_to_idx = {bed: i for i, bed in enumerate(beds)}

        matrix = SimilarityMatrix(output_path, n, mode="w+")

        for i, bed in tqdm(enumerate(beds), desc="Building (giggle) similarity matrix"):
            results = index.self_query(bed)

            for result in results:
                if result.file not in bed_to_idx:
                    continue
                j = bed_to_idx[result.file]
                matrix[i, j] = result.combo_score

        if symmetric:
            # Enforce symmetry by taking element-wise max with transpose
            arr = matrix.array
            np.maximum(arr, arr.T.copy(), out=arr)

        matrix.flush()
        return matrix
