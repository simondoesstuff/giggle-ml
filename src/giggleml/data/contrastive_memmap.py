"""Memory-mapped storage for contrastive learning data.

Consolidates embeddings and intervals from multiple zarr/BED files into
two memory-mapped files for fast preloading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import ml_dtypes
import numpy as np
import zarr
from numpy.typing import NDArray
from tqdm import tqdm

from giggleml.data.intervals import load_bed_array
from giggleml.utils.file_utils import Pathish

# Register bfloat16 dtype
_bf16_dtype = ml_dtypes.bfloat16

type MemMapMode = Literal["r", "w+"]

METADATA_FILENAME = "metadata.json"
DATA_FILENAME = "data.mmap"
CURRENT_VERSION = 2


@dataclass
class ContrastiveMemmapMetadata:
    """Metadata for contrastive memmap storage.

    Attributes:
        version: Format version number.
        embedding_dim: Dimension of embeddings.
        total_intervals: Total number of intervals across all files.
        num_files: Number of BED files stored.
        intervals_byte_offset: Byte offset where intervals data starts in data.mmap.
        bed_names: Sorted list of BED file names (without extensions).
        offsets: Start index for each file in the memmap arrays.
        lengths: Number of intervals per file.
    """

    version: int
    embedding_dim: int
    total_intervals: int
    num_files: int
    intervals_byte_offset: int = 0
    bed_names: list[str] = field(default_factory=list)
    offsets: list[int] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "embedding_dim": self.embedding_dim,
            "total_intervals": self.total_intervals,
            "num_files": self.num_files,
            "intervals_byte_offset": self.intervals_byte_offset,
            "bed_names": self.bed_names,
            "offsets": self.offsets,
            "lengths": self.lengths,
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> ContrastiveMemmapMetadata:
        """Create from dictionary."""
        return ContrastiveMemmapMetadata(
            version=int(data["version"]),  # pyright: ignore[reportArgumentType]
            embedding_dim=int(data["embedding_dim"]),  # pyright: ignore[reportArgumentType]
            total_intervals=int(data["total_intervals"]),  # pyright: ignore[reportArgumentType]
            num_files=int(data["num_files"]),  # pyright: ignore[reportArgumentType]
            intervals_byte_offset=int(data["intervals_byte_offset"]),  # pyright: ignore[reportArgumentType]
            bed_names=list(data["bed_names"]),  # pyright: ignore[reportArgumentType]
            offsets=list(data["offsets"]),  # pyright: ignore[reportArgumentType]
            lengths=list(data["lengths"]),  # pyright: ignore[reportArgumentType]
        )


class ContrastiveMemmapData:
    """Memory-mapped storage for contrastive learning embeddings and intervals.

    Stores consolidated data from multiple zarr/BED files in a single memory-mapped
    file with a JSON metadata file for offsets and ordering. Using a single file
    allows faster preloading via one contiguous read.

    File structure:
        memmap_dir/
            data.mmap      # embeddings (bfloat16) followed by intervals (int32)
            metadata.json  # Offsets, lengths, bed_names ordering, byte offsets

    Layout of data.mmap:
        [0, intervals_byte_offset): embeddings as bfloat16, shape (total_intervals, embedding_dim)
        [intervals_byte_offset, end): intervals as int32, shape (total_intervals, 3)

    Args:
        memmap_dir: Directory containing the memmap files.
        mode: File mode - 'r' for read-only, 'w+' for create/overwrite.
    """

    _dir: Path
    _metadata: ContrastiveMemmapMetadata
    _embeddings: NDArray[np.floating]
    _intervals: NDArray[np.int32]

    def __init__(
        self,
        memmap_dir: Pathish,
        *,
        mode: MemMapMode = "r",
    ) -> None:
        self._dir = Path(memmap_dir)

        if mode == "r":
            # Load existing memmap
            self._metadata = self._load_metadata()

            # Load the unified data file
            data_path = self._dir / DATA_FILENAME
            total_bytes = data_path.stat().st_size

            # Memory-map the whole file as bytes
            raw_mmap = np.memmap(data_path, dtype=np.uint8, mode="r", shape=(total_bytes,))

            # Create views into embeddings and intervals sections
            emb_bytes = self._metadata.intervals_byte_offset
            emb_shape = (self._metadata.total_intervals, self._metadata.embedding_dim)
            self._embeddings = np.ndarray(
                emb_shape, dtype=_bf16_dtype, buffer=raw_mmap[:emb_bytes]
            )

            ivs_shape = (self._metadata.total_intervals, 3)
            self._intervals = np.ndarray(
                ivs_shape, dtype=np.int32, buffer=raw_mmap[emb_bytes:]
            )
        else:
            # Will be populated by build_from_files
            self._metadata = ContrastiveMemmapMetadata(
                version=CURRENT_VERSION,
                embedding_dim=0,
                total_intervals=0,
                num_files=0,
            )

    def _load_metadata(self) -> ContrastiveMemmapMetadata:
        """Load metadata from JSON file."""
        metadata_path = self._dir / METADATA_FILENAME
        with open(metadata_path) as f:
            data = json.load(f)
        version = data.get("version", 0)
        if version != CURRENT_VERSION:
            raise ValueError(
                f"Memmap version mismatch: found v{version}, expected v{CURRENT_VERSION}. "
                f"Please rebuild the memmap with build_contrastive_memmap.py"
            )
        return ContrastiveMemmapMetadata.from_dict(data)

    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        metadata_path = self._dir / METADATA_FILENAME
        with open(metadata_path, "w") as f:
            json.dump(self._metadata.to_dict(), f, indent=2)

    @property
    def metadata(self) -> ContrastiveMemmapMetadata:
        """Access the metadata."""
        return self._metadata

    @property
    def bed_names(self) -> list[str]:
        """Sorted list of BED file names."""
        return self._metadata.bed_names

    def get_embeddings(self, idx: int) -> NDArray[np.floating]:
        """Return embeddings slice for file at index.

        Args:
            idx: Index of the file (matches bed_names ordering).

        Returns:
            Embeddings array of shape (n_intervals, embedding_dim) as bfloat16.
        """
        start = self._metadata.offsets[idx]
        length = self._metadata.lengths[idx]
        return self._embeddings[start : start + length]

    def get_intervals(self, idx: int) -> NDArray[np.int32]:
        """Return intervals slice for file at index.

        Args:
            idx: Index of the file (matches bed_names ordering).

        Returns:
            Intervals array of shape (n_intervals, 3) as int32.
        """
        start = self._metadata.offsets[idx]
        length = self._metadata.lengths[idx]
        return self._intervals[start : start + length]

    @staticmethod
    def build_from_files(
        bed_names: list[str],
        embedding_dir: Pathish,
        bed_dir: Pathish,
        output_dir: Pathish,
    ) -> ContrastiveMemmapData:
        """Build memmap from zarr embeddings and BED files.

        Args:
            bed_names: List of BED file names (without path/extension).
                Will be sorted for consistent ordering.
            embedding_dir: Directory containing zarr arrays of embeddings.
            bed_dir: Directory containing .bed files.
            output_dir: Directory to write memmap files.

        Returns:
            ContrastiveMemmapData instance for the created memmap.
        """
        embedding_dir = Path(embedding_dir)
        bed_dir = Path(bed_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sort bed names for consistent ordering
        sorted_names = sorted(bed_names)

        # First pass: compute sizes and embedding dim
        lengths: list[int] = []
        embedding_dim: int | None = None

        for name in tqdm(sorted_names, desc="Scanning files"):
            zarr_path = embedding_dir / f"{name}.zarr"
            zarr_array = zarr.open_array(zarr_path, mode="r")

            n_intervals = zarr_array.shape[0]
            lengths.append(n_intervals)

            if embedding_dim is None:
                embedding_dim = zarr_array.shape[1]
            elif embedding_dim != zarr_array.shape[1]:
                raise ValueError(
                    f"Inconsistent embedding dim: expected {embedding_dim}, "
                    f"got {zarr_array.shape[1]} for {name}"
                )

        if embedding_dim is None:
            raise ValueError("No files to process")

        # Compute offsets
        total_intervals = sum(lengths)
        offsets = [0]
        for length in lengths[:-1]:
            offsets.append(offsets[-1] + length)

        # Compute byte layout: embeddings first, then intervals
        # bfloat16 = 2 bytes per element
        embeddings_bytes = total_intervals * embedding_dim * 2
        # int32 = 4 bytes per element, 3 columns
        intervals_bytes = total_intervals * 3 * 4
        total_data_bytes = embeddings_bytes + intervals_bytes

        # Create metadata
        metadata = ContrastiveMemmapMetadata(
            version=CURRENT_VERSION,
            embedding_dim=embedding_dim,
            total_intervals=total_intervals,
            num_files=len(sorted_names),
            intervals_byte_offset=embeddings_bytes,
            bed_names=sorted_names,
            offsets=offsets,
            lengths=lengths,
        )

        # Create single unified memmap file
        data_path = output_dir / DATA_FILENAME
        raw_mmap = np.memmap(data_path, dtype=np.uint8, mode="w+", shape=(total_data_bytes,))

        # Create views for writing
        embeddings_view = np.ndarray(
            (total_intervals, embedding_dim),
            dtype=_bf16_dtype,
            buffer=raw_mmap[:embeddings_bytes],
        )
        intervals_view = np.ndarray(
            (total_intervals, 3),
            dtype=np.int32,
            buffer=raw_mmap[embeddings_bytes:],
        )

        # Second pass: write data
        for i, name in enumerate(tqdm(sorted_names, desc="Writing files")):
            offset = offsets[i]
            length = lengths[i]

            # Load embeddings from zarr and convert to bfloat16
            zarr_path = embedding_dir / f"{name}.zarr"
            zarr_array = zarr.open_array(zarr_path, mode="r")
            embeddings = np.asarray(zarr_array[:], dtype=_bf16_dtype)
            embeddings_view[offset : offset + length] = embeddings

            # Load intervals from BED file
            bed_path = bed_dir / f"{name}.bed"
            intervals = np.array(load_bed_array(bed_path), dtype=np.int32)
            intervals_view[offset : offset + length] = intervals

        # Flush to disk
        raw_mmap.flush()

        # Save metadata
        memmap_data = ContrastiveMemmapData(output_dir, mode="w+")
        memmap_data._metadata = metadata
        memmap_data._embeddings = embeddings_view
        memmap_data._intervals = intervals_view
        memmap_data._save_metadata()

        tqdm.write(f"Built memmap: {total_intervals:,} intervals from {len(sorted_names)} files")

        return memmap_data
