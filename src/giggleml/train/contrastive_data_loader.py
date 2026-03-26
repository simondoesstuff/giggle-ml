"""Contrastive Learning Data Loader for BED files."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import zarr
from jaxtyping import PRNGKeyArray
from numpy.typing import NDArray
from tqdm import tqdm

from giggleml.data.contrastive_memmap import ContrastiveMemmapData
from giggleml.data.intervals import load_bed_array
from giggleml.train.similarity_graph.community_subgraph import (
    community_subgraph_iterator,
)
from giggleml.train.similarity_graph.similarity_graph import SimilarityGraph
from giggleml.utils.file_utils import Pathish, possibly_gzipped

# Type alias for numpy arrays in host memory
type HostArray = NDArray[np.generic]


@dataclass(frozen=True)
class BedFileData:
    """Container for a single BED file's data.

    Data is stored as numpy arrays in host memory to minimize VRAM usage.
    Conversion to JAX arrays happens at batch time.

    Attributes:
        node_idx: Index of this file in the cache/graph.
        embeddings: HyenaDNA embeddings of shape (n_intervals, embedding_dim).
        intervals: Genomic intervals of shape (n_intervals, 3) as [chrom_idx, start, end].
    """

    node_idx: int
    embeddings: HostArray
    intervals: HostArray


@dataclass(frozen=True)
class ContrastiveBatch:
    """Batch of BED files with adjacency information.

    All data is stored as numpy arrays in host memory. Transfer to device
    happens in the training loop via jax.device_put.

    Attributes:
        bed_data: List of BedFileData objects for this batch.
        adjacency: Square adjacency matrix where entry [i, j] contains the edge
            type + 1 between nodes i and j, or 0 if no edge exists.
    """

    bed_data: list[BedFileData]
    adjacency: NDArray[np.int32]


class BedFileCache:
    """Cache for BED file data (embeddings + intervals).

    Handles loading from zarr/BED files or memmap, with in-memory caching.
    Can be shared between training data loader and evaluation callbacks.

    Index Mapping for Train/Test/Val Splits:
        When using a memmap built from all files but operating on a subset,
        the index_map bridges node indices (0..n_subset-1) to file indices
        in the full bed_names/memmap.

        - Node index i corresponds to file index_map[i] in bed_names/memmap
        - Example: index_map=[5, 12, 23] means node 0 loads file 5

        Without index_map, node indices map directly to file indices.

    Args:
        bed_names: List of BED file names (without path/extension). Must be the
            full sorted list when using memmap.
        embedding_dir: Directory containing zarr arrays of embeddings.
        bed_dir: Directory containing .bed files.
        memmap_dir: Optional directory containing pre-built memmap files.
        index_map: Mapping from node indices to file indices. If None, uses
            identity mapping for all bed_names.
        preload: If True, load all files into cache on init.
    """

    bed_names: list[str]
    embedding_dir: Path
    bed_dir: Path
    _cache: dict[int, BedFileData]
    _memmap: ContrastiveMemmapData | None
    _index_map: list[int] | None
    _n_files: int

    def __init__(
        self,
        bed_names: list[str],
        embedding_dir: Pathish,
        bed_dir: Pathish,
        *,
        memmap_dir: Pathish | None = None,
        index_map: list[int] | None = None,
        preload: bool = False,
    ) -> None:
        self.bed_names = sorted(bed_names)
        self.embedding_dir = Path(embedding_dir)
        self.bed_dir = Path(bed_dir)
        self._cache = {}
        self._index_map = index_map
        self._n_files = len(index_map) if index_map is not None else len(bed_names)

        # Load memmap if provided
        if memmap_dir is not None:
            self._memmap = ContrastiveMemmapData(memmap_dir, mode="r")
            if self._memmap.bed_names != self.bed_names:
                raise ValueError(
                    f"bed_names mismatch: memmap has {len(self._memmap.bed_names)} files, "
                    f"expected {len(self.bed_names)} files with matching names"
                )
        else:
            self._memmap = None

        if preload:
            self._preload_all()

    @property
    def n_files(self) -> int:
        """Number of files managed by this cache."""
        return self._n_files

    def _file_idx(self, node_idx: int) -> int:
        """Map node index to file index."""
        if self._index_map is not None:
            return self._index_map[node_idx]
        return node_idx

    def _preload_all(self) -> None:
        """Load all BED files into cache."""
        desc = "Preloading from memmap" if self._memmap else "Preloading BED files"
        for node_idx in tqdm(range(self._n_files), desc=desc):
            if node_idx not in self._cache:
                self._cache[node_idx] = self._load_uncached(node_idx)

    def _load_uncached(self, node_idx: int) -> BedFileData:
        """Load BED file data from disk (no cache check)."""
        file_idx = self._file_idx(node_idx)

        if self._memmap is not None:
            embeddings = np.array(self._memmap.get_embeddings(file_idx))
            intervals = np.array(self._memmap.get_intervals(file_idx))
        else:
            name = self.bed_names[file_idx]
            zarr_path = self.embedding_dir / f"{name}.zarr"
            zarr_array = zarr.open_array(zarr_path, mode="r")
            embeddings = np.asarray(zarr_array[:], dtype=ml_dtypes.bfloat16)

            bed_path = possibly_gzipped(self.bed_dir / f"{name}.bed")
            intervals = load_bed_array(bed_path)

        return BedFileData(
            node_idx=node_idx,
            embeddings=embeddings,
            intervals=intervals,
        )

    def get(self, node_idx: int) -> BedFileData:
        """Get BED file data for a node index (cached)."""
        if node_idx not in self._cache:
            self._cache[node_idx] = self._load_uncached(node_idx)
        return self._cache[node_idx]

    def get_all(self) -> list[BedFileData]:
        """Return all BedFileData sorted by node index.

        Loads any uncached files first.
        """
        for i in range(self._n_files):
            if i not in self._cache:
                self._cache[i] = self._load_uncached(i)
        return [self._cache[i] for i in range(self._n_files)]

    def cache_size(self) -> int:
        """Return the number of cached BED files."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()


class ContrastiveDataLoader:
    """Data loader for contrastive learning batch iteration.

    Iterates over community subgraph batches using a shared BedFileCache.
    Supports optional downsampling of large files during batch iteration.

    Args:
        cache: BedFileCache containing the BED file data.
        graph: SimilarityGraph for sampling subgraphs.
        num_anchors: Number of anchor nodes per batch.
        neighbors_per_anchor: Number of neighbors per anchor.
        max_intervals: Maximum intervals per file. Files exceeding this are
            randomly downsampled per-batch. None disables capping.
        index_map: Mapping from graph node indices to cache indices. Required
            when graph is built on a subset (e.g., train split) but cache
            contains all files. If None, graph nodes map directly to cache indices.
    """

    _cache: BedFileCache
    _graph: SimilarityGraph
    _num_anchors: int
    _neighbors_per_anchor: int
    _max_intervals: int | None
    _index_map: list[int] | None

    def __init__(
        self,
        cache: BedFileCache,
        graph: SimilarityGraph,
        num_anchors: int,
        neighbors_per_anchor: int,
        *,
        max_intervals: int | None = None,
        index_map: list[int] | None = None,
    ) -> None:
        expected_size = len(index_map) if index_map is not None else cache.n_files
        if graph.n != expected_size:
            raise ValueError(
                f"graph size ({graph.n}) must match "
                f"{'index_map length' if index_map else 'cache n_files'} ({expected_size})"
            )
        self._cache = cache
        self._graph = graph
        self._num_anchors = num_anchors
        self._neighbors_per_anchor = neighbors_per_anchor
        self._max_intervals = max_intervals
        self._index_map = index_map

    def _cache_idx(self, graph_node: int) -> int:
        """Map graph node index to cache index."""
        if self._index_map is not None:
            return self._index_map[graph_node]
        return graph_node

    def _downsample(
        self, data: BedFileData, max_intervals: int, key: PRNGKeyArray
    ) -> BedFileData:
        """Downsample a BedFileData to max_intervals if it exceeds the cap."""
        n = data.embeddings.shape[0]
        if n <= max_intervals:
            return data

        seed = np.asarray(jax.random.key_data(key))
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, size=max_intervals, replace=False)

        return BedFileData(
            node_idx=data.node_idx,
            embeddings=data.embeddings[indices],
            intervals=data.intervals[indices],
        )

    def iter_batches(self, key: PRNGKeyArray) -> Iterator[ContrastiveBatch]:
        """Iterate over batches of community subgraphs.

        Args:
            key: JAX PRNG key for reproducibility.

        Yields:
            ContrastiveBatch objects containing BED data and adjacency matrix.
        """
        key, subgraph_key = jax.random.split(key)
        for node_indices, adjacency in community_subgraph_iterator(
            self._graph,
            self._num_anchors,
            self._neighbors_per_anchor,
            key=subgraph_key,
        ):
            key, *file_keys = jax.random.split(key, len(node_indices) + 1)
            bed_data = []
            for i, graph_node in enumerate(node_indices):
                cache_idx = self._cache_idx(int(graph_node))
                data = self._cache.get(cache_idx)
                if self._max_intervals is not None:
                    data = self._downsample(data, self._max_intervals, file_keys[i])
                bed_data.append(data)
            yield ContrastiveBatch(bed_data, np.asarray(adjacency, dtype=np.int32))
