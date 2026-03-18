"""Contrastive Learning Data Loader for BED files."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import zarr
from jaxtyping import Array, Float, Int, PRNGKeyArray
from tqdm import tqdm

from giggleml.data.intervals import load_bed_array
from giggleml.train.similarity_graph.community_subgraph import (
    community_subgraph_iterator,
)
from giggleml.train.similarity_graph.similarity_graph import SimilarityGraph
from giggleml.utils.file_utils import Pathish


@dataclass(frozen=True)
class BedFileData:
    """Container for a single BED file's data.

    Attributes:
        node_idx: Index of this file in the similarity graph.
        embeddings: HyenaDNA embeddings of shape (n_intervals, embedding_dim).
        intervals: Genomic intervals of shape (n_intervals, 3) as [chrom_idx, start, end].
    """

    node_idx: int
    embeddings: Float[Array, "n edim"]
    intervals: Int[Array, "n 3"]


@dataclass(frozen=True)
class ContrastiveBatch:
    """Batch of BED files with adjacency information.

    Attributes:
        bed_data: List of BedFileData objects for this batch.
        adjacency: Square adjacency matrix where entry [i, j] contains the edge
            type + 1 between nodes i and j, or 0 if no edge exists.
    """

    bed_data: list[BedFileData]
    adjacency: Int[Array, "batch batch"]


class ContrastiveDataLoader:
    """Data loader for contrastive learning on BED files.

    Maps graph node indices to file paths via ordered bed_names list.
    Uses community_subgraph_iterator for batch sampling.

    All BedFileData is cached in memory after first load for fast access.

    Args:
        graph: SimilarityGraph for sampling subgraphs.
        bed_names: List of BED file names (without path/extension)
            matching graph node indices.
        embedding_dir: Directory containing zarr arrays of embeddings.
        bed_dir: Directory containing .bed files.
        num_anchors: Number of anchor nodes per batch.
        neighbors_per_anchor: Number of neighbors per anchor.
        preload: If True, load all BED files into cache on init. If False, load lazily.
    """

    graph: SimilarityGraph
    bed_names: list[str]
    embedding_dir: Path
    bed_dir: Path
    num_anchors: int
    neighbors_per_anchor: int
    _cache: dict[int, BedFileData]

    def __init__(
        self,
        graph: SimilarityGraph,
        bed_names: list[str],
        embedding_dir: Pathish,
        bed_dir: Pathish,
        num_anchors: int,
        neighbors_per_anchor: int,
        *,
        preload: bool = False,
    ) -> None:
        self.graph = graph
        self.bed_names = sorted(bed_names)
        self.embedding_dir = Path(embedding_dir)
        self.bed_dir = Path(bed_dir)
        self.num_anchors = num_anchors
        self.neighbors_per_anchor = neighbors_per_anchor
        self._cache = {}

        if len(bed_names) != graph.n:
            raise ValueError(
                f"bed_names length ({len(bed_names)}) must match graph size ({graph.n})"
            )

        if preload:
            self._preload_all()

    def _preload_all(self) -> None:
        """Load all BED files into cache."""
        for idx in tqdm(range(len(self.bed_names)), desc="Preloading BED files"):
            if idx not in self._cache:
                self._cache[idx] = self._load_bed_uncached(idx)

    def _load_bed_uncached(self, node_idx: int) -> BedFileData:
        """Load BED file data from disk (no cache check)."""
        name = self.bed_names[node_idx]

        # Load embeddings from zarr
        zarr_path = self.embedding_dir / f"{name}.zarr"
        zarr_array = zarr.open_array(zarr_path, mode="r")
        embeddings = jnp.array(zarr_array[:])

        # Load intervals from BED file
        bed_path = self.bed_dir / f"{name}.bed"
        intervals = load_bed_array(bed_path)

        return BedFileData(
            node_idx=node_idx,
            embeddings=embeddings,
            intervals=intervals,
        )

    def _load_bed(self, node_idx: int) -> BedFileData:
        """Load BED file data for a given node index (cached)."""
        if node_idx not in self._cache:
            self._cache[node_idx] = self._load_bed_uncached(node_idx)
        return self._cache[node_idx]

    def cache_size(self) -> int:
        """Return the number of cached BED files."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()

    def iter_batches(self, key: PRNGKeyArray) -> Iterator[ContrastiveBatch]:
        """Iterate over batches of community subgraphs.

        Args:
            key: JAX PRNG key for reproducibility.

        Yields:
            ContrastiveBatch objects containing BED data and adjacency matrix.
        """
        for node_indices, adjacency in community_subgraph_iterator(
            self.graph, self.num_anchors, self.neighbors_per_anchor, key=key
        ):
            bed_data = [self._load_bed(int(idx)) for idx in node_indices]
            yield ContrastiveBatch(bed_data, jnp.array(adjacency))
