"""Similarity graph with discretized edge weights."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SimilarityGraph:
    """Graph with colored edges based on discretized similarity values.

    Takes a symmetric similarity matrix and bin thresholds to create
    a graph where edges are colored based on which bin their similarity
    value falls into.

    Args:
        similarity_matrix: n×n symmetric numpy array of similarity values.
        thresholds: Bin boundaries in ascending order. Values below thresholds[0]
            have no edge. A value is in bin i if thresholds[i] <= value < thresholds[i+1].
            Values >= thresholds[-1] are in the last bin.
    """

    _n: int
    _num_edge_types: int
    _thresholds: NDArray[np.floating]
    # node -> edge_type -> list of neighbors
    _adjacency: dict[int, dict[int, list[int]]]

    def __init__(
        self,
        similarity_matrix: NDArray[np.floating],
        thresholds: list[float] | NDArray[np.floating],
    ) -> None:
        if similarity_matrix.ndim != 2:
            raise ValueError("similarity_matrix must be 2D")
        if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("similarity_matrix must be square")
        if len(thresholds) == 0:
            raise ValueError("thresholds must not be empty")

        self._n = similarity_matrix.shape[0]
        self._thresholds = np.asarray(thresholds)
        self._num_edge_types = len(thresholds)

        # Build adjacency lists
        self._adjacency = {i: {} for i in range(self._n)}

        # Process upper triangle only (symmetric matrix)
        for i in range(self._n):
            for j in range(i + 1, self._n):
                value = float(similarity_matrix[i, j])
                bin_idx = self._get_bin(value)
                if bin_idx is not None:
                    # Add edge in both directions
                    self._adjacency[i].setdefault(bin_idx, []).append(j)
                    self._adjacency[j].setdefault(bin_idx, []).append(i)

    def _get_bin(self, value: float) -> int | None:
        """Get bin index for a value, or None if below first threshold."""
        if value < self._thresholds[0]:
            return None

        # Find the highest threshold that value exceeds
        for i in range(len(self._thresholds) - 1, -1, -1):
            if value >= self._thresholds[i]:
                return i
        return None

    @property
    def n(self) -> int:
        """Number of nodes."""
        return self._n

    @property
    def num_edge_types(self) -> int:
        """Number of edge types (bins)."""
        return self._num_edge_types

    @property
    def thresholds(self) -> NDArray[np.floating]:
        """Bin thresholds."""
        return self._thresholds

    def neighbors(self, node: int, edge_type: int | None = None) -> list[int]:
        """Get neighbors of a node, optionally filtered by edge type.

        Args:
            node: Node index.
            edge_type: If provided, only return neighbors connected by this edge type.
                       If None, return all neighbors.

        Returns:
            List of neighbor indices.
        """
        if node < 0 or node >= self._n:
            raise ValueError(f"Node {node} out of range [0, {self._n})")

        if edge_type is not None:
            return list(self._adjacency[node].get(edge_type, []))

        # Return all neighbors
        all_neighbors: list[int] = []
        for neighbors_list in self._adjacency[node].values():
            all_neighbors.extend(neighbors_list)
        return all_neighbors

    def edge_type(self, node_a: int, node_b: int) -> int | None:
        """Get the edge type between two nodes, or None if no edge.

        Args:
            node_a: First node index.
            node_b: Second node index.

        Returns:
            Edge type (bin index) or None if no edge exists.
        """
        if node_a < 0 or node_a >= self._n:
            raise ValueError(f"Node {node_a} out of range [0, {self._n})")
        if node_b < 0 or node_b >= self._n:
            raise ValueError(f"Node {node_b} out of range [0, {self._n})")

        for et, neighbors_list in self._adjacency[node_a].items():
            if node_b in neighbors_list:
                return et
        return None

    def has_edge(self, node_a: int, node_b: int) -> bool:
        """Check if an edge exists between two nodes."""
        return self.edge_type(node_a, node_b) is not None

    def degree(self, node: int, edge_type: int | None = None) -> int:
        """Get the degree of a node, optionally filtered by edge type.

        Args:
            node: Node index.
            edge_type: If provided, count only edges of this type.

        Returns:
            Number of edges.
        """
        return len(self.neighbors(node, edge_type))
