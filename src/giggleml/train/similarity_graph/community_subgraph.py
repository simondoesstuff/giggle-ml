"""Community-based subgraph sampling from a SimilarityGraph."""

from __future__ import annotations

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray
from numpy.typing import NDArray

from .similarity_graph import SimilarityGraph


def sample_community_subgraph(
    graph: SimilarityGraph,
    num_anchors: int,
    neighbors_per_anchor: int,
    *,
    key: PRNGKeyArray,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Sample a community-structured subgraph from a SimilarityGraph.

    Samples anchor nodes at random, then for each anchor samples some of its
    neighbors. Returns an adjacency matrix capturing all edges between the
    sampled nodes.

    Args:
        graph: The SimilarityGraph to sample from.
        num_anchors: Number of anchor nodes to sample.
        neighbors_per_anchor: Number of neighbors to sample per anchor.
        key: JAX PRNG key for reproducibility.

    Returns:
        A tuple of (node_indices, adjacency_matrix) where:
        - node_indices: 1D array of the original node indices that were sampled.
        - adjacency_matrix: Square matrix where entry [i, j] contains the edge
          type + 1 between sampled nodes i and j, or 0 if no edge exists.
          Edge types are 1-indexed in the output (0 reserved for no edge).
    """
    if num_anchors <= 0:
        raise ValueError("num_anchors must be positive")
    if neighbors_per_anchor < 0:
        raise ValueError("neighbors_per_anchor must be non-negative")
    if num_anchors > graph.n:
        raise ValueError(f"num_anchors ({num_anchors}) exceeds graph size ({graph.n})")

    # Convert JAX key to numpy RNG for indexing operations
    seed = int(jax.random.bits(key, dtype=jnp.uint32))
    rng = np.random.default_rng(seed)

    # Sample anchor nodes
    anchors = rng.choice(graph.n, size=num_anchors, replace=False)

    # Collect all sampled nodes (anchors + their sampled neighbors)
    sampled_nodes: set[int] = set(anchors)

    for anchor in anchors:
        all_neighbors = graph.neighbors(anchor)
        if len(all_neighbors) > 0:
            k = min(neighbors_per_anchor, len(all_neighbors))
            if k > 0:
                chosen = rng.choice(all_neighbors, size=k, replace=False)
                sampled_nodes.update(chosen)

    # Convert to sorted array for consistent ordering
    node_indices = np.array(sorted(sampled_nodes), dtype=np.intp)
    num_nodes = len(node_indices)

    # Build adjacency matrix with edge types
    # 0 = no edge, 1+ = edge type (original edge_type + 1)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.intp)

    # Fill in all edges between sampled nodes
    for i, node_i in enumerate(node_indices):
        for j, node_j in enumerate(node_indices[i + 1 :], start=i + 1):
            edge_t = graph.edge_type(node_i, node_j)
            if edge_t is not None:
                # Store edge type + 1 (so 0 means no edge)
                adjacency[i, j] = edge_t + 1
                adjacency[j, i] = edge_t + 1

    return node_indices, adjacency


def community_subgraph_iterator(
    graph: SimilarityGraph,
    num_anchors: int,
    neighbors_per_anchor: int,
    *,
    key: PRNGKeyArray,
) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Yield community-structured subgraphs indefinitely.

    Creates an infinite iterator that samples community subgraphs from the
    given SimilarityGraph. Each iteration splits the key to ensure
    reproducibility while generating different samples.

    Args:
        graph: The SimilarityGraph to sample from.
        num_anchors: Number of anchor nodes to sample per subgraph.
        neighbors_per_anchor: Number of neighbors to sample per anchor.
        key: JAX PRNG key for reproducibility.

    Yields:
        Tuples of (node_indices, adjacency_matrix) as from sample_community_subgraph.

    Example:
        >>> key = jax.random.key(42)
        >>> iterator = community_subgraph_iterator(graph, 10, 3, key=key)
        >>> for nodes, adj in itertools.islice(iterator, 100):
        ...     # Process subgraph
        ...     pass
    """
    while True:
        key, subkey = jax.random.split(key)
        yield sample_community_subgraph(
            graph, num_anchors, neighbors_per_anchor, key=subkey
        )
