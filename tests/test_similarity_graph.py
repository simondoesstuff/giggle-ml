"""Tests for giggleml.train.similarity_graph.similarity_graph module."""

import numpy as np
import pytest

import itertools

import jax

from giggleml.train.similarity_graph.similarity_graph import SimilarityGraph
from giggleml.train.similarity_graph.community_subgraph import (
    sample_community_subgraph,
    community_subgraph_iterator,
)


class TestSimilarityGraphInit:
    """Tests for SimilarityGraph initialization."""

    def test_rejects_non_2d_matrix(self):
        with pytest.raises(ValueError, match="must be 2D"):
            SimilarityGraph(np.array([1, 2, 3]), thresholds=[0.5])

    def test_rejects_non_square_matrix(self):
        with pytest.raises(ValueError, match="must be square"):
            SimilarityGraph(np.array([[1, 2, 3], [4, 5, 6]]), thresholds=[0.5])

    def test_rejects_empty_thresholds(self):
        matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
        with pytest.raises(ValueError, match="thresholds must not be empty"):
            SimilarityGraph(matrix, thresholds=[])

    def test_properties(self):
        matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
        graph = SimilarityGraph(matrix, thresholds=[0.3, 0.7])

        assert graph.n == 2
        assert graph.num_edge_types == 2
        assert np.array_equal(graph.thresholds, [0.3, 0.7])


class TestBinning:
    """Tests for edge binning logic."""

    def test_values_below_first_threshold_have_no_edge(self):
        matrix = np.array([
            [0.0, 0.2, 0.3],
            [0.2, 0.0, 0.1],
            [0.3, 0.1, 0.0],
        ])
        graph = SimilarityGraph(matrix, thresholds=[0.5])

        # All values < 0.5, so no edges
        assert graph.neighbors(0) == []
        assert graph.neighbors(1) == []
        assert graph.neighbors(2) == []

    def test_values_at_threshold_are_included(self):
        matrix = np.array([
            [0.0, 0.5],
            [0.5, 0.0],
        ])
        graph = SimilarityGraph(matrix, thresholds=[0.5])

        # 0.5 >= 0.5, so edge exists
        assert graph.neighbors(0) == [1]
        assert graph.neighbors(1) == [0]
        assert graph.edge_type(0, 1) == 0

    def test_values_in_correct_bins(self):
        # Thresholds: [0.3, 0.6, 0.9]
        # Bin 0: 0.3 <= x < 0.6
        # Bin 1: 0.6 <= x < 0.9
        # Bin 2: x >= 0.9
        matrix = np.array([
            [0.0, 0.4, 0.7, 0.95],
            [0.4, 0.0, 0.2, 0.6],
            [0.7, 0.2, 0.0, 0.9],
            [0.95, 0.6, 0.9, 0.0],
        ])
        graph = SimilarityGraph(matrix, thresholds=[0.3, 0.6, 0.9])

        # Check edge types
        assert graph.edge_type(0, 1) == 0  # 0.4 in bin 0
        assert graph.edge_type(0, 2) == 1  # 0.7 in bin 1
        assert graph.edge_type(0, 3) == 2  # 0.95 in bin 2
        assert graph.edge_type(1, 2) is None  # 0.2 < 0.3, no edge
        assert graph.edge_type(1, 3) == 1  # 0.6 in bin 1
        assert graph.edge_type(2, 3) == 2  # 0.9 in bin 2

    def test_single_threshold(self):
        matrix = np.array([
            [0.0, 0.4, 0.8],
            [0.4, 0.0, 0.6],
            [0.8, 0.6, 0.0],
        ])
        graph = SimilarityGraph(matrix, thresholds=[0.5])

        # Only values >= 0.5 have edges, all in bin 0
        assert graph.edge_type(0, 1) is None  # 0.4 < 0.5
        assert graph.edge_type(0, 2) == 0  # 0.8 >= 0.5
        assert graph.edge_type(1, 2) == 0  # 0.6 >= 0.5


class TestNeighbors:
    """Tests for neighbor lookup."""

    @pytest.fixture
    def sample_graph(self):
        matrix = np.array([
            [0.0, 0.4, 0.7, 0.95],
            [0.4, 0.0, 0.2, 0.6],
            [0.7, 0.2, 0.0, 0.9],
            [0.95, 0.6, 0.9, 0.0],
        ])
        return SimilarityGraph(matrix, thresholds=[0.3, 0.6, 0.9])

    def test_all_neighbors(self, sample_graph):
        # Node 0 has edges to 1 (bin 0), 2 (bin 1), 3 (bin 2)
        neighbors = sample_graph.neighbors(0)
        assert sorted(neighbors) == [1, 2, 3]

    def test_neighbors_by_edge_type(self, sample_graph):
        # Node 0, edge type 0: only node 1
        assert sample_graph.neighbors(0, edge_type=0) == [1]
        # Node 0, edge type 1: only node 2
        assert sample_graph.neighbors(0, edge_type=1) == [2]
        # Node 0, edge type 2: only node 3
        assert sample_graph.neighbors(0, edge_type=2) == [3]

    def test_neighbors_empty_edge_type(self, sample_graph):
        # Node 1 has no edges of type 2
        assert sample_graph.neighbors(1, edge_type=2) == []

    def test_neighbors_out_of_range(self, sample_graph):
        with pytest.raises(ValueError, match="out of range"):
            sample_graph.neighbors(-1)
        with pytest.raises(ValueError, match="out of range"):
            sample_graph.neighbors(4)

    def test_neighbors_returns_copy(self, sample_graph):
        neighbors1 = sample_graph.neighbors(0, edge_type=0)
        neighbors2 = sample_graph.neighbors(0, edge_type=0)
        assert neighbors1 == neighbors2
        assert neighbors1 is not neighbors2


class TestEdgeType:
    """Tests for edge_type lookup."""

    def test_edge_type_symmetric(self):
        matrix = np.array([
            [0.0, 0.5],
            [0.5, 0.0],
        ])
        graph = SimilarityGraph(matrix, thresholds=[0.3])

        assert graph.edge_type(0, 1) == 0
        assert graph.edge_type(1, 0) == 0

    def test_edge_type_no_edge(self):
        matrix = np.array([
            [0.0, 0.1],
            [0.1, 0.0],
        ])
        graph = SimilarityGraph(matrix, thresholds=[0.5])

        assert graph.edge_type(0, 1) is None

    def test_edge_type_out_of_range(self):
        matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
        graph = SimilarityGraph(matrix, thresholds=[0.3])

        with pytest.raises(ValueError, match="out of range"):
            graph.edge_type(-1, 0)
        with pytest.raises(ValueError, match="out of range"):
            graph.edge_type(0, 2)


class TestHasEdge:
    """Tests for has_edge method."""

    def test_has_edge_true(self):
        matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
        graph = SimilarityGraph(matrix, thresholds=[0.3])

        assert graph.has_edge(0, 1) is True

    def test_has_edge_false(self):
        matrix = np.array([[0.0, 0.1], [0.1, 0.0]])
        graph = SimilarityGraph(matrix, thresholds=[0.5])

        assert graph.has_edge(0, 1) is False


class TestDegree:
    """Tests for degree method."""

    def test_degree_all_edges(self):
        matrix = np.array([
            [0.0, 0.4, 0.7],
            [0.4, 0.0, 0.8],
            [0.7, 0.8, 0.0],
        ])
        graph = SimilarityGraph(matrix, thresholds=[0.3])

        assert graph.degree(0) == 2
        assert graph.degree(1) == 2
        assert graph.degree(2) == 2

    def test_degree_by_edge_type(self):
        matrix = np.array([
            [0.0, 0.4, 0.7, 0.95],
            [0.4, 0.0, 0.2, 0.6],
            [0.7, 0.2, 0.0, 0.9],
            [0.95, 0.6, 0.9, 0.0],
        ])
        graph = SimilarityGraph(matrix, thresholds=[0.3, 0.6, 0.9])

        # Node 0: bin 0 -> 1 edge, bin 1 -> 1 edge, bin 2 -> 1 edge
        assert graph.degree(0, edge_type=0) == 1
        assert graph.degree(0, edge_type=1) == 1
        assert graph.degree(0, edge_type=2) == 1
        assert graph.degree(0) == 3

    def test_degree_isolated_node(self):
        matrix = np.array([
            [0.0, 0.1, 0.1],
            [0.1, 0.0, 0.5],
            [0.1, 0.5, 0.0],
        ])
        graph = SimilarityGraph(matrix, thresholds=[0.3])

        # Node 0 has no edges (0.1 < 0.3)
        assert graph.degree(0) == 0


class TestSampleCommunitySubgraph:
    """Tests for sample_community_subgraph function."""

    @pytest.fixture
    def sample_graph(self):
        matrix = np.array([
            [0.0, 0.4, 0.7, 0.95],
            [0.4, 0.0, 0.2, 0.6],
            [0.7, 0.2, 0.0, 0.9],
            [0.95, 0.6, 0.9, 0.0],
        ])
        return SimilarityGraph(matrix, thresholds=[0.3, 0.6, 0.9])

    def test_basic_sampling(self, sample_graph):
        key = jax.random.key(42)
        node_indices, adj = sample_community_subgraph(
            sample_graph, num_anchors=2, neighbors_per_anchor=1, key=key
        )

        # Should have at least the anchors
        assert len(node_indices) >= 2
        # Adjacency should be square
        assert adj.shape == (len(node_indices), len(node_indices))
        # Adjacency should be symmetric
        assert np.array_equal(adj, adj.T)
        # Diagonal should be zero (no self-loops)
        assert np.all(np.diag(adj) == 0)

    def test_edge_types_are_one_indexed(self, sample_graph):
        key = jax.random.key(123)
        node_indices, adj = sample_community_subgraph(
            sample_graph, num_anchors=4, neighbors_per_anchor=3, key=key
        )

        # All 4 nodes should be sampled
        assert len(node_indices) == 4

        # Check specific edge types (original edge_type + 1)
        idx_map = {node: i for i, node in enumerate(node_indices)}

        # Edge (0, 1) has type 0 in original -> should be 1 in adj
        if 0 in idx_map and 1 in idx_map:
            assert adj[idx_map[0], idx_map[1]] == 1  # bin 0 + 1

        # Edge (0, 2) has type 1 in original -> should be 2 in adj
        if 0 in idx_map and 2 in idx_map:
            assert adj[idx_map[0], idx_map[2]] == 2  # bin 1 + 1

        # Edge (0, 3) has type 2 in original -> should be 3 in adj
        if 0 in idx_map and 3 in idx_map:
            assert adj[idx_map[0], idx_map[3]] == 3  # bin 2 + 1

    def test_no_edge_is_zero(self, sample_graph):
        key = jax.random.key(456)
        node_indices, adj = sample_community_subgraph(
            sample_graph, num_anchors=4, neighbors_per_anchor=3, key=key
        )

        idx_map = {node: i for i, node in enumerate(node_indices)}

        # Edge (1, 2) has no edge (0.2 < 0.3) -> should be 0 in adj
        if 1 in idx_map and 2 in idx_map:
            assert adj[idx_map[1], idx_map[2]] == 0

    def test_captures_all_edges_between_sampled_nodes(self, sample_graph):
        key = jax.random.key(789)
        node_indices, adj = sample_community_subgraph(
            sample_graph, num_anchors=4, neighbors_per_anchor=3, key=key
        )

        # Verify all edges between sampled nodes are captured
        for i, node_i in enumerate(node_indices):
            for j, node_j in enumerate(node_indices):
                if i != j:
                    original_edge = sample_graph.edge_type(node_i, node_j)
                    expected = 0 if original_edge is None else original_edge + 1
                    assert adj[i, j] == expected

    def test_deterministic_with_key(self, sample_graph):
        key1 = jax.random.key(42)
        key2 = jax.random.key(42)

        nodes1, adj1 = sample_community_subgraph(
            sample_graph, num_anchors=2, neighbors_per_anchor=1, key=key1
        )
        nodes2, adj2 = sample_community_subgraph(
            sample_graph, num_anchors=2, neighbors_per_anchor=1, key=key2
        )

        assert np.array_equal(nodes1, nodes2)
        assert np.array_equal(adj1, adj2)

    def test_invalid_num_anchors(self, sample_graph):
        key = jax.random.key(0)
        with pytest.raises(ValueError, match="num_anchors must be positive"):
            sample_community_subgraph(sample_graph, num_anchors=0, neighbors_per_anchor=1, key=key)

        with pytest.raises(ValueError, match="exceeds graph size"):
            sample_community_subgraph(sample_graph, num_anchors=10, neighbors_per_anchor=1, key=key)

    def test_invalid_neighbors_per_anchor(self, sample_graph):
        key = jax.random.key(0)
        with pytest.raises(ValueError, match="neighbors_per_anchor must be non-negative"):
            sample_community_subgraph(sample_graph, num_anchors=2, neighbors_per_anchor=-1, key=key)

    def test_zero_neighbors_per_anchor(self, sample_graph):
        key = jax.random.key(42)
        node_indices, _ = sample_community_subgraph(
            sample_graph, num_anchors=2, neighbors_per_anchor=0, key=key
        )

        # Should only have the anchors
        assert len(node_indices) == 2

    def test_node_indices_are_sorted(self, sample_graph):
        key = jax.random.key(42)
        node_indices, _ = sample_community_subgraph(
            sample_graph, num_anchors=3, neighbors_per_anchor=2, key=key
        )

        # Node indices should be sorted
        assert np.array_equal(node_indices, np.sort(node_indices))


class TestCommunitySubgraphIterator:
    """Tests for community_subgraph_iterator function."""

    @pytest.fixture
    def sample_graph(self):
        matrix = np.array([
            [0.0, 0.4, 0.7, 0.95],
            [0.4, 0.0, 0.2, 0.6],
            [0.7, 0.2, 0.0, 0.9],
            [0.95, 0.6, 0.9, 0.0],
        ])
        return SimilarityGraph(matrix, thresholds=[0.3, 0.6, 0.9])

    def test_yields_valid_subgraphs(self, sample_graph):
        key = jax.random.key(42)
        iterator = community_subgraph_iterator(
            sample_graph, num_anchors=2, neighbors_per_anchor=1, key=key
        )

        for nodes, adj in itertools.islice(iterator, 5):
            assert len(nodes) >= 2
            assert adj.shape == (len(nodes), len(nodes))
            assert np.array_equal(adj, adj.T)

    def test_deterministic_iteration(self, sample_graph):
        key1 = jax.random.key(123)
        key2 = jax.random.key(123)

        iter1 = community_subgraph_iterator(
            sample_graph, num_anchors=2, neighbors_per_anchor=1, key=key1
        )
        iter2 = community_subgraph_iterator(
            sample_graph, num_anchors=2, neighbors_per_anchor=1, key=key2
        )

        for (nodes1, adj1), (nodes2, adj2) in itertools.islice(zip(iter1, iter2), 10):
            assert np.array_equal(nodes1, nodes2)
            assert np.array_equal(adj1, adj2)

    def test_different_samples_each_iteration(self, sample_graph):
        key = jax.random.key(999)
        iterator = community_subgraph_iterator(
            sample_graph, num_anchors=2, neighbors_per_anchor=1, key=key
        )

        samples = list(itertools.islice(iterator, 20))

        # With 20 samples, we should see some variation (not all identical)
        # Check that not all node_indices are the same
        first_nodes = samples[0][0]
        has_different = any(
            not np.array_equal(nodes, first_nodes) for nodes, _ in samples[1:]
        )
        assert has_different, "Iterator should produce different samples"
