"""Tests for giggleml.evaluation.ndcg module."""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from giggleml.evaluation.ndcg import (
    NDCGResult,
    compute_ndcg_at_k,
    compute_weighted_relevances,
    create_ndcg_callback,
    evaluate_ndcg,
)


class TestComputeWeightedRelevances:
    """Tests for compute_weighted_relevances function."""

    def test_basic_binning(self, tmp_path):
        """Test that values are correctly binned and weighted."""
        from giggleml.data.similarity_matrix import SimilarityMatrix

        # Create a simple similarity matrix
        matrix = SimilarityMatrix(tmp_path / "sim.mat", n=3, mode="w+")
        matrix[0, 1] = 0.05  # Below first threshold
        matrix[0, 2] = 0.15  # In first bin
        matrix[1, 2] = 0.35  # In second bin
        matrix.flush()

        thresholds = (0.1, 0.3, 0.5)
        weights = (0.5, 1.0, 2.0)

        relevances = compute_weighted_relevances(
            matrix, thresholds, weights, indices=None
        )

        # Below threshold -> weight 0
        assert relevances[0, 1] == 0.0
        # In first bin (0.1 <= 0.15 < 0.3) -> weight 0.5
        assert relevances[0, 2] == pytest.approx(0.5)
        # In second bin (0.3 <= 0.35 < 0.5) -> weight 1.0
        assert relevances[1, 2] == pytest.approx(1.0)

    def test_subset_indices(self, tmp_path):
        """Test extraction of subset using indices."""
        from giggleml.data.similarity_matrix import SimilarityMatrix

        # Create a 5x5 matrix
        matrix = SimilarityMatrix(tmp_path / "sim.mat", n=5, mode="w+")
        for i in range(5):
            for j in range(5):
                matrix[i, j] = (i + j) * 0.1
        matrix.flush()

        thresholds = (0.1,)
        weights = (1.0,)

        # Extract subset [1, 3]
        relevances = compute_weighted_relevances(
            matrix, thresholds, weights, indices=[1, 3]
        )

        # Should be 2x2
        assert relevances.shape == (2, 2)
        # [1,1] = 0.2 >= 0.1 -> weight 1.0
        assert relevances[0, 0] == pytest.approx(1.0)
        # [1,3] = 0.4 >= 0.1 -> weight 1.0
        assert relevances[0, 1] == pytest.approx(1.0)


class TestComputeNdcgAtK:
    """Tests for compute_ndcg_at_k function."""

    def test_perfect_ranking(self):
        """Test nDCG = 1.0 when model ranking matches ideal."""
        # Model similarities match ground truth order
        model_sims = jnp.array([0.0, 0.9, 0.8, 0.7, 0.6])
        relevances = jnp.array([0.0, 3.0, 2.0, 1.0, 0.5])
        k = 3

        ndcg = compute_ndcg_at_k(model_sims, relevances, k=k, exclude_idx=0)

        # Model ranks: [1, 2, 3, 4] (excluding 0)
        # Ideal ranks: [1, 2, 3, 4] (same)
        # Should be perfect nDCG
        assert ndcg == pytest.approx(1.0, abs=1e-5)

    def test_worst_ranking(self):
        """Test nDCG < 1.0 when ranking is inverted."""
        # Model similarities are inverted vs relevance
        model_sims = jnp.array([0.0, 0.1, 0.2, 0.3, 0.9])
        relevances = jnp.array([0.0, 3.0, 2.0, 1.0, 0.0])
        k = 3

        ndcg = compute_ndcg_at_k(model_sims, relevances, k=k, exclude_idx=0)

        # Model top-3 (excluding 0): [4, 3, 2] with relevances [0.0, 1.0, 2.0]
        # Ideal top-3: [1, 2, 3] with relevances [3.0, 2.0, 1.0]
        # DCG < IDCG, so nDCG < 1.0
        assert ndcg < 1.0

    def test_exclude_self(self):
        """Test that exclude_idx is properly excluded."""
        # Anchor at index 2 has high similarity but should be excluded
        model_sims = jnp.array([0.5, 0.6, 1.0, 0.7])
        relevances = jnp.array([1.0, 2.0, 10.0, 3.0])
        k = 2

        ndcg = compute_ndcg_at_k(model_sims, relevances, k=k, exclude_idx=2)

        # Index 2 should not appear in ranking
        # Top-2 excluding 2: [3, 1] with relevances [3.0, 2.0]
        # Ideal top-2 excluding 2: [3, 1] (same)
        assert ndcg == pytest.approx(1.0, abs=1e-5)

    def test_no_relevant_items(self):
        """Test nDCG = 0.0 when no relevant items exist."""
        model_sims = jnp.array([0.5, 0.6, 0.7, 0.8])
        relevances = jnp.array([0.0, 0.0, 0.0, 0.0])
        k = 2

        ndcg = compute_ndcg_at_k(model_sims, relevances, k=k, exclude_idx=0)

        # IDCG = 0, so nDCG = 0
        assert ndcg == 0.0


class TestEvaluateNdcg:
    """Tests for evaluate_ndcg function."""

    def test_basic_evaluation(self):
        """Test basic nDCG evaluation."""
        # Simple embeddings where distance correlates with index difference
        embeddings = jnp.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
        ])

        # Ground truth where nearby indices are more relevant
        ground_truth = np.array([
            [0.0, 1.0, 0.5, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)

        result = evaluate_ndcg(embeddings, ground_truth, anchor_indices=[0, 1], k=2)

        assert isinstance(result, NDCGResult)
        assert 0.0 <= result.mean_ndcg <= 1.0
        assert len(result.per_anchor_ndcg) == 2
        assert result.k == 2

    def test_all_anchors_default(self):
        """Test that all anchors are used when anchor_indices is None."""
        n = 5
        embeddings = jnp.eye(n)  # Identity -> self has highest similarity
        ground_truth = np.eye(n, dtype=np.float32)

        result = evaluate_ndcg(embeddings, ground_truth, anchor_indices=None, k=2)

        # Should evaluate all n anchors
        assert len(result.per_anchor_ndcg) == n

    def test_perfect_embeddings(self):
        """Test nDCG = 1.0 when embeddings perfectly match ground truth ranking."""
        # Embeddings designed so cosine sim matches ground truth order
        embeddings = jnp.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        ])

        # Ground truth: anchor 0's relevances match cosine similarity order
        # cos(0,1) > cos(0,2) > cos(0,3) -> relevances should be 3, 2, 1
        ground_truth = np.array([
            [0.0, 3.0, 2.0, 1.0],
            [3.0, 0.0, 2.0, 1.0],
            [2.0, 2.0, 0.0, 2.0],
            [1.0, 1.0, 2.0, 0.0],
        ], dtype=np.float32)

        result = evaluate_ndcg(embeddings, ground_truth, anchor_indices=[0], k=3)

        # Anchor 0: model ranks [1, 2, 3], ideal ranks [1, 2, 3] -> nDCG = 1.0
        assert result.mean_ndcg == pytest.approx(1.0, abs=1e-4)


class TestNDCGResult:
    """Tests for NDCGResult dataclass."""

    def test_dataclass_fields(self):
        """Test NDCGResult has expected fields."""
        result = NDCGResult(
            mean_ndcg=0.85,
            per_anchor_ndcg=np.array([0.8, 0.9], dtype=np.float32),
            k=10,
        )

        assert result.mean_ndcg == 0.85
        assert len(result.per_anchor_ndcg) == 2
        assert result.k == 10


class TestCreateNdcgCallback:
    """Tests for create_ndcg_callback function."""

    @pytest.fixture
    def mock_bed_data(self):
        """Create mock BedFileData objects."""
        from giggleml.train.contrastive_data_loader import BedFileData

        data = []
        for i in range(4):
            # Embeddings matching model's expected seq_dim (32)
            embeddings = np.random.randn(5, 32).astype(np.float32)
            intervals = np.array([[0, i * 100 + j * 10, i * 100 + j * 10 + 50] for j in range(5)], dtype=np.int32)
            data.append(BedFileData(node_idx=i, embeddings=embeddings, intervals=intervals))
        return data

    @pytest.fixture
    def mock_similarity_matrix(self, tmp_path):
        """Create a mock similarity matrix."""
        from giggleml.data.similarity_matrix import SimilarityMatrix

        matrix = SimilarityMatrix(tmp_path / "sim.mat", n=4, mode="w+")
        # Set some similarities
        for i in range(4):
            for j in range(4):
                if i != j:
                    # Closer indices = higher similarity
                    matrix[i, j] = 50.0 - abs(i - j) * 10.0
        matrix.flush()
        return matrix

    def test_callback_returns_float(self, mock_bed_data, mock_similarity_matrix):
        """Test that the callback returns a float, not a dict."""
        from giggleml.models.cmodel import create_cmodel

        callback = create_ndcg_callback(
            bed_data=mock_bed_data,
            similarity_matrix=mock_similarity_matrix,
            bin_thresholds=(10, 20, 30, 40),
            bin_weights=(0.25, 0.5, 0.75, 1.0),
            anchor_indices=None,
            k=2,
        )

        # Create a minimal CModel using the helper
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            output_dim=16,
        )

        result = callback(model)

        # Result should be a float, not a dict
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_callback_with_anchor_subset(self, mock_bed_data, mock_similarity_matrix):
        """Test callback with a subset of anchors."""
        from giggleml.models.cmodel import create_cmodel

        callback = create_ndcg_callback(
            bed_data=mock_bed_data,
            similarity_matrix=mock_similarity_matrix,
            bin_thresholds=(10, 20, 30, 40),
            bin_weights=(0.25, 0.5, 0.75, 1.0),
            anchor_indices=[0, 2],  # Only use anchors 0 and 2
            k=2,
        )

        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            output_dim=16,
        )

        result = callback(model)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
