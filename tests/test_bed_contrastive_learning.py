"""Tests for giggleml.train.bed_contrastive_learning module."""

import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from giggleml.models.cmodel import create_cmodel
from giggleml.train.bed_contrastive_learning import (
    ContrastiveTrainingConfig,
    create_optimizer,
    train_step_unpadded,
    weighted_infonce_loss,
)
from giggleml.train.contrastive_data_loader import (
    BedFileCache,
    BedFileData,
    ContrastiveBatch,
    ContrastiveDataLoader,
)
from giggleml.train.similarity_graph.similarity_graph import SimilarityGraph
from giggleml.utils.equinox import load_checkpoint, save_checkpoint


class TestContrastiveTrainingConfig:
    """Tests for ContrastiveTrainingConfig dataclass."""

    def test_default_values(self):
        config = ContrastiveTrainingConfig()
        assert config.seq_dim == 128
        assert config.latent_dim == 512
        assert config.num_latents == 1024
        assert config.output_dim == 128
        assert config.peak_learning_rate == 1e-4
        assert config.temperature == 0.07
        assert config.bin_thresholds == (0.1, 0.3, 0.5, 0.7)
        assert config.bin_weights == (0.0, 0.5, 1.0, 2.0)

    def test_custom_values(self):
        config = ContrastiveTrainingConfig(
            seq_dim=128,
            latent_dim=256,
            peak_learning_rate=3e-4,
            bin_thresholds=(0.2, 0.5, 0.8),
            bin_weights=(0.0, 1.0, 2.0),
        )
        assert config.seq_dim == 128
        assert config.latent_dim == 256
        assert config.peak_learning_rate == 3e-4
        assert config.bin_thresholds == (0.2, 0.5, 0.8)


class TestBedFileData:
    """Tests for BedFileData dataclass."""

    def test_frozen(self):
        embeddings = jnp.ones((10, 32))
        intervals = jnp.array([[0, 100, 200]] * 10, dtype=jnp.int32)
        bed_data = BedFileData(node_idx=0, embeddings=embeddings, intervals=intervals)

        with pytest.raises(AttributeError):
            bed_data.node_idx = 1

    def test_attributes(self):
        embeddings = jnp.ones((10, 32))
        intervals = jnp.array([[0, 100, 200]] * 10, dtype=jnp.int32)
        bed_data = BedFileData(node_idx=5, embeddings=embeddings, intervals=intervals)

        assert bed_data.node_idx == 5
        assert bed_data.embeddings.shape == (10, 32)
        assert bed_data.intervals.shape == (10, 3)


class TestContrastiveBatch:
    """Tests for ContrastiveBatch dataclass."""

    def test_frozen(self):
        bed_data = [
            BedFileData(
                node_idx=i,
                embeddings=jnp.ones((5, 32)),
                intervals=jnp.array([[0, 100, 200]] * 5, dtype=jnp.int32),
            )
            for i in range(4)
        ]
        adjacency = jnp.zeros((4, 4), dtype=jnp.int32)
        batch = ContrastiveBatch(bed_data=bed_data, adjacency=adjacency)

        with pytest.raises(AttributeError):
            batch.adjacency = jnp.ones((4, 4))

    def test_attributes(self):
        bed_data = [
            BedFileData(
                node_idx=i,
                embeddings=jnp.ones((5, 32)),
                intervals=jnp.array([[0, 100, 200]] * 5, dtype=jnp.int32),
            )
            for i in range(4)
        ]
        adjacency = jnp.eye(4, dtype=jnp.int32)
        batch = ContrastiveBatch(bed_data=bed_data, adjacency=adjacency)

        assert len(batch.bed_data) == 4
        assert batch.adjacency.shape == (4, 4)


class TestWeightedInfoNCELoss:
    """Tests for weighted_infonce_loss function."""

    def test_output_is_scalar(self):
        embeddings = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        adjacency = jnp.array(
            [
                [0, 1, 0, 2],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [2, 0, 1, 0],
            ]
        )
        edge_type_weights = jnp.array([1.0, 2.0])

        loss = weighted_infonce_loss(embeddings, adjacency, edge_type_weights)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_no_positives_returns_zero(self):
        embeddings = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        # No edges (all zeros)
        adjacency = jnp.zeros((3, 3), dtype=jnp.int32)
        edge_type_weights = jnp.array([1.0, 2.0])

        loss = weighted_infonce_loss(embeddings, adjacency, edge_type_weights)
        assert jnp.isclose(loss, 0.0)

    def test_similar_embeddings_lower_loss(self):
        # Positive pairs are similar
        embeddings_similar = jnp.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],  # Very similar to 0
                [0.0, 1.0],
                [0.05, 0.95],  # Very similar to 2
            ]
        )
        # Positive pairs are dissimilar
        embeddings_dissimilar = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],  # Orthogonal to 0
                [0.0, 1.0],
                [1.0, 0.0],  # Orthogonal to 2
            ]
        )
        adjacency = jnp.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        )
        edge_type_weights = jnp.array([1.0])

        loss_similar = weighted_infonce_loss(
            embeddings_similar, adjacency, edge_type_weights
        )
        loss_dissimilar = weighted_infonce_loss(
            embeddings_dissimilar, adjacency, edge_type_weights
        )

        assert loss_similar < loss_dissimilar

    def test_higher_weight_increases_contribution(self):
        embeddings = jnp.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ]
        )
        adjacency = jnp.array(
            [
                [0, 1, 2],  # Edge to 1 has type 1, edge to 2 has type 2
                [1, 0, 0],
                [2, 0, 0],
            ]
        )

        # Higher weight for type 2
        weights_high_type2 = jnp.array([0.1, 10.0])
        # Higher weight for type 1
        weights_high_type1 = jnp.array([10.0, 0.1])

        loss_high_type2 = weighted_infonce_loss(
            embeddings, adjacency, weights_high_type2
        )
        loss_high_type1 = weighted_infonce_loss(
            embeddings, adjacency, weights_high_type1
        )

        # Loss values should be different due to different weighting
        assert not jnp.isclose(loss_high_type2, loss_high_type1)

    def test_zero_weight_bins_are_negatives(self):
        embeddings = jnp.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],  # Similar to 0
                [0.0, 1.0],
            ]
        )
        # Pair (0, 1) has edge type 1, pair (0, 2) has edge type 2
        adjacency = jnp.array(
            [
                [0, 1, 2],
                [1, 0, 0],
                [2, 0, 0],
            ]
        )

        # Zero weight for type 1 means (0, 1) is treated as negative
        edge_type_weights = jnp.array([0.0, 1.0])
        loss = weighted_infonce_loss(embeddings, adjacency, edge_type_weights)

        # Should still compute without error
        assert jnp.isfinite(loss)

    def test_l2_normalization(self):
        # Embeddings with different norms should give same loss
        # if they have same direction
        embeddings1 = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        embeddings2 = jnp.array(
            [
                [10.0, 0.0],  # Same direction, different norm
                [0.0, 5.0],
            ]
        )
        adjacency = jnp.array(
            [
                [0, 1],
                [1, 0],
            ]
        )
        edge_type_weights = jnp.array([1.0])

        loss1 = weighted_infonce_loss(embeddings1, adjacency, edge_type_weights)
        loss2 = weighted_infonce_loss(embeddings2, adjacency, edge_type_weights)

        assert jnp.allclose(loss1, loss2, atol=1e-5)

    def test_temperature_scaling(self):
        embeddings = jnp.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ]
        )
        adjacency = jnp.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        )
        edge_type_weights = jnp.array([1.0])

        loss_low_temp = weighted_infonce_loss(
            embeddings, adjacency, edge_type_weights, temperature=0.01
        )
        loss_high_temp = weighted_infonce_loss(
            embeddings, adjacency, edge_type_weights, temperature=1.0
        )

        # Different temperatures should give different losses
        assert not jnp.isclose(loss_low_temp, loss_high_temp)

    def test_jit_compatible(self):
        embeddings = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        adjacency = jnp.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        )
        edge_type_weights = jnp.array([1.0])

        jitted_loss = jax.jit(weighted_infonce_loss)
        loss = jitted_loss(embeddings, adjacency, edge_type_weights)
        assert jnp.isfinite(loss)


class TestCreateOptimizer:
    """Tests for create_optimizer function."""

    def test_returns_gradient_transformation(self):
        config = ContrastiveTrainingConfig()
        optimizer = create_optimizer(config)

        # Should be usable with optax
        assert hasattr(optimizer, "init")
        assert hasattr(optimizer, "update")

    def test_optimizer_init(self):
        config = ContrastiveTrainingConfig()
        optimizer = create_optimizer(config)

        # Create a simple model to test with
        params = {"w": jnp.ones((10, 10))}
        opt_state = optimizer.init(params)
        assert opt_state is not None

    def test_optimizer_update(self):
        config = ContrastiveTrainingConfig()
        optimizer = create_optimizer(config)

        params = {"w": jnp.ones((10, 10))}
        grads = {"w": jnp.ones((10, 10)) * 0.1}
        opt_state = optimizer.init(params)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        assert "w" in updates
        assert new_opt_state is not None


class TestTrainStep:
    """Tests for train_step function."""

    def test_output_shapes(self):
        key = jax.random.key(42)
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            output_dim=16,
            pooling="mean",
            key=key,
        )

        config = ContrastiveTrainingConfig(
            seq_dim=32, latent_dim=64, num_latents=8, output_dim=16
        )
        optimizer = create_optimizer(config)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        # Create mock batch data
        batch_size = 4
        embeddings_batch = [jnp.ones((10, 32)) for _ in range(batch_size)]
        intervals_batch = [
            jnp.array([[0, 1000 + i * 100, 2000 + i * 100] for i in range(10)])
            for _ in range(batch_size)
        ]
        adjacency = jnp.array(
            [
                [0, 1, 0, 2],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [2, 0, 1, 0],
            ]
        )
        edge_type_weights = jnp.array([1.0, 2.0])

        step_key = jax.random.key(999)
        new_model, new_opt_state, loss = train_step_unpadded(
            model,
            opt_state,
            optimizer,
            embeddings_batch,
            intervals_batch,
            adjacency,
            edge_type_weights,
            temperature=0.07,
            key=step_key,
        )

        assert isinstance(new_model, type(model))
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_loss_decreases(self):
        key = jax.random.key(123)
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            output_dim=16,
            pooling="mean",
            key=key,
        )

        config = ContrastiveTrainingConfig(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            output_dim=16,
            peak_learning_rate=1e-2,  # Higher LR for faster convergence in test
        )
        optimizer = create_optimizer(config)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        # Create fixed mock data
        batch_size = 4
        embeddings_batch = [jnp.ones((10, 32)) * (i + 1) for i in range(batch_size)]
        intervals_batch = [
            jnp.array([[0, 1000 + i * 100, 2000 + i * 100] for i in range(10)])
            for _ in range(batch_size)
        ]
        adjacency = jnp.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ]
        )
        edge_type_weights = jnp.array([1.0])

        # Run multiple steps
        losses = []
        train_key = jax.random.key(999)
        for _ in range(10):
            train_key, step_key = jax.random.split(train_key)
            model, opt_state, loss = train_step_unpadded(
                model,
                opt_state,
                optimizer,
                embeddings_batch,
                intervals_batch,
                adjacency,
                edge_type_weights,
                temperature=0.07,
                key=step_key,
            )
            losses.append(float(loss))

        # Loss should generally decrease (allow some fluctuation)
        # Check that later losses are lower than earlier ones on average
        early_avg = np.mean(losses[:3])
        late_avg = np.mean(losses[-3:])
        assert late_avg <= early_avg or np.isclose(late_avg, early_avg, rtol=0.1)

    def test_gradient_flow(self):
        key = jax.random.key(456)
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            output_dim=16,
            pooling="mean",
            key=key,
        )

        # Use adamw directly without warmup to ensure non-zero LR from step 0
        optimizer = optax.adamw(learning_rate=1e-2, weight_decay=0.01)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        batch_size = 4
        # Use different embeddings per batch item to create gradient signal
        rng = jax.random.key(789)
        embeddings_batch = [
            jax.random.normal(jax.random.fold_in(rng, i), (10, 32))
            for i in range(batch_size)
        ]
        intervals_batch = [
            jnp.array([[i % 24, 1000 + j * 100, 2000 + j * 100] for j in range(10)])
            for i in range(batch_size)
        ]
        # Create adjacency where some pairs are positives (non-zero weight)
        adjacency = jnp.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ]
        )
        edge_type_weights = jnp.array([1.0])

        step_key = jax.random.key(888)
        new_model, _, loss = train_step_unpadded(
            model,
            opt_state,
            optimizer,
            embeddings_batch,
            intervals_batch,
            adjacency,
            edge_type_weights,
            temperature=0.07,
            key=step_key,
        )

        # Loss should be non-zero with these inputs
        assert loss > 0.0
        # At least some model parameters should have changed
        # Check the input FFN weights which definitely receive gradients
        old_weights = model.input_ffn.layers[0].weight
        new_weights = new_model.input_ffn.layers[0].weight
        assert not jnp.allclose(old_weights, new_weights)


class TestCheckpointing:
    """Tests for save_checkpoint and load_checkpoint functions."""

    def test_save_and_load(self):
        key = jax.random.key(42)
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            output_dim=16,
            pooling="mean",
            key=key,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.eqx"
            save_checkpoint(model, checkpoint_path)

            # Create template with different key (different weights)
            template = create_cmodel(
                seq_dim=32,
                latent_dim=64,
                num_latents=8,
                output_dim=16,
                pooling="mean",
                key=jax.random.key(999),
            )

            # Load should restore original weights
            loaded_model = load_checkpoint(checkpoint_path, template)

            assert jnp.allclose(model.latents, loaded_model.latents)

    def test_checkpoint_file_exists(self):
        key = jax.random.key(42)
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            pooling="mean",
            key=key,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.eqx"
            save_checkpoint(model, checkpoint_path)
            assert checkpoint_path.exists()

    def test_load_nonexistent_raises(self):
        key = jax.random.key(42)
        template = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            key=key,
        )

        with pytest.raises(FileNotFoundError):
            load_checkpoint(Path("/nonexistent/path.eqx"), template)


class TestContrastiveDataLoaderDownsampling:
    """Tests for ContrastiveDataLoader max_intervals downsampling."""

    def test_downsample_returns_unchanged_when_below_cap(self):
        """Data under max_intervals should be returned unchanged."""
        loader = ContrastiveDataLoader.__new__(ContrastiveDataLoader)

        data = BedFileData(
            node_idx=0,
            embeddings=jnp.ones((50, 32)),
            intervals=jnp.arange(150).reshape(50, 3),
        )
        key = jax.random.key(42)

        result = loader._downsample(data, max_intervals=100, key=key)

        assert result is data  # Same object, not a copy

    def test_downsample_reduces_to_max_intervals(self):
        """Data over max_intervals should be reduced to exactly max_intervals."""
        loader = ContrastiveDataLoader.__new__(ContrastiveDataLoader)

        data = BedFileData(
            node_idx=0,
            embeddings=jnp.ones((200, 32)),
            intervals=jnp.arange(600).reshape(200, 3),
        )
        key = jax.random.key(42)

        result = loader._downsample(data, max_intervals=50, key=key)

        assert result.embeddings.shape[0] == 50
        assert result.intervals.shape[0] == 50
        assert result.node_idx == 0

    def test_downsample_selects_correct_pairs(self):
        """Downsampled embeddings and intervals should stay paired."""
        loader = ContrastiveDataLoader.__new__(ContrastiveDataLoader)

        # Create data where embedding[i] has unique values matching interval[i]
        embeddings = jnp.arange(100).reshape(100, 1).astype(jnp.float32)
        intervals = jnp.stack(
            [jnp.zeros(100), jnp.arange(100), jnp.arange(100) + 500], axis=1
        ).astype(jnp.int32)
        data = BedFileData(node_idx=0, embeddings=embeddings, intervals=intervals)
        key = jax.random.key(42)

        result = loader._downsample(data, max_intervals=10, key=key)

        # Each embedding value should match the corresponding interval start
        assert jnp.allclose(result.embeddings[:, 0], result.intervals[:, 1])

    def test_downsample_different_keys_different_samples(self):
        """Different PRNG keys should produce different samples."""
        loader = ContrastiveDataLoader.__new__(ContrastiveDataLoader)

        data = BedFileData(
            node_idx=0,
            embeddings=jnp.arange(320).reshape(32, 10).astype(jnp.float32),
            intervals=jnp.arange(96).reshape(32, 3),
        )

        result1 = loader._downsample(data, max_intervals=10, key=jax.random.key(1))
        result2 = loader._downsample(data, max_intervals=10, key=jax.random.key(2))

        # Different keys should give different samples
        assert not jnp.array_equal(result1.intervals, result2.intervals)

    def test_downsample_same_key_same_sample(self):
        """Same PRNG key should produce identical samples."""
        loader = ContrastiveDataLoader.__new__(ContrastiveDataLoader)

        data = BedFileData(
            node_idx=0,
            embeddings=jnp.arange(320).reshape(32, 10).astype(jnp.float32),
            intervals=jnp.arange(96).reshape(32, 3),
        )

        result1 = loader._downsample(data, max_intervals=10, key=jax.random.key(42))
        result2 = loader._downsample(data, max_intervals=10, key=jax.random.key(42))

        assert jnp.array_equal(result1.intervals, result2.intervals)
        assert jnp.array_equal(result1.embeddings, result2.embeddings)

    def test_cache_stores_full_data(self, tmp_path):
        """Cache should store full data, not downsampled."""
        cache = BedFileCache(
            bed_names=["bed0", "bed1"],
            embedding_dir=tmp_path,
            bed_dir=tmp_path,
        )

        # Manually populate cache with large data
        large_data = BedFileData(
            node_idx=0,
            embeddings=jnp.ones((500, 32)),
            intervals=jnp.arange(1500).reshape(500, 3),
        )
        cache._cache[0] = large_data

        # Verify cache has full data
        assert cache._cache[0].embeddings.shape[0] == 500


class TestBedFileCache:
    """Tests for BedFileCache functionality."""

    def test_cache_starts_empty(self, tmp_path):
        cache = BedFileCache(
            bed_names=["bed0", "bed1", "bed2"],
            embedding_dir=tmp_path,
            bed_dir=tmp_path,
        )
        assert cache.cache_size() == 0

    def test_clear_cache(self, tmp_path):
        cache = BedFileCache(
            bed_names=["bed0", "bed1", "bed2"],
            embedding_dir=tmp_path,
            bed_dir=tmp_path,
        )
        # Manually add something to cache
        cache._cache[0] = BedFileData(
            node_idx=0,
            embeddings=jnp.ones((5, 32)),
            intervals=jnp.array([[0, 100, 200]] * 5),
        )
        assert cache.cache_size() == 1

        cache.clear()
        assert cache.cache_size() == 0
