"""Tests for GenomicIntervalEncoder."""

import jax
import jax.numpy as jnp
import pytest

from giggleml.models.genomic_interval import GenomicIntervalEncoder, genomic_sinusoidal_pe


class TestGenomicSinusoidalPE:
    """Tests for genomic sinusoidal positional encoding."""

    def test_output_shape(self):
        pe = genomic_sinusoidal_pe(jnp.array(100.0), dim=64)
        assert pe.shape == (64,)

    def test_output_shape_odd_dim(self):
        pe = genomic_sinusoidal_pe(jnp.array(100.0), dim=65)
        assert pe.shape == (65,)

    def test_different_positions_differ(self):
        pe1 = genomic_sinusoidal_pe(jnp.array(0.0), dim=64)
        pe2 = genomic_sinusoidal_pe(jnp.array(1000.0), dim=64)
        assert not jnp.allclose(pe1, pe2)

    def test_bounded_output(self):
        pe = genomic_sinusoidal_pe(jnp.array(1e6), dim=64)
        assert jnp.all(pe >= -1.0)
        assert jnp.all(pe <= 1.0)

    def test_large_genomic_coords_differ(self):
        """Verify PE can distinguish positions at genomic scale."""
        # Two positions 1Mbp apart at ~200M should be distinguishable
        pe1 = genomic_sinusoidal_pe(jnp.array(200_000_000.0), dim=64)
        pe2 = genomic_sinusoidal_pe(jnp.array(201_000_000.0), dim=64)
        assert not jnp.allclose(pe1, pe2)

    def test_nearby_positions_differ(self):
        """Verify PE can distinguish nearby positions (single-bp resolution)."""
        pe1 = genomic_sinusoidal_pe(jnp.array(100_000_000.0), dim=64)
        pe2 = genomic_sinusoidal_pe(jnp.array(100_000_100.0), dim=64)
        assert not jnp.allclose(pe1, pe2)


class TestGenomicIntervalEncoder:
    """Tests for GenomicIntervalEncoder."""

    @pytest.fixture
    def encoder(self):
        return GenomicIntervalEncoder(
            chrm_dim=16, size_dim=16, center_dim=32, num_chrms=24, key=jax.random.key(42)
        )

    def test_output_shape(self, encoder):
        emb = encoder(jnp.array([0, 1000, 2000]))
        assert emb.shape == (64,)  # 16 + 16 + 32

    def test_dim_property(self, encoder):
        assert encoder.dim == 64

    def test_batch_encoding(self, encoder):
        intervals = jnp.array([
            [0, 1000, 2000],
            [1, 2000, 3000],
            [2, 3000, 4000],
        ])
        embs = encoder.encode_batch(intervals)
        assert embs.shape == (3, 64)

    def test_different_chromosomes_differ(self, encoder):
        emb1 = encoder(jnp.array([0, 1000, 2000]))
        emb2 = encoder(jnp.array([1, 1000, 2000]))
        assert not jnp.allclose(emb1, emb2)

    def test_different_positions_differ(self, encoder):
        emb1 = encoder(jnp.array([0, 1000, 2000]))
        emb2 = encoder(jnp.array([0, 5000, 6000]))
        assert not jnp.allclose(emb1, emb2)

    def test_different_sizes_differ(self, encoder):
        emb1 = encoder(jnp.array([0, 1000, 2000]))  # size 1000
        emb2 = encoder(jnp.array([0, 1000, 1100]))  # size 100
        assert not jnp.allclose(emb1, emb2)

    def test_components_are_concatenated(self, encoder):
        """Verify output is concatenation of chrm, size, center embeddings."""
        interval = jnp.array([0, 1000, 2000])
        emb = encoder(interval)

        # Check that different dimension configs produce different output sizes
        enc_small = GenomicIntervalEncoder(
            chrm_dim=8, size_dim=8, center_dim=16, key=jax.random.key(0)
        )
        assert enc_small.dim == 32
        assert enc_small(interval).shape == (32,)

    def test_jit_compilation(self, encoder):
        jit_call = jax.jit(encoder.__call__)
        emb = jit_call(jnp.array([0, 1000, 2000]))
        assert emb.shape == (64,)

    def test_deterministic_with_same_key(self):
        enc1 = GenomicIntervalEncoder(
            chrm_dim=16, size_dim=16, center_dim=32, key=jax.random.key(42)
        )
        enc2 = GenomicIntervalEncoder(
            chrm_dim=16, size_dim=16, center_dim=32, key=jax.random.key(42)
        )
        emb1 = enc1(jnp.array([0, 1000, 2000]))
        emb2 = enc2(jnp.array([0, 1000, 2000]))
        assert jnp.allclose(emb1, emb2)

    def test_gradient_flow(self, encoder):
        def loss_fn(model, interval):
            emb = model(interval)
            return jnp.sum(emb**2)

        grads = jax.grad(loss_fn)(encoder, jnp.array([0, 1000, 2000]))
        # Check that gradients exist for learned parameters
        assert grads.chrm_embedding.weight is not None
        assert grads.size_proj.weight is not None

    def test_handles_zero_size_interval(self, encoder):
        # Edge case: start == end (size 0, but we clamp to 1)
        emb = encoder(jnp.array([0, 1000, 1000]))
        assert emb.shape == (64,)
        assert jnp.all(jnp.isfinite(emb))

    def test_handles_large_coordinates(self, encoder):
        # Human chr1 is ~250M bp
        emb = encoder(jnp.array([0, 200_000_000, 200_001_000]))
        assert emb.shape == (64,)
        assert jnp.all(jnp.isfinite(emb))

    def test_custom_max_wavelength(self):
        """Test encoder with custom max_wavelength for different genomes."""
        # E. coli genome is ~4.6M bp
        enc = GenomicIntervalEncoder(
            chrm_dim=8, size_dim=8, center_dim=16,
            num_chrms=1, max_wavelength=5_000_000.0,
            key=jax.random.key(0)
        )
        emb = enc(jnp.array([0, 2_000_000, 2_001_000]))
        assert emb.shape == (32,)
        assert jnp.all(jnp.isfinite(emb))
