"""Tests for giggleml.models.cmodel module."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from giggleml.models.cmodel import (
    CModel,
    CrossAttention,
    EncoderBlock,
    create_cmodel,
)
from giggleml.models.genomic_interval import GenomicIntervalEncoder


class TestCrossAttention:
    """Tests for CrossAttention module."""

    def test_output_shape(self):
        key = jax.random.key(0)
        cross_attn = CrossAttention(latent_dim=64, input_dim=32, num_heads=4, key=key)

        latents = jnp.ones((8, 64))  # 8 latents, 64 dims
        inputs = jnp.ones((100, 32))  # 100 inputs, 32 dims

        output = cross_attn(latents, inputs)
        assert output.shape == (8, 64)

    def test_residual_connection(self):
        key = jax.random.key(0)
        cross_attn = CrossAttention(latent_dim=64, input_dim=32, num_heads=4, key=key)

        latents = jnp.ones((8, 64))
        inputs = jnp.zeros((100, 32))  # zeros to minimize attention contribution

        output = cross_attn(latents, inputs)
        # Output should be close to latents due to residual (not exact due to layer norm)
        assert output.shape == latents.shape

    def test_mask_excludes_positions(self):
        key = jax.random.key(0)
        cross_attn = CrossAttention(latent_dim=64, input_dim=32, num_heads=4, key=key)

        latents = jnp.ones((8, 64))
        inputs = jnp.ones((10, 32))

        # Mask all positions - should still work (degenerate case)
        mask = jnp.ones(10, dtype=bool)
        output = cross_attn(latents, inputs, mask=mask)
        assert output.shape == (8, 64)

        # Mask some positions
        mask = jnp.array([True, False, True, False, True, False, True, False, True, False])
        output = cross_attn(latents, inputs, mask=mask)
        assert output.shape == (8, 64)


class TestEncoderBlock:
    """Tests for EncoderBlock module."""

    def test_output_shape(self):
        key = jax.random.key(0)
        block = EncoderBlock(dim=64, num_heads=4, ff_hidden_dim=256, key=key)

        x = jnp.ones((16, 64))
        output = block(x)
        assert output.shape == (16, 64)

    def test_residual_preserves_scale(self):
        key = jax.random.key(0)
        block = EncoderBlock(dim=64, num_heads=4, ff_hidden_dim=256, key=key)

        x = jnp.ones((16, 64))
        output = block(x)
        # Due to residual connections, output magnitude should be similar to input
        assert jnp.abs(output.mean()) < 10  # reasonable bound


class TestCModel:
    """Tests for CModel module."""

    @pytest.fixture
    def sample_intervals(self):
        """Sample intervals for testing."""
        return jnp.array([
            [0, 1000 + i * 1000, 2000 + i * 1000] for i in range(10)
        ])

    def test_factory_creates_valid_model(self):
        model = create_cmodel(seq_dim=32, latent_dim=64, num_latents=8)
        assert isinstance(model, CModel)
        assert model.latents.shape == (8, 64)
        assert model.interval_encoder is not None

    def test_default_seq_dim(self):
        model = create_cmodel(latent_dim=64, num_latents=8)
        assert model.seq_dim == 128  # default

    def test_forward_pass_mean_pooling(self, sample_intervals):
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            pooling="mean",
        )
        seq_emb = jnp.ones((10, 32))
        output = model(seq_emb, sample_intervals)
        assert output.shape == (64,)

    def test_forward_pass_first_pooling(self, sample_intervals):
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            pooling="first",
        )
        seq_emb = jnp.ones((10, 32))
        output = model(seq_emb, sample_intervals)
        assert output.shape == (64,)

    def test_forward_pass_no_pooling(self, sample_intervals):
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            pooling="none",
        )
        seq_emb = jnp.ones((10, 32))
        output = model(seq_emb, sample_intervals)
        assert output.shape == (8, 64)

    def test_variable_input_length(self):
        model = create_cmodel(seq_dim=32, latent_dim=64, num_latents=8)

        # Different input lengths should work
        intervals1 = jnp.array([[0, i * 1000, (i + 1) * 1000] for i in range(50)])
        intervals2 = jnp.array([[0, i * 1000, (i + 1) * 1000] for i in range(200)])

        out1 = model(jnp.ones((50, 32)), intervals1)
        out2 = model(jnp.ones((200, 32)), intervals2)

        assert out1.shape == out2.shape == (64,)

    def test_weight_sharing_config(self):
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            shared_per_stack=2,
            num_stacks=3,
        )
        # Should have 2 unique encoder blocks
        assert len(model.encoder_blocks) == 2
        assert model.shared_per_stack == 2
        assert model.num_stacks == 3

    def test_effective_depth(self):
        # 2 blocks x 3 stacks = 6 effective layers
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            shared_per_stack=2,
            num_stacks=3,
        )
        # Verify by checking unique blocks vs total applications
        assert len(model.encoder_blocks) * model.num_stacks == 6

    def test_input_ffn_dimensions(self):
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            input_ff_mult=2,
            interval_chrm_dim=8,
            interval_size_dim=8,
            interval_center_dim=16,
        )
        # input_dim = seq_dim + interval_dim = 32 + 32 = 64
        assert model.input_ffn.in_size == 64
        assert model.input_ffn.out_size == 64
        assert model.input_ffn.width_size == 128  # 64 * 2

    def test_encoder_ffn_dimensions(self):
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            encoder_ff_mult=4,
        )
        # Encoder FFN hidden should be latent_dim * mult
        assert model.encoder_blocks[0].ff.width_size == 256  # 64 * 4

    def test_decoupled_ff_dims(self):
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            input_ff_mult=2,
            encoder_ff_mult=4,
            interval_chrm_dim=8,
            interval_size_dim=8,
            interval_center_dim=16,
        )
        # input_dim = 32 + 32 = 64
        assert model.input_ffn.width_size == 128  # 64 * 2
        assert model.encoder_blocks[0].ff.width_size == 256  # 64 * 4

    def test_invalid_pooling_raises(self):
        interval_encoder = GenomicIntervalEncoder(
            chrm_dim=8, size_dim=8, center_dim=16, key=jax.random.key(0)
        )
        with pytest.raises(ValueError, match="pooling must be"):
            CModel(
                seq_dim=32,
                interval_encoder=interval_encoder,
                latent_dim=64,
                num_latents=8,
                shared_per_stack=2,
                num_stacks=3,
                num_heads=4,
                input_ff_mult=2,
                encoder_ff_hidden_dim=256,
                pooling="invalid",
                key=jax.random.key(0),
            )

    def test_gradient_flow(self, sample_intervals):
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            shared_per_stack=1,
            num_stacks=2,
        )
        seq_emb = jnp.ones((10, 32))

        def loss_fn(m: CModel, s: jax.Array, iv: jax.Array) -> jax.Array:
            return jnp.mean(m(s, iv) ** 2)

        grads = eqx.filter_grad(loss_fn)(model, seq_emb, sample_intervals)

        # Check gradients exist for key parameters
        assert grads.latents is not None
        assert not jnp.allclose(grads.latents, 0)

    def test_jit_compilation(self, sample_intervals):
        model = create_cmodel(seq_dim=32, latent_dim=64, num_latents=8)
        seq_emb = jnp.ones((10, 32))

        # Should compile without errors
        jitted = eqx.filter_jit(model)
        output = jitted(seq_emb, sample_intervals)
        assert output.shape == (64,)

    def test_deterministic_with_same_key(self):
        model1 = create_cmodel(seq_dim=32, latent_dim=64, key=jax.random.key(42))
        model2 = create_cmodel(seq_dim=32, latent_dim=64, key=jax.random.key(42))

        assert jnp.allclose(model1.latents, model2.latents)

    def test_different_with_different_key(self):
        model1 = create_cmodel(seq_dim=32, latent_dim=64, key=jax.random.key(0))
        model2 = create_cmodel(seq_dim=32, latent_dim=64, key=jax.random.key(1))

        assert not jnp.allclose(model1.latents, model2.latents)

    def test_interval_encoder_created(self):
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            interval_chrm_dim=8,
            interval_size_dim=8,
            interval_center_dim=16,
        )
        assert model.interval_encoder is not None
        assert model.interval_encoder.dim == 32  # 8 + 8 + 16

    def test_input_dim_is_seq_dim_plus_interval_dim(self):
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            interval_chrm_dim=8,
            interval_size_dim=8,
            interval_center_dim=16,
        )
        # input_dim should be seq_dim + interval_dim = 32 + 32 = 64
        assert model.input_ffn.in_size == 64

    def test_forward_features_bypasses_interval_encoding(self):
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
        )
        # forward_features should work with raw features of the right size
        # input_dim = seq_dim (32) + interval_dim (128 default: 8+8+112) = 160
        raw_features = jnp.ones((10, 160))
        output = model.forward_features(raw_features)
        assert output.shape == (64,)

    def test_gradient_flow_with_intervals(self, sample_intervals):
        model = create_cmodel(seq_dim=32, latent_dim=64, num_latents=8)
        seq_emb = jnp.ones((10, 32))

        def loss_fn(m: CModel, s: jax.Array, iv: jax.Array) -> jax.Array:
            return jnp.mean(m(s, iv) ** 2)

        grads = eqx.filter_grad(loss_fn)(model, seq_emb, sample_intervals)

        # Check gradients exist for key parameters including interval encoder
        assert grads.latents is not None
        assert not jnp.allclose(grads.latents, 0)
        assert grads.interval_encoder is not None
        assert grads.interval_encoder.chrm_embedding.weight is not None


class TestCModelMasking:
    """Tests for CModel masking functionality."""

    @pytest.fixture
    def model(self):
        return create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
            interval_chrm_dim=8,
            interval_size_dim=8,
            interval_center_dim=16,
        )

    @pytest.fixture
    def sample_data(self):
        seq_emb = jnp.ones((10, 32))
        intervals = jnp.array([
            [0, 1000 + i * 1000, 2000 + i * 1000] for i in range(10)
        ])
        return seq_emb, intervals

    def test_no_mask_works(self, model, sample_data):
        seq_emb, intervals = sample_data
        output = model(seq_emb, intervals, mask=None)
        assert output.shape == (64,)

    def test_mask_shape_validation(self, model, sample_data):
        seq_emb, intervals = sample_data
        # Mask should be (input_len, 2)
        mask = jnp.zeros((10, 2), dtype=bool)
        output = model(seq_emb, intervals, mask=mask)
        assert output.shape == (64,)

    def test_seq_only_mask_replaces_with_embedding(self, model, sample_data):
        seq_emb, intervals = sample_data
        # Mask only seq at position 0
        mask = jnp.zeros((10, 2), dtype=bool)
        mask = mask.at[0, 0].set(True)  # seq masked, interval not

        output_masked = model(seq_emb, intervals, mask=mask)
        output_unmasked = model(seq_emb, intervals, mask=None)

        # Outputs should be different due to masking
        assert not jnp.allclose(output_masked, output_unmasked)

    def test_interval_only_mask_replaces_with_embedding(self, model, sample_data):
        seq_emb, intervals = sample_data
        # Mask only interval at position 0
        mask = jnp.zeros((10, 2), dtype=bool)
        mask = mask.at[0, 1].set(True)  # interval masked, seq not

        output_masked = model(seq_emb, intervals, mask=mask)
        output_unmasked = model(seq_emb, intervals, mask=None)

        # Outputs should be different due to masking
        assert not jnp.allclose(output_masked, output_unmasked)

    def test_both_masked_excludes_from_attention(self, model, sample_data):
        seq_emb, intervals = sample_data
        # Mask both seq and interval at position 0
        mask = jnp.zeros((10, 2), dtype=bool)
        mask = mask.at[0, 0].set(True)
        mask = mask.at[0, 1].set(True)

        output_masked = model(seq_emb, intervals, mask=mask)
        output_unmasked = model(seq_emb, intervals, mask=None)

        # Outputs should be different due to position being excluded
        assert not jnp.allclose(output_masked, output_unmasked)

    def test_all_positions_masked_both(self, model, sample_data):
        seq_emb, intervals = sample_data
        # Mask all positions completely (both seq and interval)
        mask = jnp.ones((10, 2), dtype=bool)

        # Should still work (degenerate case - all positions excluded)
        output = model(seq_emb, intervals, mask=mask)
        assert output.shape == (64,)

    def test_mask_embeddings_are_learned(self, model):
        # Check that mask embeddings exist and have correct shapes
        assert model.seq_mask_emb is not None
        assert model.interval_mask_emb is not None
        # Embeddings are eqx.nn.Embedding with num_embeddings=1
        assert model.seq_mask_emb.weight.shape == (1, 32)  # seq_dim
        assert model.interval_mask_emb.weight.shape == (1, 32)  # interval_dim (8+8+16)

    def test_mask_gradient_flow(self, model, sample_data):
        seq_emb, intervals = sample_data
        mask = jnp.zeros((10, 2), dtype=bool)
        mask = mask.at[0, 0].set(True)  # partial mask

        def loss_fn(m: CModel, s: jax.Array, iv: jax.Array, mk: jax.Array) -> jax.Array:
            return jnp.mean(m(s, iv, mk) ** 2)

        grads = eqx.filter_grad(loss_fn)(model, seq_emb, intervals, mask)

        # Gradients should flow to mask embeddings
        assert grads.seq_mask_emb is not None

    def test_mask_with_jit(self, model, sample_data):
        seq_emb, intervals = sample_data
        mask = jnp.zeros((10, 2), dtype=bool)
        mask = mask.at[0, 0].set(True)
        mask = mask.at[1, 1].set(True)
        mask = mask.at[2, 0].set(True)
        mask = mask.at[2, 1].set(True)

        jitted = eqx.filter_jit(model)
        output = jitted(seq_emb, intervals, mask)
        assert output.shape == (64,)

    def test_mixed_masking_patterns(self, model, sample_data):
        seq_emb, intervals = sample_data
        # Create a mixed mask pattern
        mask = jnp.zeros((10, 2), dtype=bool)
        mask = mask.at[0, 0].set(True)  # seq only
        mask = mask.at[1, 1].set(True)  # interval only
        mask = mask.at[2, 0].set(True)  # both
        mask = mask.at[2, 1].set(True)
        # positions 3-9 unmasked

        output = model(seq_emb, intervals, mask=mask)
        assert output.shape == (64,)

    def test_forward_features_with_mask(self, model):
        # forward_features accepts a 1D mask for attention exclusion
        input_dim = 32 + 32  # seq_dim + interval_dim
        raw_features = jnp.ones((10, input_dim))
        mask = jnp.array([True, False, True, False, True, False, True, False, True, False])

        output = model.forward_features(raw_features, mask=mask)
        assert output.shape == (64,)
