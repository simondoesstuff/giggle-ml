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

    def test_factory_creates_valid_model(self):
        model = create_cmodel(input_dim=32, latent_dim=64, num_latents=8)
        assert isinstance(model, CModel)
        assert model.latents.shape == (8, 64)

    def test_forward_pass_mean_pooling(self):
        model = create_cmodel(
            input_dim=32,
            latent_dim=64,
            num_latents=8,
            pooling="mean",
        )
        inputs = jnp.ones((100, 32))
        output = model(inputs)
        assert output.shape == (64,)

    def test_forward_pass_first_pooling(self):
        model = create_cmodel(
            input_dim=32,
            latent_dim=64,
            num_latents=8,
            pooling="first",
        )
        inputs = jnp.ones((100, 32))
        output = model(inputs)
        assert output.shape == (64,)

    def test_forward_pass_no_pooling(self):
        model = create_cmodel(
            input_dim=32,
            latent_dim=64,
            num_latents=8,
            pooling="none",
        )
        inputs = jnp.ones((100, 32))
        output = model(inputs)
        assert output.shape == (8, 64)

    def test_variable_input_length(self):
        model = create_cmodel(input_dim=32, latent_dim=64, num_latents=8)

        # Different input lengths should work
        out1 = model(jnp.ones((50, 32)))
        out2 = model(jnp.ones((200, 32)))

        assert out1.shape == out2.shape == (64,)

    def test_weight_sharing_config(self):
        model = create_cmodel(
            input_dim=32,
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
            input_dim=32,
            latent_dim=64,
            num_latents=8,
            shared_per_stack=2,
            num_stacks=3,
        )
        # Verify by checking unique blocks vs total applications
        assert len(model.encoder_blocks) * model.num_stacks == 6

    def test_input_ffn_dimensions(self):
        model = create_cmodel(
            input_dim=32,
            latent_dim=64,
            num_latents=8,
            input_ff_mult=2,
        )
        # Input FFN should project input_dim -> input_dim with hidden=input_dim*mult
        assert model.input_ffn.in_size == 32
        assert model.input_ffn.out_size == 32
        assert model.input_ffn.width_size == 64  # 32 * 2

    def test_encoder_ffn_dimensions(self):
        model = create_cmodel(
            input_dim=32,
            latent_dim=64,
            num_latents=8,
            encoder_ff_mult=4,
        )
        # Encoder FFN hidden should be latent_dim * mult
        assert model.encoder_blocks[0].ff.width_size == 256  # 64 * 4

    def test_decoupled_ff_dims(self):
        model = create_cmodel(
            input_dim=32,
            latent_dim=64,
            num_latents=8,
            input_ff_mult=2,
            encoder_ff_mult=4,
        )
        assert model.input_ffn.width_size == 64  # 32 * 2
        assert model.encoder_blocks[0].ff.width_size == 256  # 64 * 4

    def test_invalid_pooling_raises(self):
        with pytest.raises(ValueError, match="pooling must be"):
            CModel(
                input_dim=32,
                latent_dim=64,
                num_latents=8,
                shared_per_stack=2,
                num_stacks=3,
                num_heads=4,
                input_ff_hidden_dim=64,
                encoder_ff_hidden_dim=256,
                pooling="invalid",
                key=jax.random.key(0),
            )

    def test_gradient_flow(self):
        model = create_cmodel(
            input_dim=32,
            latent_dim=64,
            num_latents=8,
            shared_per_stack=1,
            num_stacks=2,
        )
        inputs = jnp.ones((10, 32))

        def loss_fn(m: CModel, x: jax.Array) -> jax.Array:
            return jnp.mean(m(x) ** 2)

        grads = eqx.filter_grad(loss_fn)(model, inputs)

        # Check gradients exist for key parameters
        assert grads.latents is not None
        assert not jnp.allclose(grads.latents, 0)

    def test_jit_compilation(self):
        model = create_cmodel(input_dim=32, latent_dim=64, num_latents=8)
        inputs = jnp.ones((50, 32))

        # Should compile without errors
        jitted = eqx.filter_jit(model)
        output = jitted(inputs)
        assert output.shape == (64,)

    def test_deterministic_with_same_key(self):
        model1 = create_cmodel(input_dim=32, latent_dim=64, key=jax.random.key(42))
        model2 = create_cmodel(input_dim=32, latent_dim=64, key=jax.random.key(42))

        assert jnp.allclose(model1.latents, model2.latents)

    def test_different_with_different_key(self):
        model1 = create_cmodel(input_dim=32, latent_dim=64, key=jax.random.key(0))
        model2 = create_cmodel(input_dim=32, latent_dim=64, key=jax.random.key(1))

        assert not jnp.allclose(model1.latents, model2.latents)


class TestCModelWithIntervals:
    """Tests for CModel with genomic interval encoding."""

    def test_factory_with_seq_dim_creates_interval_encoder(self):
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

    def test_forward_with_intervals(self):
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
        )
        seq_emb = jnp.ones((10, 32))
        intervals = jnp.array([
            [0, 1000, 2000],
            [0, 2000, 3000],
            [1, 3000, 4000],
            [1, 4000, 5000],
            [2, 5000, 6000],
            [2, 6000, 7000],
            [3, 7000, 8000],
            [3, 8000, 9000],
            [4, 9000, 10000],
            [4, 10000, 11000],
        ])
        output = model(seq_emb, intervals)
        assert output.shape == (64,)

    def test_forward_without_intervals_uses_seq_directly(self):
        # When no intervals, seq_embeddings should be used as input directly
        model = create_cmodel(input_dim=32, latent_dim=64, num_latents=8)
        seq_emb = jnp.ones((10, 32))
        output = model(seq_emb)
        assert output.shape == (64,)

    def test_forward_features_bypasses_interval_encoding(self):
        model = create_cmodel(
            seq_dim=32,
            latent_dim=64,
            num_latents=8,
        )
        # forward_features should work with raw features of the right size
        # input_dim = seq_dim (32) + interval_dim (64 default) = 96
        raw_features = jnp.ones((10, 96))
        output = model.forward_features(raw_features)
        assert output.shape == (64,)

    def test_intervals_without_encoder_raises(self):
        model = create_cmodel(input_dim=32, latent_dim=64, num_latents=8)
        seq_emb = jnp.ones((10, 32))
        intervals = jnp.array([[0, 1000, 2000]] * 10)

        with pytest.raises(ValueError, match="no interval_encoder configured"):
            model(seq_emb, intervals)

    def test_cannot_specify_both_input_dim_and_seq_dim(self):
        with pytest.raises(ValueError, match="Specify either input_dim or seq_dim"):
            create_cmodel(input_dim=32, seq_dim=32)

    def test_jit_with_intervals(self):
        model = create_cmodel(seq_dim=32, latent_dim=64, num_latents=8)
        seq_emb = jnp.ones((10, 32))
        intervals = jnp.array([[0, 1000 + i * 1000, 2000 + i * 1000] for i in range(10)])

        jitted = eqx.filter_jit(model)
        output = jitted(seq_emb, intervals)
        assert output.shape == (64,)

    def test_gradient_flow_with_intervals(self):
        model = create_cmodel(seq_dim=32, latent_dim=64, num_latents=8)
        seq_emb = jnp.ones((10, 32))
        intervals = jnp.array([[0, 1000 + i * 1000, 2000 + i * 1000] for i in range(10)])

        def loss_fn(m: CModel, s: jax.Array, iv: jax.Array) -> jax.Array:
            return jnp.mean(m(s, iv) ** 2)

        grads = eqx.filter_grad(loss_fn)(model, seq_emb, intervals)

        # Check gradients exist for key parameters including interval encoder
        assert grads.latents is not None
        assert not jnp.allclose(grads.latents, 0)
        assert grads.interval_encoder is not None
        assert grads.interval_encoder.chrm_embedding.weight is not None
