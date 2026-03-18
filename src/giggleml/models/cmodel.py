"""CModel: A PerceiverIO-based architecture using Equinox and einx.

Based on PerceiverIO with the following modifications:
- Added input FFN for feature extraction before cross-attention
- Simplified to encoder-only (no decoder cross-attention)
- Configurable weight sharing in the latent transformer stack

Core PerceiverIO ideas retained (~90%):
- Single cross-attention to project variable-length inputs to fixed-size latents
- Efficient self-attention processing entirely in the compact latent space
- O(n·m + m²) complexity where n=input length, m=num_latents

Reference: Jaegle et al. "Perceiver IO: A General Architecture for Structured Inputs & Outputs"
"""

import einx
import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray


class CrossAttention(eqx.Module):
    """Cross-attention: latents attend to inputs with pre-norm and residual."""

    attn: eqx.nn.MultiheadAttention
    ln: eqx.nn.LayerNorm

    def __init__(
        self,
        latent_dim: int,
        input_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()
        self.attn = eqx.nn.MultiheadAttention(
            num_heads, latent_dim, key_size=input_dim, value_size=input_dim, key=key
        )
        self.ln = eqx.nn.LayerNorm(latent_dim)

    def __call__(
        self,
        latents: Float[Array, "num_latents latent_dim"],
        inputs: Float[Array, "input_len input_dim"],
    ) -> Float[Array, "num_latents latent_dim"]:
        x = jax.vmap(self.ln)(latents)
        return latents + self.attn(x, inputs, inputs)


class EncoderBlock(eqx.Module):
    """Standard transformer encoder block: self-attention + feedforward with pre-norm."""

    attn: eqx.nn.MultiheadAttention
    ff: eqx.nn.MLP
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()
        keys = jax.random.split(key, 2)

        self.attn = eqx.nn.MultiheadAttention(num_heads, dim, key=keys[0])
        self.ff = eqx.nn.MLP(
            dim, dim, ff_hidden_dim, depth=1, activation=jax.nn.gelu, key=keys[1]
        )
        self.ln1 = eqx.nn.LayerNorm(dim)
        self.ln2 = eqx.nn.LayerNorm(dim)

    def __call__(
        self,
        x: Float[Array, "seq_len dim"],
    ) -> Float[Array, "seq_len dim"]:
        normed = jax.vmap(self.ln1)(x)
        x = x + self.attn(normed, normed, normed)
        x = x + jax.vmap(self.ff)(jax.vmap(self.ln2)(x))
        return x


class CModel(eqx.Module):
    """PerceiverIO-based encoder with input FFN and configurable weight sharing.

    Architecture:
        1. Input FFN: feature extraction (input_dim -> input_dim)
        2. Cross-attention: project inputs to latent space
        3. Encoder stack: self-attention blocks with weight sharing
        4. Pooling: decode (default), mean, first, or none

    Pooling modes:
        - decode: apply output FFN to first latent (default)
        - mean: mean pool all latents
        - first: return first latent directly
        - none: return all latents

    Weight sharing is controlled by (shared_per_stack, num_stacks):
        - shared_per_stack: number of unique encoder blocks per stack
        - num_stacks: how many times to repeat the shared blocks
        - Total depth = shared_per_stack * num_stacks
        - Example: (2, 3) = 2 unique blocks repeated 3 times = 6 effective layers
    """

    latents: Float[Array, "num_latents latent_dim"]
    input_ffn: eqx.nn.MLP
    cross_attn: CrossAttention
    encoder_blocks: list[EncoderBlock]
    output_norm: eqx.nn.LayerNorm
    output_ffn: eqx.nn.MLP | None
    shared_per_stack: int = eqx.field(static=True)
    num_stacks: int = eqx.field(static=True)
    pooling: str = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_latents: int,
        shared_per_stack: int,
        num_stacks: int,
        num_heads: int,
        input_ff_hidden_dim: int,
        encoder_ff_hidden_dim: int,
        *,
        output_dim: int | None = None,
        output_ff_hidden_dim: int | None = None,
        pooling: str = "decode",
        key: PRNGKeyArray,
    ):
        super().__init__()
        if pooling not in ("decode", "mean", "first", "none"):
            raise ValueError(
                f"pooling must be 'decode', 'mean', 'first', or 'none', got {pooling}"
            )

        self.pooling = pooling
        self.shared_per_stack = shared_per_stack
        self.num_stacks = num_stacks

        keys = jax.random.split(key, shared_per_stack + 4)

        self.latents = jax.random.normal(keys[0], (num_latents, latent_dim)) * 0.02

        self.input_ffn = eqx.nn.MLP(
            input_dim,
            input_dim,
            input_ff_hidden_dim,
            depth=1,
            activation=jax.nn.gelu,
            key=keys[1],
        )

        self.cross_attn = CrossAttention(latent_dim, input_dim, num_heads, key=keys[2])

        self.encoder_blocks = [
            EncoderBlock(latent_dim, num_heads, encoder_ff_hidden_dim, key=k)
            for k in keys[3 : 3 + shared_per_stack]
        ]
        self.output_norm = eqx.nn.LayerNorm(latent_dim)

        if pooling == "decode":
            if output_dim is None:
                output_dim = latent_dim
            if output_ff_hidden_dim is None:
                output_ff_hidden_dim = latent_dim * 4
            self.output_ffn = eqx.nn.MLP(
                latent_dim,
                output_dim,
                output_ff_hidden_dim,
                depth=1,
                activation=jax.nn.gelu,
                key=keys[3 + shared_per_stack],
            )
        else:
            self.output_ffn = None

    def __call__(
        self,
        inputs: Float[Array, "input_len input_dim"],
    ) -> Float[Array, "..."]:
        inputs = jax.vmap(self.input_ffn)(inputs)

        # Cross-attention: project inputs to latent space
        latents = self.cross_attn(self.latents, inputs)

        # Encoder stack with weight sharing
        for _ in range(self.num_stacks):
            for block in self.encoder_blocks:
                latents = block(latents)

        latents = jax.vmap(self.output_norm)(latents)

        if self.pooling == "decode":
            assert self.output_ffn is not None
            return self.output_ffn(latents[0])
        elif self.pooling == "mean":
            return einx.mean("[n] d", latents)
        elif self.pooling == "first":
            return latents[0]
        return latents


def create_cmodel(
    input_dim: int = 256,
    latent_dim: int = 512,
    num_latents: int = 64,
    shared_per_stack: int = 2,
    num_stacks: int = 3,
    num_heads: int = 8,
    input_ff_mult: int = 4,
    encoder_ff_mult: int = 4,
    output_dim: int | None = None,
    output_ff_mult: int = 4,
    pooling: str = "decode",
    key: PRNGKeyArray | None = None,
) -> CModel:
    """Factory function to create a CModel with common defaults.

    Args:
        input_dim: Dimension of input features.
        latent_dim: Dimension of latent space.
        num_latents: Number of latent vectors.
        shared_per_stack: Number of unique encoder blocks (weight sharing unit).
        num_stacks: Number of times to repeat the shared blocks.
        num_heads: Number of attention heads.
        input_ff_mult: Input FFN hidden dimension multiplier (relative to input_dim).
        encoder_ff_mult: Encoder FFN hidden dimension multiplier (relative to latent_dim).
        output_dim: Output dimension for decode mode (defaults to latent_dim).
        output_ff_mult: Output FFN hidden dimension multiplier (relative to latent_dim).
        pooling: Output pooling strategy ('decode', 'mean', 'first', or 'none').
        key: PRNG key for initialization.

    Returns:
        Configured CModel.
    """
    if key is None:
        key = jax.random.key(0)

    return CModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_latents=num_latents,
        shared_per_stack=shared_per_stack,
        num_stacks=num_stacks,
        num_heads=num_heads,
        input_ff_hidden_dim=input_dim * input_ff_mult,
        encoder_ff_hidden_dim=latent_dim * encoder_ff_mult,
        output_dim=output_dim,
        output_ff_hidden_dim=latent_dim * output_ff_mult,
        pooling=pooling,
        key=key,
    )
