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
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from giggleml.models.genomic_interval import GenomicIntervalEncoder


class CrossAttention(eqx.Module):
    """Cross-attention: latents attend to inputs with pre-norm and residual."""

    attn: eqx.nn.MultiheadAttention
    ln: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        latent_dim: int,
        input_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()
        self.attn = eqx.nn.MultiheadAttention(
            num_heads, latent_dim, key_size=input_dim, value_size=input_dim, key=key
        )
        self.ln = eqx.nn.LayerNorm(latent_dim)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        latents: Float[Array, "num_latents latent_dim"],
        inputs: Float[Array, "input_len input_dim"],
        mask: Bool[Array, "input_len"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "num_latents latent_dim"]:
        x = jax.vmap(self.ln)(latents)
        # mask shape for MHA: (num_latents, input_len), True = attend
        # Our input mask: True = masked/removed, so we invert
        attn_mask: Bool[Array, "num_latents input_len"] | None = None
        if mask is not None:
            # Broadcast inverted mask to (num_latents, input_len)
            attn_mask = jnp.broadcast_to(~mask, (latents.shape[0], mask.shape[0]))
        attn_out = self.attn(x, inputs, inputs, mask=attn_mask)
        return latents + self.dropout(attn_out, key=key)


class EncoderBlock(eqx.Module):
    """Standard transformer encoder block: self-attention + feedforward with pre-norm."""

    attn: eqx.nn.MultiheadAttention
    ff: eqx.nn.MLP
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout_rate: float = 0.0,
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
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        x: Float[Array, "seq_len dim"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len dim"]:
        key1, key2 = (None, None) if key is None else jax.random.split(key)
        normed = jax.vmap(self.ln1)(x)
        x = x + self.dropout1(self.attn(normed, normed, normed), key=key1)
        x = x + self.dropout2(jax.vmap(self.ff)(jax.vmap(self.ln2)(x)), key=key2)
        return x


class CModel(eqx.Module):
    """PerceiverIO-based encoder with input FFN and configurable weight sharing.

    Architecture:
        1. Genomic interval encoding: encode intervals and concat with seq embeddings
        2. Input FFN: feature extraction (input_dim -> input_dim)
        3. Cross-attention: project inputs to latent space
        4. Encoder stack: self-attention blocks with weight sharing
        5. Pooling: decode (default), mean, first, or none

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
    input_dropout: eqx.nn.Dropout
    cross_attn: CrossAttention
    encoder_blocks: list[EncoderBlock]
    output_norm: eqx.nn.LayerNorm
    output_ffn: eqx.nn.MLP | None
    interval_encoder: GenomicIntervalEncoder
    seq_mask_emb: eqx.nn.Embedding
    interval_mask_emb: eqx.nn.Embedding
    shared_per_stack: int = eqx.field(static=True)
    num_stacks: int = eqx.field(static=True)
    pooling: str = eqx.field(static=True)
    seq_dim: int = eqx.field(static=True)

    def __init__(
        self,
        seq_dim: int,
        interval_encoder: GenomicIntervalEncoder,
        latent_dim: int,
        num_latents: int,
        shared_per_stack: int,
        num_stacks: int,
        num_heads: int,
        input_ff_mult: int,
        encoder_ff_hidden_dim: int,
        *,
        output_dim: int | None = None,
        output_ff_hidden_dim: int | None = None,
        dropout_rate: float = 0.0,
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
        self.seq_dim = seq_dim
        self.interval_encoder = interval_encoder

        input_dim = seq_dim + interval_encoder.dim
        input_ff_hidden_dim = input_dim * input_ff_mult

        keys = jax.random.split(key, shared_per_stack + 5)

        self.latents = jax.random.normal(keys[0], (num_latents, latent_dim)) * 0.02

        self.input_ffn = eqx.nn.MLP(
            input_dim,
            input_dim,
            input_ff_hidden_dim,
            depth=1,
            activation=jax.nn.gelu,
            key=keys[1],
        )
        self.input_dropout = eqx.nn.Dropout(dropout_rate)

        self.cross_attn = CrossAttention(
            latent_dim, input_dim, num_heads, dropout_rate, key=keys[2]
        )

        self.encoder_blocks = [
            EncoderBlock(
                latent_dim, num_heads, encoder_ff_hidden_dim, 0.3 * dropout_rate, key=k
            )
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

        # Mask embeddings for partial masking (single embedding each)
        mask_key = jax.random.split(keys[4 + shared_per_stack], 2)
        self.seq_mask_emb = eqx.nn.Embedding(1, seq_dim, key=mask_key[0])
        self.interval_mask_emb = eqx.nn.Embedding(
            1, interval_encoder.dim, key=mask_key[1]
        )

    def forward_features(
        self,
        inputs: Float[Array, "input_len input_dim"],
        mask: Bool[Array, "input_len"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "..."]:
        """Process pre-computed feature vectors through the model.

        Use this method directly when you have raw feature vectors that don't
        need interval encoding. This is the original forward pass logic.

        Args:
            inputs: Input features of shape (input_len, input_dim).
            mask: Optional mask of shape (input_len,). True = position is removed
                (both seq and interval masked). These positions are excluded from
                cross-attention.
            key: Optional PRNG key for dropout. If None, dropout is disabled
                (inference mode). Use eqx.nn.inference_mode() for cleaner inference.

        Returns:
            Output based on pooling mode.
        """
        # Split keys for each dropout location
        num_blocks = self.shared_per_stack * self.num_stacks
        if key is not None:
            keys = jax.random.split(key, 2 + num_blocks)
            input_key, cross_key = keys[0], keys[1]
            block_keys = keys[2:]
        else:
            input_key = cross_key = None
            block_keys = [None] * num_blocks

        inputs = jax.vmap(self.input_ffn)(inputs)
        inputs = self.input_dropout(inputs, key=input_key)

        # Cross-attention: project inputs to latent space
        latents = self.cross_attn(self.latents, inputs, mask=mask, key=cross_key)

        # Encoder stack with weight sharing
        block_idx = 0
        for _ in range(self.num_stacks):
            for block in self.encoder_blocks:
                latents = block(latents, key=block_keys[block_idx])
                block_idx += 1

        latents = jax.vmap(self.output_norm)(latents)

        if self.pooling == "decode":
            assert self.output_ffn is not None
            return self.output_ffn(latents[0])
        elif self.pooling == "mean":
            return einx.mean("[n] d", latents)
        elif self.pooling == "first":
            return latents[0]
        return latents

    def __call__(
        self,
        seq_embeddings: Float[Array, "input_len seq_dim"],
        intervals: Int[Array, "input_len 3"],
        mask: Bool[Array, "input_len 2"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "..."]:
        """Process sequence embeddings with genomic interval encoding.

        Args:
            seq_embeddings: Sequence embeddings of shape (input_len, seq_dim).
            intervals: Genomic intervals of shape (input_len, 3) where
                each row is [chrm, start, end]. Intervals are encoded
                and concatenated with seq_embeddings.
            mask: Optional mask of shape (input_len, 2). Column 0 is seq mask,
                column 1 is interval mask. True = masked.
                - If only seq is masked: replace seq with learned seq_mask_emb
                - If only interval is masked: replace interval with learned interval_mask_emb
                - If both are masked: position is removed (excluded from cross-attention)
            key: Optional PRNG key for dropout. If None, dropout is disabled
                (inference mode). Use eqx.nn.inference_mode() for cleaner inference.

        Returns:
            Output based on pooling mode.
        """
        interval_embs = self.interval_encoder.encode_batch(intervals)

        # Handle masking
        attn_mask = None
        if mask is not None:
            seq_mask = mask[:, 0]  # (input_len,)
            interval_mask = mask[:, 1]  # (input_len,)

            # Partial masking: replace with learned embeddings
            # Only seq masked (not interval): replace seq with embedding
            seq_only_masked = seq_mask & ~interval_mask
            seq_embeddings = jnp.where(
                seq_only_masked[:, None], self.seq_mask_emb(0), seq_embeddings
            )

            # Only interval masked (not seq): replace interval with embedding
            interval_only_masked = interval_mask & ~seq_mask
            interval_embs = jnp.where(
                interval_only_masked[:, None], self.interval_mask_emb(0), interval_embs
            )

            # Full masking: both seq and interval masked = removed position
            attn_mask = seq_mask & interval_mask

        inputs = jnp.concatenate([seq_embeddings, interval_embs], axis=-1)
        return self.forward_features(inputs, mask=attn_mask, key=key)


def create_cmodel(
    seq_dim: int = 128,
    latent_dim: int = 512,
    num_latents: int = 1024,
    shared_per_stack: int = 2,
    num_stacks: int = 3,
    num_heads: int = 8,
    input_ff_mult: int = 2,
    encoder_ff_mult: int = 4,
    output_dim: int | None = None,
    output_ff_mult: int = 4,
    dropout_rate: float = 0.0,
    pooling: str = "decode",
    interval_chrm_dim: int = 8,
    interval_size_dim: int = 8,
    interval_center_dim: int = 112,
    num_chrms: int = 24,
    max_wavelength: float = 250_000_000.0,
    key: PRNGKeyArray | None = None,
) -> CModel:
    """Factory function to create a CModel with common defaults.

    Args:
        seq_dim: Dimension of sequence embeddings.
        latent_dim: Dimension of latent space.
        num_latents: Number of latent vectors.
        shared_per_stack: Number of unique encoder blocks (weight sharing unit).
        num_stacks: Number of times to repeat the shared blocks.
        num_heads: Number of attention heads.
        input_ff_mult: Input FFN hidden dimension multiplier (relative to input_dim).
        encoder_ff_mult: Encoder FFN hidden dimension multiplier (relative to latent_dim).
        output_dim: Output dimension for decode mode (defaults to latent_dim).
        output_ff_mult: Output FFN hidden dimension multiplier (relative to latent_dim).
        dropout_rate: Dropout rate for all dropout layers (default 0.0 = disabled).
        pooling: Output pooling strategy ('decode', 'mean', 'first', or 'none').
        interval_chrm_dim: Chromosome embedding dimension.
        interval_size_dim: Size encoding dimension.
        interval_center_dim: Center PE dimension.
        num_chrms: Number of chromosomes.
        max_wavelength: Max wavelength for genomic PE.
        key: PRNG key for initialization.

    Returns:
        Configured CModel.
    """
    if key is None:
        key = jax.random.key(0)

    keys = jax.random.split(key, 2)
    interval_encoder = GenomicIntervalEncoder(
        chrm_dim=interval_chrm_dim,
        size_dim=interval_size_dim,
        center_dim=interval_center_dim,
        num_chrms=num_chrms,
        max_wavelength=max_wavelength,
        key=keys[0],
    )

    return CModel(
        seq_dim=seq_dim,
        interval_encoder=interval_encoder,
        latent_dim=latent_dim,
        num_latents=num_latents,
        shared_per_stack=shared_per_stack,
        num_stacks=num_stacks,
        num_heads=num_heads,
        input_ff_mult=input_ff_mult,
        encoder_ff_hidden_dim=latent_dim * encoder_ff_mult,
        output_dim=output_dim,
        output_ff_hidden_dim=latent_dim * output_ff_mult,
        dropout_rate=dropout_rate,
        pooling=pooling,
        key=keys[1],
    )
