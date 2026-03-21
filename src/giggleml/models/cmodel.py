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


class ChunkedCrossAttention(eqx.Module):
    """Cross-attention with chunking over key/value dimension for memory efficiency.

    When inputs (N) >> latents (M), chunking over N reduces peak memory from
    O(M * N) to O(M * chunk_size) while maintaining numerical correctness via
    the log-sum-exp trick for softmax aggregation.
    """

    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    ln: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        input_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        chunk_size: int = 1024,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.chunk_size = chunk_size

        keys = jax.random.split(key, 4)
        self.query_proj = eqx.nn.Linear(latent_dim, latent_dim, key=keys[0])
        self.key_proj = eqx.nn.Linear(input_dim, latent_dim, key=keys[1])
        self.value_proj = eqx.nn.Linear(input_dim, latent_dim, key=keys[2])
        self.output_proj = eqx.nn.Linear(latent_dim, latent_dim, key=keys[3])
        self.ln = eqx.nn.LayerNorm(latent_dim)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def _chunked_attention(
        self,
        q: Float[Array, "num_latents num_heads head_dim"],
        k: Float[Array, "input_len num_heads head_dim"],
        v: Float[Array, "input_len num_heads head_dim"],
        mask: Bool[Array, "input_len"] | None,
    ) -> Float[Array, "num_latents num_heads head_dim"]:
        """Compute attention with chunking over key/value dimension."""
        num_latents = q.shape[0]
        input_len = k.shape[0]
        scale = self.head_dim**-0.5

        # Initialize accumulators for online softmax
        # Shape: (num_latents, num_heads)
        max_scores = jnp.full((num_latents, self.num_heads), -jnp.inf)
        sum_exp = jnp.zeros((num_latents, self.num_heads))
        # Shape: (num_latents, num_heads, head_dim)
        output = jnp.zeros((num_latents, self.num_heads, self.head_dim))

        def process_chunk(
            carry: tuple[Array, Array, Array], chunk_idx: Array
        ) -> tuple[tuple[Array, Array, Array], None]:
            max_scores, sum_exp, output = carry
            start = chunk_idx * self.chunk_size
            end = jnp.minimum(start + self.chunk_size, input_len)

            # Dynamic slice for keys and values
            k_chunk = jax.lax.dynamic_slice(
                k, (start, 0, 0), (self.chunk_size, self.num_heads, self.head_dim)
            )
            v_chunk = jax.lax.dynamic_slice(
                v, (start, 0, 0), (self.chunk_size, self.num_heads, self.head_dim)
            )

            # Compute attention scores: (num_latents, num_heads, chunk_size)
            scores: Array = einx.dot("m h d, n h d -> m h n", q, k_chunk) * scale

            # Apply mask if provided (True = masked out)
            if mask is not None:
                chunk_mask = jax.lax.dynamic_slice(mask, (start,), (self.chunk_size,))
                # Create position mask for padding beyond input_len
                pos_mask = jnp.arange(self.chunk_size) >= (end - start)
                combined_mask = chunk_mask | pos_mask
                scores = jnp.where(combined_mask[None, None, :], -jnp.inf, scores)
            else:
                # Still need to mask positions beyond input_len
                pos_mask = jnp.arange(self.chunk_size) >= (end - start)
                scores = jnp.where(pos_mask[None, None, :], -jnp.inf, scores)

            # Online softmax update (log-sum-exp trick)
            chunk_max = scores.max(axis=-1)  # (num_latents, num_heads)
            new_max = jnp.maximum(max_scores, chunk_max)

            # Rescale factors - handle -inf safely
            # When new_max is -inf (all positions masked so far), set scales to 0
            # to avoid NaN from exp(-inf - (-inf))
            new_max_is_neg_inf = new_max == -jnp.inf
            scale_prev = jnp.where(
                new_max_is_neg_inf, 0.0, jnp.exp(max_scores - new_max)
            )
            scale_curr = jnp.where(
                new_max_is_neg_inf, 0.0, jnp.exp(chunk_max - new_max)
            )

            # Compute exp(scores - chunk_max) for numerical stability
            # When chunk_max is -inf, set to 0 to avoid NaN (will be scaled by 0 anyway)
            chunk_max_is_neg_inf = chunk_max == -jnp.inf
            scores_shifted = jnp.where(
                chunk_max_is_neg_inf[:, :, None], 0.0, scores - chunk_max[:, :, None]
            )
            exp_scores = jnp.exp(scores_shifted)
            chunk_sum = exp_scores.sum(axis=-1)  # (num_latents, num_heads)

            # Update accumulators
            sum_exp = sum_exp * scale_prev + chunk_sum * scale_curr

            # Weighted values: (num_latents, num_heads, head_dim)
            weighted_v: Array = einx.dot("m h n, n h d -> m h d", exp_scores, v_chunk)
            output = output * scale_prev[:, :, None] + weighted_v * scale_curr[:, :, None]

            return (new_max, sum_exp, output), None

        num_chunks = (input_len + self.chunk_size - 1) // self.chunk_size
        (max_scores, sum_exp, output), _ = jax.lax.scan(
            process_chunk,
            (max_scores, sum_exp, output),
            jnp.arange(num_chunks),
        )

        # Final normalization - avoid division by zero when all positions masked
        # Replace 0 with 1 to get 0/1=0 with valid gradients (jnp.where computes
        # gradients through both branches, so division by 0 would give NaN grads)
        sum_exp_safe = jnp.where(sum_exp > 0, sum_exp, 1.0)
        output = output / sum_exp_safe[:, :, None]
        return output

    def __call__(
        self,
        latents: Float[Array, "num_latents latent_dim"],
        inputs: Float[Array, "input_len input_dim"],
        mask: Bool[Array, "input_len"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "num_latents latent_dim"]:
        num_latents = latents.shape[0]
        input_len = inputs.shape[0]

        # Pre-norm on queries
        x = jax.vmap(self.ln)(latents)

        # Project to multi-head format
        q = jax.vmap(self.query_proj)(x)  # (num_latents, latent_dim)
        k = jax.vmap(self.key_proj)(inputs)  # (input_len, latent_dim)
        v = jax.vmap(self.value_proj)(inputs)  # (input_len, latent_dim)

        # Reshape to (seq_len, num_heads, head_dim)
        q = q.reshape(num_latents, self.num_heads, self.head_dim)
        k = k.reshape(input_len, self.num_heads, self.head_dim)
        v = v.reshape(input_len, self.num_heads, self.head_dim)

        # Pad k and v to be divisible by chunk_size for scan efficiency
        pad_len = (self.chunk_size - input_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            k = jnp.pad(k, ((0, pad_len), (0, 0), (0, 0)))
            v = jnp.pad(v, ((0, pad_len), (0, 0), (0, 0)))
            if mask is not None:
                # Pad mask with True (masked out)
                mask = jnp.pad(mask, (0, pad_len), constant_values=True)

        # Chunked attention
        attn_out = self._chunked_attention(q, k, v, mask)

        # Reshape back and project
        attn_out = attn_out.reshape(num_latents, -1)
        attn_out = jax.vmap(self.output_proj)(attn_out)

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
    cross_attn: ChunkedCrossAttention
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
        cross_attn_chunk_size: int = 1024,
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

        self.cross_attn = ChunkedCrossAttention(
            latent_dim, input_dim, num_heads, dropout_rate, cross_attn_chunk_size, key=keys[2]
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
    cross_attn_chunk_size: int = 1024,
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
        cross_attn_chunk_size: Chunk size for chunked cross-attention (default 1024).
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
        cross_attn_chunk_size=cross_attn_chunk_size,
        key=keys[1],
    )
