"""GenomicIntervalEncoder: Embed genomic intervals (chrm, start, end) using Equinox.

Encoding strategy:
- Chromosome: learned embedding (breaks spatial bias between chromosomes)
- Size: logarithmic encoding (handles wide range of interval sizes)
- Center: sinusoidal positional encoding (genomic-scale wavelengths)

Components are concatenated (not summed) since they're informatively orthogonal.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


def genomic_sinusoidal_pe(
    position: Float[Array, ""],
    dim: int,
    max_wavelength: float = 250_000_000.0,
) -> Float[Array, "dim"]:
    """Compute sinusoidal PE with wavelengths spanning 1bp to max_wavelength.

    Uses log-spaced wavelengths appropriate for genomic coordinates (up to ~250Mbp
    per chromosome). Standard transformer PE with base 10000 is insufficient for
    positions in the hundreds of millions.

    Args:
        position: Genomic coordinate.
        dim: Output dimension (should be even).
        max_wavelength: Maximum wavelength in bp (default: 250M for human chromosomes).

    Returns:
        Positional encoding vector of shape (dim,).
    """
    half_dim = dim // 2

    # Log-spaced wavelengths from 1bp to max_wavelength
    log_wavelengths = jnp.linspace(0.0, jnp.log(max_wavelength), half_dim)
    wavelengths = jnp.exp(log_wavelengths)

    # Angular frequencies: 2π / wavelength
    frequencies = (2.0 * jnp.pi) / wavelengths

    # Compute sin and cos
    angles = position * frequencies
    pe_sin = jnp.sin(angles)
    pe_cos = jnp.cos(angles)

    # Interleave sin and cos
    pe = jnp.stack([pe_sin, pe_cos], axis=-1).reshape(-1)

    # Handle odd dimensions
    if dim % 2 == 1:
        pe = jnp.concatenate([pe, jnp.zeros(1)])

    return pe


class GenomicIntervalEncoder(eqx.Module):
    """Encode genomic intervals (chromosome, start, end) into embeddings.

    Components (concatenated):
        - Chromosome: learned embedding to avoid spatial bias between chromosomes
        - Size: log(end - start) projected to embedding space
        - Center: sinusoidal PE of interval midpoint (genomic-scale wavelengths)

    Output dimension = chrm_dim + size_dim + center_dim
    """

    chrm_embedding: eqx.nn.Embedding
    size_proj: eqx.nn.Linear
    chrm_dim: int = eqx.field(static=True)
    size_dim: int = eqx.field(static=True)
    center_dim: int = eqx.field(static=True)
    num_chrms: int = eqx.field(static=True)
    max_wavelength: float = eqx.field(static=True)

    def __init__(
        self,
        chrm_dim: int = 16,
        size_dim: int = 16,
        center_dim: int = 32,
        num_chrms: int = 24,
        max_wavelength: float = 250_000_000.0,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the genomic interval encoder.

        Args:
            chrm_dim: Dimension for chromosome embedding.
            size_dim: Dimension for size encoding.
            center_dim: Dimension for center PE (should be even).
            num_chrms: Number of chromosomes (default 24 for human: 1-22, X, Y).
            max_wavelength: Max wavelength for PE in bp (default: 250M).
            key: PRNG key for initialization.
        """
        super().__init__()
        self.chrm_dim = chrm_dim
        self.size_dim = size_dim
        self.center_dim = center_dim
        self.num_chrms = num_chrms
        self.max_wavelength = max_wavelength

        keys = jax.random.split(key, 2)

        self.chrm_embedding = eqx.nn.Embedding(num_chrms, chrm_dim, key=keys[0])
        self.size_proj = eqx.nn.Linear(1, size_dim, key=keys[1])

    @property
    def dim(self) -> int:
        """Total output dimension."""
        return self.chrm_dim + self.size_dim + self.center_dim

    def __call__(
        self,
        interval: Int[Array, "3"],
    ) -> Float[Array, "dim"]:
        """Encode a single genomic interval.

        Args:
            interval: Array of [chrm, start, end] where:
                - chrm: Chromosome index (0-indexed)
                - start: Interval start position (0-indexed, inclusive)
                - end: Interval end position (0-indexed, exclusive)

        Returns:
            Embedding vector of shape (dim,).
        """
        chrm, start, end = interval[0], interval[1], interval[2]

        # Chromosome embedding
        chrm_emb = self.chrm_embedding(chrm)

        # Log-size encoding
        size = jnp.maximum(end - start, 1)  # Avoid log(0)
        log_size = jnp.log(size.astype(jnp.float32))
        size_emb = self.size_proj(log_size[None])

        # Center sinusoidal PE
        center = (start + end).astype(jnp.float32) / 2.0
        center_emb = genomic_sinusoidal_pe(center, self.center_dim, self.max_wavelength)

        return jnp.concatenate([chrm_emb, size_emb, center_emb])

    def encode_batch(
        self,
        intervals: Int[Array, "batch 3"],
    ) -> Float[Array, "batch dim"]:
        """Encode a batch of genomic intervals.

        Args:
            intervals: Array of shape (batch, 3) where each row is [chrm, start, end].

        Returns:
            Embedding vectors of shape (batch, dim).
        """
        return jax.vmap(self.__call__)(intervals)
