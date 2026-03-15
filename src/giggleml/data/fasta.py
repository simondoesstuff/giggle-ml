from collections.abc import Iterable, Iterator, Sequence
from typing import cast, overload

import jax
import pyfastx
from jaxtyping import Array, Int

from giggleml.data.intervals import DEFAULT_CHROMOSOMES
from giggleml.types import GenomicInterval
from giggleml.utils.file_utils import Pathish, file_ext
from giggleml.utils.lru_cache import lru_cache


@lru_cache
def load_fasta(path: Pathish, *, gzip: bool | None = None) -> dict[str, str]:
    """Load a FASTA file into an in-memory dict mapping sequence names to sequences.

    Uses pyfastx for efficient parsing. Creates an index file alongside the FASTA
    if one doesn't exist.

    Args:
        path: Path to the FASTA file (.fa, .fasta, or .gz compressed).
        gzip: Whether file is gzip compressed. If None, inferred from extension.

    Returns:
        Dictionary mapping sequence names to their sequences.
    """
    is_gzip = file_ext(path) == "gz" if gzip is None else gzip
    fa = pyfastx.Fasta(str(path), build_index=not is_gzip)
    return {seq.name: seq.seq for seq in fa}


def fasta_keys(fasta: dict[str, str]) -> list[str]:
    """Get ordered chromosome keys from a fasta dict.

    Args:
        fasta: Dictionary mapping chromosome names to sequences.

    Returns:
        List of chromosome names in insertion order.
    """
    return list(fasta.keys())


@overload
def fasta_map(
    fasta: dict[str, str], intervals: Iterable[GenomicInterval]
) -> Iterator[str]: ...


@overload
def fasta_map(
    fasta: dict[str, str],
    intervals: Int[Array, "intervals 3"],
    *,
    chromosomes: Sequence[str] = DEFAULT_CHROMOSOMES,
) -> Iterator[str]: ...


def fasta_map(
    fasta: dict[str, str],
    intervals: Iterable[GenomicInterval] | Int[Array, "intervals 3"],
    *,
    chromosomes: Sequence[str] = DEFAULT_CHROMOSOMES,
) -> Iterator[str]:
    """Extract sequences from a FASTA dict for genomic intervals.

    Args:
        fasta: Dictionary mapping chromosome names to sequences.
        intervals: Either nested iterable of (chrom, start, end) intervals,
            or an Array of shape (intervals, 3) with columns [chrom_idx, start, end].
        chromosomes: Sequence mapping chromosome index to name (for array input).

    Returns:
        Iterators of extracted sequences.
    """
    if isinstance(intervals, jax.Array):
        # Convert the entire JAX array to a Python list of lists at once.
        # This avoids the massive host-accelerator dispatch overhead of
        # iterating through JAX arrays row-by-row in standard Python logic.
        for row in intervals.tolist():
            chrom_idx, start, end = cast(tuple[int, int, int], row)
            yield fasta[chromosomes[chrom_idx]][start:end]
    else:
        for chrom, start, end in intervals:
            yield fasta[chrom][start:end]
