import gzip as gzip_module
from collections.abc import Iterator, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Int

from giggleml.types import GenomicInterval
from giggleml.utils.file_utils import Pathish, file_ext

DEFAULT_CHROMOSOMES: tuple[str, ...] = (
    "chr1",
    "chr2",
    "chr3",
    "chr4",
    "chr5",
    "chr6",
    "chr7",
    "chr8",
    "chr9",
    "chr10",
    "chr11",
    "chr12",
    "chr13",
    "chr14",
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr19",
    "chr20",
    "chr21",
    "chr22",
    "chrX",
    "chrY",
    "chrM",
)


def load_bed(path: Pathish, *, gzip: bool | None = None) -> Iterator[GenomicInterval]:
    """Load genomic intervals from a BED file.

    Args:
        path: Path to the BED file (.bed or .gz compressed).
        gzip: Whether file is gzip compressed. If None, inferred from extension.

    Yields:
        GenomicInterval tuples (chrom, start, end).
    """
    is_gzip = file_ext(path) == "gz" if gzip is None else gzip
    open_fn = gzip_module.open if is_gzip else open

    with open_fn(path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            yield fields[0], int(fields[1]), int(fields[2])


def interval_to_array(
    interval: GenomicInterval,
    chromosomes: Sequence[str] = DEFAULT_CHROMOSOMES,
) -> Int[Array, "3"]:
    """Convert a GenomicInterval to an array [chrom_idx, start, end].

    Args:
        interval: A (chrom, start, end) tuple.
        chromosomes: Sequence mapping chromosome index to name.

    Returns:
        Array of shape (3,) with [chrom_idx, start, end].
    """
    chrom, start, end = interval
    chrom_idx = chromosomes.index(chrom)
    return jnp.array([chrom_idx, start, end], dtype=jnp.int32)


def load_bed_array(
    path: Pathish,
    *,
    gzip: bool | None = None,
    chromosomes: Sequence[str] = DEFAULT_CHROMOSOMES,
) -> Int[Array, "intervals 3"]:
    """Load genomic intervals from a BED file as an array.

    Args:
        path: Path to the BED file (.bed or .gz compressed).
        gzip: Whether file is gzip compressed. If None, inferred from extension.
        chromosomes: Sequence mapping chromosome index to name.

    Returns:
        Array of shape (intervals, 3) with columns [chrom_idx, start, end].
    """
    # Optimized for JAX: Build a standard Python list of lists first,
    # then cast to a JAX array once to minimize accelerator overhead.
    raw_intervals = [
        [chromosomes.index(iv[0]), iv[1], iv[2]] for iv in load_bed(path, gzip=gzip)
    ]
    return jnp.array(raw_intervals, dtype=jnp.int32)
