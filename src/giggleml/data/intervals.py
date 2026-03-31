import atexit
import gzip as gzip_module
import warnings
from collections.abc import Iterable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from giggleml.types import GenomicInterval
from giggleml.utils.file_utils import Pathish, file_ext

# Track chromosomes skipped due to not being in DEFAULT_CHROMOSOMES
_skipped_chromosomes: set[str] = set()


def _warn_skipped_chromosomes() -> None:
    """Warn about skipped chromosomes at exit."""
    if _skipped_chromosomes:
        warnings.warn(
            f"Skipped {len(_skipped_chromosomes)} unknown chromosome(s): "
            f"{', '.join(sorted(_skipped_chromosomes))}"
        )


atexit.register(_warn_skipped_chromosomes)

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


def filter_chromosomes(
    intervals: Iterable[GenomicInterval],
    chromosomes: Sequence[str] = DEFAULT_CHROMOSOMES,
) -> Iterator[GenomicInterval]:
    """Filter intervals to only include those on known chromosomes.

    Unknown chromosomes are tracked and warned about at program exit.

    Args:
        intervals: Iterable of (chrom, start, end) tuples.
        chromosomes: Sequence of valid chromosome names.

    Yields:
        GenomicInterval tuples on known chromosomes.
    """
    chrom_set = set(chromosomes)
    for iv in intervals:
        if iv[0] in chrom_set:
            yield iv
        else:
            _skipped_chromosomes.add(iv[0])


def interval_to_array(
    interval: GenomicInterval,
    chromosomes: Sequence[str] = DEFAULT_CHROMOSOMES,
) -> NDArray[np.int32]:
    """Convert a GenomicInterval to an array [chrom_idx, start, end].

    Args:
        interval: A (chrom, start, end) tuple.
        chromosomes: Sequence mapping chromosome index to name.

    Returns:
        Array of shape (3,) with [chrom_idx, start, end].
    """
    chrom, start, end = interval
    chrom_idx = chromosomes.index(chrom)
    return np.array([chrom_idx, start, end], dtype=np.int32)


def load_bed_array(
    path: Pathish,
    *,
    gzip: bool | None = None,
    chromosomes: Sequence[str] = DEFAULT_CHROMOSOMES,
) -> NDArray[np.int32]:
    """Load genomic intervals from a BED file as a numpy array.

    Intervals on chromosomes not in the chromosomes list are skipped.
    Unknown chromosomes are tracked and warned about at program exit.

    Args:
        path: Path to the BED file (.bed or .gz compressed).
        gzip: Whether file is gzip compressed. If None, inferred from extension.
        chromosomes: Sequence mapping chromosome index to name.

    Returns:
        Numpy array of shape (intervals, 3) with columns [chrom_idx, start, end].
    """
    chrom_set = set(chromosomes)
    raw_intervals: list[list[int]] = []

    for iv in load_bed(path, gzip=gzip):
        if iv[0] in chrom_set:
            raw_intervals.append([chromosomes.index(iv[0]), iv[1], iv[2]])
        else:
            _skipped_chromosomes.add(iv[0])

    return np.array(raw_intervals, dtype=np.int32)


def sorted_by_size(
    intervals: Sequence[GenomicInterval], descending: bool = False
) -> Iterator[GenomicInterval]:
    """Lazily yield intervals sorted by size (smallest first).

    Takes a Sequence to signal that input must be materialized. Returns an
    Iterator to signal single-use (not streaming). Sorting is deferred until
    the first element is pulled.

    Args:
        intervals: Materialized sequence of (chrom, start, end) tuples.
        descending: Sort descending

    Yields:
        GenomicInterval tuples sorted by (end - start) ascending.
    """
    sign = -1 if descending else 1
    yield from sorted(intervals, key=lambda iv: sign * (iv[2] - iv[1]))


def crop_intervals(
    intervals: Iterable[GenomicInterval],
    size: int,
    centers: Iterable[int] | None = None,
) -> Iterator[GenomicInterval]:
    """Crop genomic intervals to a fixed size.

    Args:
        intervals: Iterable of (chrom, start, end) tuples.
        size: Target size in base pairs for each interval.
        centers: Optional centers for cropping. If provided, crop symmetrically
            around each center. If None, use interval start as anchor and take
            `size` bp to the right.

    Yields:
        Cropped GenomicInterval tuples (chrom, start, end).
    """
    if centers is None:
        for chrom, start, end in intervals:
            yield chrom, start, min(start + size, end)
    else:
        half = size // 2
        for (chrom, start, end), center in zip(intervals, centers):
            new_start = max(center - half, start)
            yield chrom, new_start, min(new_start + size, end)
