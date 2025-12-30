import gzip
import random
from collections.abc import Iterator, Sequence
from functools import cached_property
from pathlib import Path
from typing import Callable, Protocol, final, override

from giggleml.utils.path_utils import as_path

from ..utils.types import GenomicInterval, PathLike, lazy
from .kind_dataset import KindDataset


class IntervalDataset(Protocol):
    @property
    def associated_fasta_path(self) -> Path | None: ...

    def __len__(self) -> int: ...

    # def __getitem__(self, idx: int) -> GenomicInterval: ...

    def __iter__(self) -> Iterator[GenomicInterval]: ...


@lazy
class LateIntervalDataset(KindDataset[GenomicInterval]):
    """
    An IntervalDataset that treats the backing intervals with a lazy getter
    function. This enables it to be passed to other processes without incurring
    a serialization tax.

    @param lazyLength: None implies inferred from size of lazyGetter()
    """

    def __init__(
        self,
        lazy_getter: Callable[[], list[GenomicInterval] | tuple[list[GenomicInterval], list[list[str]]]],
        lazy_length: Callable[[], int] | int | None,
        associated_fasta_path: PathLike | None,
    ):
        self.lazy_getter: Callable[[], list[GenomicInterval] | tuple[list[GenomicInterval], list[list[str]]]] = lazy_getter
        self.lazy_length: Callable[[], int] | int | None = lazy_length
        self.associated_fasta_path: Path | None = as_path(associated_fasta_path)

    @cached_property
    def _full_content(self):
        result = self.lazy_getter()
        if isinstance(result, tuple) and len(result) == 2:
            return result  # Return (intervals, extra_columns)
        # For backward compatibility, if no extra columns, return empty lists
        return result, [[] for _ in range(len(result))]

    @cached_property
    def _length(self):
        if self.lazy_length is None:
            intervals, _ = self._full_content
            return len(intervals)
        elif isinstance(self.lazy_length, int):
            return self.lazy_length
        else:
            return self.lazy_length()

    @override
    def __len__(self):
        return self._length

    @override
    def __getitem__(self, idx: int):
        intervals, _ = self._full_content
        return intervals[idx]

    def iter_with_extra_columns(self) -> Iterator[tuple[GenomicInterval, list[str]]]:
        """Iterate over intervals with their additional columns."""
        intervals, extra_columns = self._full_content
        return iter(zip(intervals, extra_columns))
    
    @override
    def __iter__(self) -> Iterator[GenomicInterval]:
        return (interval for interval, _ in self.iter_with_extra_columns())


@final
class MemoryIntervalDataset(KindDataset[GenomicInterval]):
    def __init__(
        self,
        intervals: Sequence[GenomicInterval],
        associated_fasta_path: PathLike | None = None,
        extra_columns: Sequence[list[str]] | None = None,
    ):
        super().__init__()
        self.intervals = intervals
        self.associated_fasta_path = as_path(associated_fasta_path)
        self.extra_columns = extra_columns or [[] for _ in range(len(intervals))]

    @override
    def __len__(self):
        return len(self.intervals)

    @override
    def __getitem__(self, idx: int) -> GenomicInterval:
        return self.intervals[idx]

    def iter_with_extra_columns(self) -> Iterator[tuple[GenomicInterval, list[str]]]:
        """Iterate over intervals with their additional columns."""
        return iter(zip(self.intervals, self.extra_columns))
    
    @override
    def __iter__(self) -> Iterator[GenomicInterval]:
        return (interval for interval, _ in self.iter_with_extra_columns())


@lazy
class BedDataset(LateIntervalDataset):
    def __init__(
        self,
        path: PathLike,
        associated_fasta_path: PathLike | None = None,
        limit: int | None = None,
        shuffle: bool = False,
    ):
        """
        Bed files are parsed lazily (so this passes pickling barrier).
        @param path: can be either a .bed or .bed.gz file.
        @param shuffle: reads full file before shuffling even when used with a read limit
        """
        self.path: Path = as_path(path)
        self.shuffle: bool = shuffle
        self.limit: float = limit or float("inf")

        super().__init__(
            lazy_getter=self._load,
            lazy_length=None,
            associated_fasta_path=associated_fasta_path,
        )

    def _load(self):
        random.seed(42)

        def parse(file):
            intervals = list[GenomicInterval]()
            extra_columns = list[list[str]]()

            for line in file:
                if not self.shuffle and len(intervals) >= self.limit:
                    break

                if line.startswith("#"):
                    continue

                parts = line.strip().split("\t")
                name, start, stop = parts[:3]
                intervals.append((name, int(start), int(stop)))
                
                # Store any additional columns beyond the first 3
                extra_columns.append(parts[3:] if len(parts) > 3 else [])

            if self.shuffle:
                # Shuffle both lists together
                combined = list(zip(intervals, extra_columns))
                random.shuffle(combined)
                intervals, extra_columns = zip(*combined) if combined else ([], [])
                intervals = list(intervals)
                extra_columns = list(extra_columns)

                if self.limit != float("inf"):
                    # when shuffling, we apply the limit after a full read
                    intervals = intervals[: self.limit]
                    extra_columns = extra_columns[: self.limit]

            return intervals, extra_columns

        if self.path.suffixes[-2:] == [".bed", ".gz"]:
            with gzip.open(self.path, "rt") as file:  # 'rt' mode for text reading
                return parse(file)
        elif self.path.suffix == ".bed":
            with open(self.path, "r") as file:
                return parse(file)
        else:
            raise ValueError("BedDataset expects inputs of either .bed.gz or .bed")
