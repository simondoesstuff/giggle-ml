from collections.abc import Iterable

import torch

from giggleml.utils.types import GenomicInterval


def intersect(x: GenomicInterval, y: GenomicInterval) -> GenomicInterval | None:
    ch1, start1, end1 = x
    ch2, start2, end2 = y

    if ch1 != ch2:
        return None

    start = max(start1, start2)
    end = min(end1, end2)

    if start >= end:
        return None

    return (ch1, start, end)


def overlap_degree(x: GenomicInterval, y: GenomicInterval) -> int:
    overlap = intersect(x, y)

    if not overlap:
        return 0

    return overlap[2] - overlap[1]


def intervals_to_tensor(
    intervals: Iterable[GenomicInterval],
    dtype: torch.dtype = torch.int32,
    pin_memory: bool = True,
) -> torch.Tensor:
    special_chrms = {
        "x": 23,
        "y": 24,
        "xy": 25,  # PAR
        "par": 25,
        "m": 26,
        "mt": 26,
    }

    def _chrm_id(chrm):
        if not chrm.startswith("chr"):
            raise ValueError(f"Unknown chromosome {chrm}")

        id = chrm[3:].lower()

        if (special := special_chrms.get(id, None)) is not None:
            return special
        return int(id) - 1

    clean_intervals = [(_chrm_id(chrm), start, end) for (chrm, start, end) in intervals]
    return torch.tensor(clean_intervals, dtype=dtype, pin_memory=pin_memory)
