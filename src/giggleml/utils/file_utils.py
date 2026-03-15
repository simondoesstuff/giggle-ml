from os import PathLike
from pathlib import Path

type Pathish = PathLike[str] | str


def file_ext(file: Pathish) -> str:
    """the final file suffix, excluding the period"""
    return Path(file).suffix[1:]
