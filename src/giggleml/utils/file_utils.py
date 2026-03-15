from os import PathLike
from pathlib import Path

type Pathish = PathLike[str] | str


def file_ext(file: Pathish) -> str:
    """the final file suffix, excluding the period"""
    return Path(file).suffix[1:]


def file_stem(file: Pathish, remove_gz: bool = True) -> str:
    """the file stem (name without extension), optionally removing .gz"""
    p = Path(file)
    stem = p.stem
    if remove_gz and p.suffix == ".gz":
        stem = Path(stem).stem
    return stem
