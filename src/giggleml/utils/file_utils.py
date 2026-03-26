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


def possibly_gzipped(file: Pathish) -> Path:
    path = Path(file)

    if path.exists():
        return path

    if path.suffix == ".gz":
        alt_path = path.with_suffix("")
    else:
        alt_path = path.with_name(path.name + ".gz")

    if alt_path.exists():
        return alt_path

    return path
