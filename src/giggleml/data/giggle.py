"""Pythonic wrapper for the giggle genomic interval search tool."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from giggleml.utils.file_utils import Pathish


def _check_tool_in_path(tool: str) -> None:
    """Raise RuntimeError if tool is not in PATH."""
    if shutil.which(tool) is None:
        raise RuntimeError(f"{tool} not found in PATH")


@dataclass
class GiggleResult:
    """Result from a giggle search query.

    Field names match giggle's output header.
    """

    file: str
    file_size: int
    overlaps: int
    odds_ratio: float
    fishers_two_tail: float
    fishers_left_tail: float
    fishers_right_tail: float
    combo_score: float


class GiggleIndex:
    """Pythonic wrapper for giggle index operations.
    WARN: does not support modification after init.

    Lazily creates and maintains a giggle index of all BED files in a directory.
    Assumes the index is complete if it already exists.

    Args:
        directory: Path to directory containing BED files to index.
        index_dir: Path to store the giggle index. Defaults to {directory}.giggle
        sorted: Whether the input BED files are already sorted. Passed to giggle index -s.
        genome_size: Genome size for significance testing (default: human genome).
    """

    _directory: Path
    _index_dir_path: Path
    _sorted: bool
    _genome_size: int | None

    def __init__(
        self,
        directory: Pathish,
        index_dir: Pathish | None = None,
        *,
        sorted: bool = False,
        genome_size: int | None = None,
    ) -> None:
        _check_tool_in_path("giggle")
        _check_tool_in_path("bgzip")

        self._directory = Path(directory)
        if not self._directory.is_dir():
            raise ValueError(f"Directory does not exist: {self._directory}")

        self._index_dir_path = (
            Path(index_dir) if index_dir else self._directory.with_suffix(".giggle")
        )
        self._sorted = sorted
        self._genome_size = genome_size

    @property
    def directory(self) -> Path:
        """The directory containing indexed BED files."""
        return self._directory

    @property
    def genome_size(self) -> int | None:
        """Genome size for significance testing."""
        return self._genome_size

    @property
    def index_dir(self) -> Path:
        """The giggle index directory. Call build_index() first if it doesn't exist."""
        return self._index_dir_path

    @property
    def exists(self) -> bool:
        """Whether the index already exists."""
        return self._index_dir_path.is_dir()

    def build_index(self, *, force: bool = False) -> Path:
        """Build the giggle index.

        Args:
            force: If True, rebuild even if index already exists.

        Returns:
            Path to the index directory.
        """
        if force and self._index_dir_path.is_dir():
            shutil.rmtree(self._index_dir_path)

        if not self._index_dir_path.is_dir():
            cmd = [
                "giggle",
                "index",
                "-i",
                str(self._directory / "*.bed.gz"),
                "-o",
                str(self._index_dir_path),
            ]
            if self._sorted:
                cmd.append("-s")
            subprocess.run(cmd, check=True, capture_output=True, text=True)

        return self._index_dir_path

    @cached_property
    def list_beds(self) -> set[str]:
        """All BED file names in the index.

        Returns:
            Set of BED file basenames (without path).
        """
        result = subprocess.run(
            ["giggle", "search", "-i", str(self.index_dir), "-l"],
            check=True,
            capture_output=True,
            text=True,
        )
        beds = set()
        for line in result.stdout.strip().split("\n"):
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts:
                filename = Path(parts[0]).name
                if filename:
                    beds.add(filename)
        return beds

    def _parse_output(self, output: str) -> list[GiggleResult]:
        """Parse giggle search -s output into GiggleResult objects.

        Header: #file\tfile_size\toverlaps\todds_ratio\tfishers_two_tail\tfishers_left_tail\tfishers_right_tail\tcombo_score
        """
        results = []
        for line in output.strip().split("\n"):
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 8:
                continue

            try:
                results.append(
                    GiggleResult(
                        file=Path(parts[0]).name,
                        file_size=int(parts[1]),
                        overlaps=int(parts[2]),
                        odds_ratio=float(parts[3]),
                        fishers_two_tail=float(parts[4]),
                        fishers_left_tail=float(parts[5]),
                        fishers_right_tail=float(parts[6]),
                        combo_score=float(parts[7]),
                    )
                )
            except (IndexError, ValueError):
                continue

        return results

    def query(self, query_file: Pathish) -> list[GiggleResult]:
        """Query the index with a BED file.

        Args:
            query_file: Path to a BED file to query against the index.

        Returns:
            List of GiggleResult objects with match statistics.
        """
        query_path = Path(query_file)
        if not query_path.is_file():
            raise ValueError(f"Query file does not exist: {query_path}")

        needs_bgzip = not query_path.name.endswith(".gz")
        temp_file = None

        try:
            if needs_bgzip:
                temp_file = tempfile.NamedTemporaryFile(suffix=".bed.gz", delete=False)
                temp_file.close()
                with open(temp_file.name, "wb") as f:
                    subprocess.run(
                        ["bgzip", "-c", str(query_path)],
                        stdout=f,
                        check=True,
                    )
                query_to_use = temp_file.name
            else:
                query_to_use = str(query_path)

            cmd = [
                "giggle",
                "search",
                "-i",
                str(self.index_dir),
                "-q",
                query_to_use,
                "-s",
            ]
            if self._genome_size is not None:
                cmd.extend(["-g", str(self._genome_size)])

            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return self._parse_output(result.stdout)

        finally:
            if temp_file is not None:
                Path(temp_file.name).unlink(missing_ok=True)

    def self_query(self, bed_name: str) -> list[GiggleResult]:
        """Query the index using a BED file already in the index.

        Useful for KNN-style queries where you want to find similar BED files.

        Args:
            bed_name: Name of a BED file in the index (as returned by list_beds).

        Returns:
            List of GiggleResult objects with match statistics.
        """
        beds = self.list_beds
        if bed_name not in beds:
            sample = list(beds)[:5]
            raise ValueError(
                f"BED file '{bed_name}' not found in index. "
                f"Available: {sample}{'...' if len(beds) > 5 else ''}"
            )

        query_path = self._directory / bed_name
        if not query_path.is_file():
            for suffix in [".gz", ""]:
                candidate = self._directory / f"{bed_name}{suffix}"
                if candidate.is_file():
                    query_path = candidate
                    break
            else:
                raise ValueError(f"Cannot find BED file for '{bed_name}' in directory")

        return self.query(query_path)

    def query_regions(self, regions: list[str] | str) -> list[GiggleResult]:
        """Query the index with specific regions.

        Args:
            regions: Region(s) in format "chr:start-end" or list of such strings.

        Returns:
            List of GiggleResult objects with match statistics.
        """
        if isinstance(regions, str):
            regions = [regions]

        regions_csv = ",".join(regions)
        cmd = ["giggle", "search", "-i", str(self.index_dir), "-r", regions_csv, "-s"]
        if self._genome_size is not None:
            cmd.extend(["-g", str(self._genome_size)])

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return self._parse_output(result.stdout)

    def reindex(self) -> None:
        """Force recreation of the index."""
        # Clear cached list_beds
        self.__dict__.pop("list_beds", None)
        self.build_index(force=True)
