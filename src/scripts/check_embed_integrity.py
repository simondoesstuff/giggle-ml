#!/usr/bin/env python3
"""Check integrity of zarr embeddings produced by embed_intervals.

Compares expected outputs (from bed files) against actual zarr files,
identifies missing files, corrupted arrays, and shape mismatches.

Usage:
    uv run src/scripts/check_embed_integrity.py data/roadmap_epigenomics/beds data/roadmap_epigenomics/embeds
    uv run src/scripts/check_embed_integrity.py --help
"""

import argparse
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm

from giggleml.utils.file_utils import file_stem


@dataclass
class IntegrityReport:
    """Summary of embedding integrity check."""

    total_expected: int
    total_found: int
    missing: list[str]
    corrupted: list[tuple[str, str]]  # (name, error)
    shape_mismatches: list[
        tuple[str, tuple[int, ...], int]
    ]  # (name, shape, expected_rows)
    has_nan: list[str]
    has_inf: list[str]
    empty: list[str]

    @property
    def n_problems(self) -> int:
        """Total number of problematic files."""
        return (
            len(self.missing)
            + len(self.corrupted)
            + len(self.shape_mismatches)
            + len(self.has_nan)
            + len(self.has_inf)
            + len(self.empty)
        )

    @property
    def is_healthy(self) -> bool:
        return (
            len(self.missing) == 0
            and len(self.corrupted) == 0
            and len(self.shape_mismatches) == 0
            and len(self.has_nan) == 0
            and len(self.has_inf) == 0
            and len(self.empty) == 0
        )

    def summary(self) -> str:
        lines = [
            "Embedding Integrity Report",
            "=" * 50,
            f"Expected files:     {self.total_expected}",
            f"Found files:        {self.total_found}",
            f"Missing files:      {len(self.missing)}",
            f"Corrupted files:    {len(self.corrupted)}",
            f"Shape mismatches:   {len(self.shape_mismatches)}",
            f"Files with NaN:     {len(self.has_nan)}",
            f"Files with Inf:     {len(self.has_inf)}",
            f"Empty files:        {len(self.empty)}",
            "=" * 50,
        ]
        if self.is_healthy:
            lines.append("Status: HEALTHY")
        else:
            lines.append("Status: ISSUES FOUND")
        return "\n".join(lines)


def iter_bed_files(bed_dir: Path) -> Iterator[Path]:
    """Iterate over bed files in directory."""
    for ext in ("*.bed", "*.bed.gz"):
        yield from bed_dir.glob(ext)


def is_zarr_array(path: Path) -> bool:
    """Check if a path is a valid zarr array (v2 or v3)."""
    if not path.exists():
        return False
    # Zarr v3: has zarr.json
    if (path / "zarr.json").exists():
        return True
    # Zarr v2: has .zarray
    if (path / ".zarray").exists():
        return True
    return False


def iter_zarr_arrays(embed_dir: Path) -> Iterator[Path]:
    """Iterate over zarr arrays in a directory.

    Detects both .zarr suffixed paths and plain directories containing zarr metadata.
    """
    for item in embed_dir.iterdir():
        if item.is_dir() and is_zarr_array(item):
            yield item
        elif item.suffix == ".zarr" and is_zarr_array(item):
            yield item


def count_bed_intervals(bed_path: Path) -> int:
    """Count intervals in a bed file (handles gzipped files)."""
    import gzip

    count = 0
    if bed_path.suffix == ".gz":
        f = gzip.open(bed_path, "rt")
    else:
        f = open(bed_path)
    with f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("track"):
                count += 1
    return count


def check_zarr_integrity(
    zarr_path: Path,
    expected_rows: int | None = None,
    *,
    thorough: bool = False,
) -> tuple[bool, str | None]:
    """Check if a zarr array is valid.

    Args:
        zarr_path: Path to the zarr array
        expected_rows: Expected number of rows (if known)
        thorough: If True, check ALL chunks for NaN/Inf (slower but complete)

    Returns:
        (is_valid, error_message)
    """
    if not zarr_path.exists():
        return False, "File does not exist"

    try:
        arr = zarr.open_array(zarr_path, mode="r")
    except Exception as e:
        return False, f"Failed to open: {e}"

    # Check shape
    if len(arr.shape) != 2:
        return False, f"Expected 2D array, got shape {arr.shape}"

    if expected_rows is not None and arr.shape[0] != expected_rows:
        return False, f"Row mismatch: expected {expected_rows}, got {arr.shape[0]}"

    # Check for empty
    if arr.shape[0] == 0:
        return False, "Array is empty (0 rows)"

    try:
        chunk_size = arr.chunks[0] if arr.chunks else arr.shape[0]

        if thorough:
            # Check ALL chunks for NaN/Inf
            for chunk_start in range(0, arr.shape[0], chunk_size):
                chunk_end = min(chunk_start + chunk_size, arr.shape[0])
                chunk_data = arr[chunk_start:chunk_end]
                if np.any(np.isnan(chunk_data)):
                    return False, f"Contains NaN values (chunk at row {chunk_start})"
                if np.any(np.isinf(chunk_data)):
                    return False, f"Contains Inf values (chunk at row {chunk_start})"
        else:
            # Sample check: first, middle, and last chunks
            n_rows = arr.shape[0]
            chunks_to_check = [0]  # First chunk

            # Middle chunk (if exists and different from first/last)
            if n_rows > chunk_size * 2:
                mid_start = (n_rows // 2 // chunk_size) * chunk_size
                chunks_to_check.append(mid_start)

            # Last chunk (if different from first)
            if n_rows > chunk_size:
                last_start = max(0, n_rows - chunk_size)
                if last_start not in chunks_to_check:
                    chunks_to_check.append(last_start)

            for chunk_start in chunks_to_check:
                chunk_end = min(chunk_start + chunk_size, n_rows)
                chunk_data = arr[chunk_start:chunk_end]
                if np.any(np.isnan(chunk_data)):
                    return False, f"Contains NaN values (chunk at row {chunk_start})"
                if np.any(np.isinf(chunk_data)):
                    return False, f"Contains Inf values (chunk at row {chunk_start})"

    except Exception as e:
        return False, f"Failed to read data: {e}"

    return True, None


def check_integrity(
    bed_dir: Path,
    embed_dir: Path,
    *,
    check_row_counts: bool = True,
    thorough: bool = False,
    verbose: bool = False,
) -> IntegrityReport:
    """Check integrity of embeddings against source bed files.

    Args:
        bed_dir: Directory containing source .bed or .bed.gz files
        embed_dir: Directory containing output .zarr files
        check_row_counts: Whether to verify row counts match bed intervals
        thorough: Whether to check ALL chunks for NaN/Inf (slower)
        verbose: Print progress information
    """
    bed_files = sorted(iter_bed_files(bed_dir))
    expected_stems = {file_stem(b): b for b in bed_files}

    if verbose:
        print(f"Found {len(expected_stems)} bed files in {bed_dir}")

    # Find all zarr arrays (with or without .zarr extension)
    zarr_files = sorted(iter_zarr_arrays(embed_dir))
    found_stems = {z.stem: z for z in zarr_files}

    if verbose:
        print(f"Found {len(found_stems)} zarr files in {embed_dir}")

    # Track issues
    missing: list[str] = []
    corrupted: list[tuple[str, str]] = []
    shape_mismatches: list[tuple[str, tuple[int, ...], int]] = []
    has_nan: list[str] = []
    has_inf: list[str] = []
    empty: list[str] = []

    # Check for missing files
    for stem in tqdm(expected_stems, desc="Checking for missing", leave=False):
        if stem not in found_stems:
            missing.append(stem)
            if verbose:
                tqdm.write(f"  MISSING: {stem}")

    # Check each found zarr file
    for stem, zarr_path in tqdm(
        found_stems.items(), desc="Checking zarr files", leave=False
    ):
        if stem not in expected_stems:
            if verbose:
                tqdm.write(f"  EXTRA: {stem} (no matching bed file)")
            continue

        # Get expected row count if checking
        expected_rows = None
        if check_row_counts:
            try:
                expected_rows = count_bed_intervals(expected_stems[stem])
            except Exception as e:
                if verbose:
                    tqdm.write(f"  WARNING: Could not count intervals in {stem}: {e}")

        is_valid, error = check_zarr_integrity(zarr_path, expected_rows, thorough=thorough)

        if not is_valid:
            assert error is not None
            if "does not exist" in error:
                missing.append(stem)
            elif "Failed to open" in error or "Failed to read" in error:
                corrupted.append((stem, error))
            elif "Row mismatch" in error:
                # Extract actual shape for reporting
                try:
                    arr = zarr.open_array(zarr_path, mode="r")
                    shape_mismatches.append((stem, arr.shape, expected_rows or 0))
                except Exception:
                    corrupted.append((stem, error))
            elif "NaN" in error:
                has_nan.append(stem)
            elif "Inf" in error:
                has_inf.append(stem)
            elif "empty" in error.lower():
                empty.append(stem)
            else:
                corrupted.append((stem, error))

            if verbose:
                tqdm.write(f"  ISSUE: {stem}: {error}")
        elif verbose:
            tqdm.write(f"  OK: {stem}")

    return IntegrityReport(
        total_expected=len(expected_stems),
        total_found=len(found_stems),
        missing=missing,
        corrupted=corrupted,
        shape_mismatches=shape_mismatches,
        has_nan=has_nan,
        has_inf=has_inf,
        empty=empty,
    )


def find_holes(
    bed_dir: Path,
    embed_dir: Path,
    *,
    check_row_counts: bool = True,
    thorough: bool = False,
) -> list[Path]:
    """Return list of bed files that need to be re-processed.

    This is useful for rerunning only the missing files.
    """
    report = check_integrity(
        bed_dir, embed_dir, check_row_counts=check_row_counts, thorough=thorough
    )

    bed_files = {file_stem(b): b for b in iter_bed_files(bed_dir)}

    # All problematic stems
    problem_stems = (
        set(report.missing)
        | {n for n, _ in report.corrupted}
        | {n for n, _, _ in report.shape_mismatches}
        | set(report.has_nan)
        | set(report.has_inf)
        | set(report.empty)
    )

    return [bed_files[stem] for stem in problem_stems if stem in bed_files]


def main():
    parser = argparse.ArgumentParser(
        description="Check integrity of zarr embeddings against source bed files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic integrity check (verifies row counts by default)
    uv run src/scripts/check_embed_integrity.py data/beds data/embeds

    # Fast check without row count verification
    uv run src/scripts/check_embed_integrity.py data/beds data/embeds --skip-row-check

    # Thorough check: verify ALL chunks for NaN/Inf (slowest)
    uv run src/scripts/check_embed_integrity.py data/beds data/embeds --thorough

    # List bed files that need reprocessing
    uv run src/scripts/check_embed_integrity.py data/beds data/embeds --list-holes
        """,
    )
    parser.add_argument("bed_dir", type=Path, help="Directory containing bed files")
    parser.add_argument(
        "embed_dir", type=Path, help="Directory containing zarr embeddings"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--skip-row-check",
        action="store_true",
        help="Skip verifying row counts match bed intervals (faster but less thorough)",
    )
    parser.add_argument(
        "--thorough",
        action="store_true",
        help="Check ALL chunks for NaN/Inf, not just first/middle/last (slowest)",
    )
    parser.add_argument(
        "--list-holes",
        action="store_true",
        help="List bed files that need reprocessing (one per line)",
    )
    parser.add_argument(
        "--list-missing",
        action="store_true",
        help="List missing zarr stems (one per line)",
    )

    args = parser.parse_args()

    if not args.bed_dir.exists():
        print(f"Error: bed directory not found: {args.bed_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.embed_dir.exists():
        print(f"Error: embed directory not found: {args.embed_dir}", file=sys.stderr)
        sys.exit(1)

    check_rows = not args.skip_row_check

    if args.list_holes:
        holes = find_holes(
            args.bed_dir,
            args.embed_dir,
            check_row_counts=check_rows,
            thorough=args.thorough,
        )
        for bed_path in holes:
            print(bed_path)
        sys.exit(0 if not holes else 1)

    report = check_integrity(
        args.bed_dir,
        args.embed_dir,
        check_row_counts=check_rows,
        thorough=args.thorough,
        verbose=args.verbose,
    )

    if args.list_missing:
        for stem in report.missing:
            print(stem)
        sys.exit(0 if not report.missing else 1)

    print(report.summary())

    if report.missing:
        print(f"\nMissing files ({len(report.missing)}):")
        for stem in report.missing[:20]:
            print(f"  - {stem}")
        if len(report.missing) > 20:
            print(f"  ... and {len(report.missing) - 20} more")

    if report.corrupted:
        print(f"\nCorrupted files ({len(report.corrupted)}):")
        for stem, error in report.corrupted[:10]:
            print(f"  - {stem}: {error}")
        if len(report.corrupted) > 10:
            print(f"  ... and {len(report.corrupted) - 10} more")

    if report.shape_mismatches:
        print(f"\nShape mismatches ({len(report.shape_mismatches)}):")
        for stem, shape, expected in report.shape_mismatches[:10]:
            print(f"  - {stem}: got {shape[0]} rows, expected {expected}")
        if len(report.shape_mismatches) > 10:
            print(f"  ... and {len(report.shape_mismatches) - 10} more")

    sys.exit(0 if report.is_healthy else 1)


if __name__ == "__main__":
    main()
