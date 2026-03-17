"""Tests for the check_embed_integrity script."""

import gzip
from pathlib import Path

import numpy as np
import pytest
import zarr

from scripts.check_embed_integrity import (
    IntegrityReport,
    check_integrity,
    check_zarr_integrity,
    count_bed_intervals,
    find_holes,
    is_zarr_array,
    iter_bed_files,
    iter_zarr_arrays,
)


class TestIterBedFiles:
    """Tests for iter_bed_files function."""

    def test_finds_bed_files(self, tmp_path: Path):
        (tmp_path / "a.bed").touch()
        (tmp_path / "b.bed").touch()
        beds = list(iter_bed_files(tmp_path))
        assert len(beds) == 2

    def test_finds_gzipped_bed_files(self, tmp_path: Path):
        (tmp_path / "a.bed.gz").touch()
        (tmp_path / "b.bed.gz").touch()
        beds = list(iter_bed_files(tmp_path))
        assert len(beds) == 2

    def test_finds_mixed_bed_files(self, tmp_path: Path):
        (tmp_path / "a.bed").touch()
        (tmp_path / "b.bed.gz").touch()
        beds = list(iter_bed_files(tmp_path))
        assert len(beds) == 2

    def test_ignores_non_bed_files(self, tmp_path: Path):
        (tmp_path / "a.bed").touch()
        (tmp_path / "b.txt").touch()
        (tmp_path / "c.zarr").mkdir()
        beds = list(iter_bed_files(tmp_path))
        assert len(beds) == 1

    def test_empty_directory(self, tmp_path: Path):
        beds = list(iter_bed_files(tmp_path))
        assert len(beds) == 0


class TestIsZarrArray:
    """Tests for is_zarr_array function."""

    def test_zarr_v3_array(self, tmp_path: Path):
        arr_path = tmp_path / "test_array"
        zarr.create_array(arr_path, shape=(10, 8), dtype="float32")
        assert is_zarr_array(arr_path)

    def test_zarr_v3_with_extension(self, tmp_path: Path):
        arr_path = tmp_path / "test_array.zarr"
        zarr.create_array(arr_path, shape=(10, 8), dtype="float32")
        assert is_zarr_array(arr_path)

    def test_empty_directory_not_zarr(self, tmp_path: Path):
        arr_path = tmp_path / "not_zarr"
        arr_path.mkdir()
        assert not is_zarr_array(arr_path)

    def test_nonexistent_path(self, tmp_path: Path):
        assert not is_zarr_array(tmp_path / "nonexistent")

    def test_regular_file_not_zarr(self, tmp_path: Path):
        file_path = tmp_path / "regular.txt"
        file_path.touch()
        assert not is_zarr_array(file_path)


class TestIterZarrArrays:
    """Tests for iter_zarr_arrays function."""

    def test_finds_zarr_without_extension(self, tmp_path: Path):
        zarr.create_array(tmp_path / "array1", shape=(10, 8), dtype="float32")
        zarr.create_array(tmp_path / "array2", shape=(5, 8), dtype="float32")
        arrays = list(iter_zarr_arrays(tmp_path))
        assert len(arrays) == 2

    def test_finds_zarr_with_extension(self, tmp_path: Path):
        zarr.create_array(tmp_path / "array1.zarr", shape=(10, 8), dtype="float32")
        zarr.create_array(tmp_path / "array2.zarr", shape=(5, 8), dtype="float32")
        arrays = list(iter_zarr_arrays(tmp_path))
        assert len(arrays) == 2

    def test_finds_mixed_zarr(self, tmp_path: Path):
        zarr.create_array(tmp_path / "array1", shape=(10, 8), dtype="float32")
        zarr.create_array(tmp_path / "array2.zarr", shape=(5, 8), dtype="float32")
        arrays = list(iter_zarr_arrays(tmp_path))
        assert len(arrays) == 2

    def test_ignores_non_zarr_directories(self, tmp_path: Path):
        zarr.create_array(tmp_path / "array1", shape=(10, 8), dtype="float32")
        (tmp_path / "not_zarr").mkdir()
        arrays = list(iter_zarr_arrays(tmp_path))
        assert len(arrays) == 1

    def test_empty_directory(self, tmp_path: Path):
        arrays = list(iter_zarr_arrays(tmp_path))
        assert len(arrays) == 0


class TestCountBedIntervals:
    """Tests for count_bed_intervals function."""

    def test_counts_plain_bed(self, tmp_path: Path):
        bed_path = tmp_path / "test.bed"
        bed_path.write_text("chr1\t0\t100\nchr1\t100\t200\nchr2\t0\t50\n")
        assert count_bed_intervals(bed_path) == 3

    def test_counts_gzipped_bed(self, tmp_path: Path):
        bed_path = tmp_path / "test.bed.gz"
        with gzip.open(bed_path, "wt") as f:
            f.write("chr1\t0\t100\nchr1\t100\t200\n")
        assert count_bed_intervals(bed_path) == 2

    def test_ignores_comments(self, tmp_path: Path):
        bed_path = tmp_path / "test.bed"
        bed_path.write_text("# comment\nchr1\t0\t100\n#another\nchr1\t100\t200\n")
        assert count_bed_intervals(bed_path) == 2

    def test_ignores_track_lines(self, tmp_path: Path):
        bed_path = tmp_path / "test.bed"
        bed_path.write_text("track name=test\nchr1\t0\t100\n")
        assert count_bed_intervals(bed_path) == 1

    def test_ignores_empty_lines(self, tmp_path: Path):
        bed_path = tmp_path / "test.bed"
        bed_path.write_text("chr1\t0\t100\n\n\nchr1\t100\t200\n")
        assert count_bed_intervals(bed_path) == 2

    def test_empty_file(self, tmp_path: Path):
        bed_path = tmp_path / "test.bed"
        bed_path.write_text("")
        assert count_bed_intervals(bed_path) == 0


class TestCheckZarrIntegrity:
    """Tests for check_zarr_integrity function."""

    def test_valid_array(self, tmp_path: Path):
        arr_path = tmp_path / "valid"
        arr = zarr.create_array(arr_path, shape=(10, 8), dtype="float32")
        arr[:] = np.random.randn(10, 8).astype(np.float32)
        is_valid, error = check_zarr_integrity(arr_path)
        assert is_valid
        assert error is None

    def test_nonexistent_path(self, tmp_path: Path):
        is_valid, error = check_zarr_integrity(tmp_path / "nonexistent")
        assert not is_valid
        assert error is not None
        assert "does not exist" in error

    def test_row_mismatch(self, tmp_path: Path):
        arr_path = tmp_path / "array"
        arr = zarr.create_array(arr_path, shape=(10, 8), dtype="float32")
        arr[:] = np.random.randn(10, 8).astype(np.float32)
        is_valid, error = check_zarr_integrity(arr_path, expected_rows=20)
        assert not is_valid
        assert error is not None
        assert "Row mismatch" in error

    def test_correct_row_count(self, tmp_path: Path):
        arr_path = tmp_path / "array"
        arr = zarr.create_array(arr_path, shape=(10, 8), dtype="float32")
        arr[:] = np.random.randn(10, 8).astype(np.float32)
        is_valid, error = check_zarr_integrity(arr_path, expected_rows=10)
        assert is_valid
        assert error is None

    def test_nan_values(self, tmp_path: Path):
        arr_path = tmp_path / "array"
        arr = zarr.create_array(arr_path, shape=(10, 8), dtype="float32")
        data = np.random.randn(10, 8).astype(np.float32)
        data[5, 3] = np.nan
        arr[:] = data
        is_valid, error = check_zarr_integrity(arr_path)
        assert not is_valid
        assert error is not None
        assert "NaN" in error

    def test_inf_values(self, tmp_path: Path):
        arr_path = tmp_path / "array"
        arr = zarr.create_array(arr_path, shape=(10, 8), dtype="float32")
        data = np.random.randn(10, 8).astype(np.float32)
        data[5, 3] = np.inf
        arr[:] = data
        is_valid, error = check_zarr_integrity(arr_path)
        assert not is_valid
        assert error is not None
        assert "Inf" in error

    def test_empty_array(self, tmp_path: Path):
        arr_path = tmp_path / "array"
        zarr.create_array(arr_path, shape=(0, 8), dtype="float32")
        is_valid, error = check_zarr_integrity(arr_path)
        assert not is_valid
        assert error is not None
        assert "empty" in error.lower()

    def test_wrong_dimensions(self, tmp_path: Path):
        arr_path = tmp_path / "array"
        arr = zarr.create_array(arr_path, shape=(10,), dtype="float32")
        arr[:] = np.random.randn(10).astype(np.float32)
        is_valid, error = check_zarr_integrity(arr_path)
        assert not is_valid
        assert error is not None
        assert "2D" in error

    def test_nan_in_last_chunk(self, tmp_path: Path):
        arr_path = tmp_path / "array"
        # Create array with multiple chunks
        arr = zarr.create_array(
            arr_path, shape=(100, 8), chunks=(10, 8), dtype="float32"
        )
        data = np.random.randn(100, 8).astype(np.float32)
        data[95, 3] = np.nan  # NaN in last chunk
        arr[:] = data
        is_valid, error = check_zarr_integrity(arr_path)
        assert not is_valid
        assert error is not None
        assert "NaN" in error

    def test_nan_in_middle_chunk_detected_by_sampling(self, tmp_path: Path):
        arr_path = tmp_path / "array"
        # Create array with many chunks - middle chunk sampling should catch this
        arr = zarr.create_array(
            arr_path, shape=(100, 8), chunks=(10, 8), dtype="float32"
        )
        data = np.random.randn(100, 8).astype(np.float32)
        data[55, 3] = np.nan  # NaN in middle chunk (chunk 5)
        arr[:] = data
        # Default sampling checks first, middle, last - should catch middle
        is_valid, error = check_zarr_integrity(arr_path)
        assert not is_valid
        assert error is not None
        assert "NaN" in error

    def test_nan_in_unchecked_chunk_missed_without_thorough(self, tmp_path: Path):
        arr_path = tmp_path / "array"
        # Create array with many chunks
        arr = zarr.create_array(
            arr_path, shape=(100, 8), chunks=(10, 8), dtype="float32"
        )
        data = np.random.randn(100, 8).astype(np.float32)
        # Put NaN in chunk 2 (rows 20-29) - not first (0), middle (50), or last (90)
        data[25, 3] = np.nan
        arr[:] = data
        # Default sampling misses this
        is_valid, _ = check_zarr_integrity(arr_path, thorough=False)
        assert is_valid  # Missed!

        # Thorough mode catches it
        is_valid, error = check_zarr_integrity(arr_path, thorough=True)
        assert not is_valid
        assert error is not None
        assert "NaN" in error

    def test_thorough_checks_all_chunks(self, tmp_path: Path):
        arr_path = tmp_path / "array"
        arr = zarr.create_array(
            arr_path, shape=(50, 8), chunks=(10, 8), dtype="float32"
        )
        data = np.random.randn(50, 8).astype(np.float32)
        data[15, 0] = np.inf  # Inf in chunk 1 (not first or last)
        arr[:] = data

        is_valid, error = check_zarr_integrity(arr_path, thorough=True)
        assert not is_valid
        assert error is not None
        assert "Inf" in error
        assert "chunk at row 10" in error


class TestCheckIntegrity:
    """Tests for check_integrity function."""

    @pytest.fixture
    def setup_dirs(self, tmp_path: Path):
        """Create bed and embed directories."""
        bed_dir = tmp_path / "beds"
        embed_dir = tmp_path / "embeds"
        bed_dir.mkdir()
        embed_dir.mkdir()
        return bed_dir, embed_dir

    def test_all_healthy(self, setup_dirs: tuple[Path, Path]):
        bed_dir, embed_dir = setup_dirs
        # Create matching bed and zarr files
        (bed_dir / "sample1.bed").write_text("chr1\t0\t100\nchr1\t100\t200\n")
        (bed_dir / "sample2.bed").write_text("chr1\t0\t50\n")

        arr1 = zarr.create_array(embed_dir / "sample1", shape=(2, 8), dtype="float32")
        arr1[:] = np.random.randn(2, 8).astype(np.float32)
        arr2 = zarr.create_array(embed_dir / "sample2", shape=(1, 8), dtype="float32")
        arr2[:] = np.random.randn(1, 8).astype(np.float32)

        report = check_integrity(bed_dir, embed_dir, check_row_counts=True)
        assert report.is_healthy
        assert report.total_expected == 2
        assert report.total_found == 2

    def test_missing_file(self, setup_dirs: tuple[Path, Path]):
        bed_dir, embed_dir = setup_dirs
        (bed_dir / "sample1.bed").write_text("chr1\t0\t100\n")
        (bed_dir / "sample2.bed").write_text("chr1\t0\t100\n")

        arr = zarr.create_array(embed_dir / "sample1", shape=(1, 8), dtype="float32")
        arr[:] = np.random.randn(1, 8).astype(np.float32)
        # sample2 is missing

        report = check_integrity(bed_dir, embed_dir, check_row_counts=False)
        assert not report.is_healthy
        assert len(report.missing) == 1
        assert "sample2" in report.missing

    def test_gzipped_bed_matching(self, setup_dirs: tuple[Path, Path]):
        bed_dir, embed_dir = setup_dirs
        # Create gzipped bed file
        with gzip.open(bed_dir / "sample.bed.gz", "wt") as f:
            f.write("chr1\t0\t100\nchr1\t100\t200\n")

        arr = zarr.create_array(embed_dir / "sample", shape=(2, 8), dtype="float32")
        arr[:] = np.random.randn(2, 8).astype(np.float32)

        report = check_integrity(bed_dir, embed_dir, check_row_counts=True)
        assert report.is_healthy
        assert report.total_expected == 1
        assert report.total_found == 1

    def test_zarr_with_extension(self, setup_dirs: tuple[Path, Path]):
        bed_dir, embed_dir = setup_dirs
        (bed_dir / "sample.bed").write_text("chr1\t0\t100\n")

        arr = zarr.create_array(
            embed_dir / "sample.zarr", shape=(1, 8), dtype="float32"
        )
        arr[:] = np.random.randn(1, 8).astype(np.float32)

        report = check_integrity(bed_dir, embed_dir, check_row_counts=False)
        assert report.is_healthy

    def test_shape_mismatch(self, setup_dirs: tuple[Path, Path]):
        bed_dir, embed_dir = setup_dirs
        (bed_dir / "sample.bed").write_text("chr1\t0\t100\nchr1\t100\t200\nchr1\t200\t300\n")

        # Array has wrong number of rows
        arr = zarr.create_array(embed_dir / "sample", shape=(2, 8), dtype="float32")
        arr[:] = np.random.randn(2, 8).astype(np.float32)

        report = check_integrity(bed_dir, embed_dir, check_row_counts=True)
        assert not report.is_healthy
        assert len(report.shape_mismatches) == 1
        stem, shape, expected = report.shape_mismatches[0]
        assert stem == "sample"
        assert shape[0] == 2
        assert expected == 3

    def test_skip_row_counts_when_disabled(self, setup_dirs: tuple[Path, Path]):
        bed_dir, embed_dir = setup_dirs
        (bed_dir / "sample.bed").write_text("chr1\t0\t100\nchr1\t100\t200\nchr1\t200\t300\n")

        arr = zarr.create_array(embed_dir / "sample", shape=(2, 8), dtype="float32")
        arr[:] = np.random.randn(2, 8).astype(np.float32)

        # With check_row_counts=False, mismatch should not be detected
        report = check_integrity(bed_dir, embed_dir, check_row_counts=False)
        assert report.is_healthy
        assert len(report.shape_mismatches) == 0

    def test_detects_nan(self, setup_dirs: tuple[Path, Path]):
        bed_dir, embed_dir = setup_dirs
        (bed_dir / "sample.bed").write_text("chr1\t0\t100\n")

        arr = zarr.create_array(embed_dir / "sample", shape=(1, 8), dtype="float32")
        data = np.zeros((1, 8), dtype=np.float32)
        data[0, 0] = np.nan
        arr[:] = data

        report = check_integrity(bed_dir, embed_dir, check_row_counts=False)
        assert not report.is_healthy
        assert "sample" in report.has_nan

    def test_detects_empty(self, setup_dirs: tuple[Path, Path]):
        bed_dir, embed_dir = setup_dirs
        (bed_dir / "sample.bed").write_text("chr1\t0\t100\n")

        zarr.create_array(embed_dir / "sample", shape=(0, 8), dtype="float32")

        report = check_integrity(bed_dir, embed_dir, check_row_counts=False)
        assert not report.is_healthy
        assert "sample" in report.empty


class TestFindHoles:
    """Tests for find_holes function."""

    def test_finds_missing_beds(self, tmp_path: Path):
        bed_dir = tmp_path / "beds"
        embed_dir = tmp_path / "embeds"
        bed_dir.mkdir()
        embed_dir.mkdir()

        (bed_dir / "sample1.bed").write_text("chr1\t0\t100\n")
        (bed_dir / "sample2.bed").write_text("chr1\t0\t100\n")

        arr = zarr.create_array(embed_dir / "sample1", shape=(1, 8), dtype="float32")
        arr[:] = np.random.randn(1, 8).astype(np.float32)

        holes = find_holes(bed_dir, embed_dir)
        assert len(holes) == 1
        assert holes[0].stem == "sample2"

    def test_no_holes_when_complete(self, tmp_path: Path):
        bed_dir = tmp_path / "beds"
        embed_dir = tmp_path / "embeds"
        bed_dir.mkdir()
        embed_dir.mkdir()

        (bed_dir / "sample.bed").write_text("chr1\t0\t100\n")
        arr = zarr.create_array(embed_dir / "sample", shape=(1, 8), dtype="float32")
        arr[:] = np.random.randn(1, 8).astype(np.float32)

        holes = find_holes(bed_dir, embed_dir)
        assert len(holes) == 0

    def test_finds_row_count_mismatches_by_default(self, tmp_path: Path):
        bed_dir = tmp_path / "beds"
        embed_dir = tmp_path / "embeds"
        bed_dir.mkdir()
        embed_dir.mkdir()

        # Bed has 3 intervals but zarr has 2 rows
        (bed_dir / "sample.bed").write_text("chr1\t0\t100\nchr1\t100\t200\nchr1\t200\t300\n")
        arr = zarr.create_array(embed_dir / "sample", shape=(2, 8), dtype="float32")
        arr[:] = np.random.randn(2, 8).astype(np.float32)

        holes = find_holes(bed_dir, embed_dir)
        assert len(holes) == 1
        assert holes[0].stem == "sample"

    def test_skips_row_check_when_disabled(self, tmp_path: Path):
        bed_dir = tmp_path / "beds"
        embed_dir = tmp_path / "embeds"
        bed_dir.mkdir()
        embed_dir.mkdir()

        (bed_dir / "sample.bed").write_text("chr1\t0\t100\nchr1\t100\t200\nchr1\t200\t300\n")
        arr = zarr.create_array(embed_dir / "sample", shape=(2, 8), dtype="float32")
        arr[:] = np.random.randn(2, 8).astype(np.float32)

        holes = find_holes(bed_dir, embed_dir, check_row_counts=False)
        assert len(holes) == 0  # Mismatch not detected


class TestIntegrityReport:
    """Tests for IntegrityReport dataclass."""

    def test_is_healthy_when_empty(self):
        report = IntegrityReport(
            total_expected=10,
            total_found=10,
            missing=[],
            corrupted=[],
            shape_mismatches=[],
            has_nan=[],
            has_inf=[],
            empty=[],
        )
        assert report.is_healthy
        assert report.n_problems == 0

    def test_not_healthy_with_missing(self):
        report = IntegrityReport(
            total_expected=10,
            total_found=9,
            missing=["sample1"],
            corrupted=[],
            shape_mismatches=[],
            has_nan=[],
            has_inf=[],
            empty=[],
        )
        assert not report.is_healthy
        assert report.n_problems == 1

    def test_n_problems_counts_all(self):
        report = IntegrityReport(
            total_expected=10,
            total_found=8,
            missing=["a", "b"],
            corrupted=[("c", "error")],
            shape_mismatches=[("d", (5, 8), 10)],
            has_nan=["e"],
            has_inf=["f"],
            empty=["g"],
        )
        assert report.n_problems == 7

    def test_summary_contains_key_info(self):
        report = IntegrityReport(
            total_expected=100,
            total_found=95,
            missing=["a", "b", "c"],
            corrupted=[],
            shape_mismatches=[],
            has_nan=[],
            has_inf=[],
            empty=[],
        )
        summary = report.summary()
        assert "Expected files:     100" in summary
        assert "Found files:        95" in summary
        assert "Missing files:      3" in summary
        assert "ISSUES FOUND" in summary

    def test_summary_shows_healthy(self):
        report = IntegrityReport(
            total_expected=10,
            total_found=10,
            missing=[],
            corrupted=[],
            shape_mismatches=[],
            has_nan=[],
            has_inf=[],
            empty=[],
        )
        summary = report.summary()
        assert "HEALTHY" in summary
