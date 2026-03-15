"""Tests for giggleml.data.intervals module."""

import gzip
import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest

from giggleml.data.intervals import (
    DEFAULT_CHROMOSOMES,
    crop_intervals,
    interval_to_array,
    load_bed,
    load_bed_array,
)


@pytest.fixture
def simple_bed_file(tmp_path: Path) -> Path:
    """Create a simple BED file for testing."""
    bed_path = tmp_path / "test.bed"
    bed_path.write_text("chr1\t100\t200\nchr2\t300\t400\nchr3\t500\t600\n")
    return bed_path


@pytest.fixture
def bed_with_comments(tmp_path: Path) -> Path:
    """Create a BED file with comments and blank lines."""
    bed_path = tmp_path / "comments.bed"
    bed_path.write_text(
        "# This is a comment\n"
        "chr1\t100\t200\n"
        "\n"
        "# Another comment\n"
        "chr2\t300\t400\n"
    )
    return bed_path


@pytest.fixture
def gzipped_bed_file(tmp_path: Path) -> Path:
    """Create a gzipped BED file for testing."""
    bed_path = tmp_path / "test.bed.gz"
    with gzip.open(bed_path, "wt") as f:
        f.write("chr1\t100\t200\nchr2\t300\t400\n")
    return bed_path


class TestDefaultChromosomes:
    """Tests for DEFAULT_CHROMOSOMES constant."""

    def test_contains_autosomes(self):
        for i in range(1, 23):
            assert f"chr{i}" in DEFAULT_CHROMOSOMES

    def test_contains_sex_chromosomes(self):
        assert "chrX" in DEFAULT_CHROMOSOMES
        assert "chrY" in DEFAULT_CHROMOSOMES

    def test_contains_mitochondrial(self):
        assert "chrM" in DEFAULT_CHROMOSOMES

    def test_total_count(self):
        assert len(DEFAULT_CHROMOSOMES) == 25


class TestLoadBed:
    """Tests for load_bed function."""

    def test_loads_simple_bed(self, simple_bed_file: Path):
        intervals = list(load_bed(simple_bed_file))
        assert intervals == [
            ("chr1", 100, 200),
            ("chr2", 300, 400),
            ("chr3", 500, 600),
        ]

    def test_skips_comments_and_blank_lines(self, bed_with_comments: Path):
        intervals = list(load_bed(bed_with_comments))
        assert intervals == [
            ("chr1", 100, 200),
            ("chr2", 300, 400),
        ]

    def test_loads_gzipped_bed(self, gzipped_bed_file: Path):
        intervals = list(load_bed(gzipped_bed_file))
        assert intervals == [
            ("chr1", 100, 200),
            ("chr2", 300, 400),
        ]

    def test_explicit_gzip_false(self, simple_bed_file: Path):
        intervals = list(load_bed(simple_bed_file, gzip=False))
        assert len(intervals) == 3

    def test_explicit_gzip_true(self, gzipped_bed_file: Path):
        intervals = list(load_bed(gzipped_bed_file, gzip=True))
        assert len(intervals) == 2

    def test_is_generator(self, simple_bed_file: Path):
        result = load_bed(simple_bed_file)
        assert hasattr(result, "__next__")

    def test_empty_bed_file(self, tmp_path: Path):
        bed_path = tmp_path / "empty.bed"
        bed_path.write_text("")
        intervals = list(load_bed(bed_path))
        assert intervals == []


class TestIntervalToArray:
    """Tests for interval_to_array function."""

    def test_simple_conversion(self):
        interval = ("chr1", 100, 200)
        arr = interval_to_array(interval)
        expected = jnp.array([0, 100, 200], dtype=jnp.int32)
        assert jnp.array_equal(arr, expected)

    def test_different_chromosome(self):
        interval = ("chr5", 500, 600)
        arr = interval_to_array(interval)
        expected = jnp.array([4, 500, 600], dtype=jnp.int32)
        assert jnp.array_equal(arr, expected)

    def test_chrx_index(self):
        interval = ("chrX", 1000, 2000)
        arr = interval_to_array(interval)
        # chrX is after chr22, so index 22
        assert arr[0] == 22

    def test_custom_chromosomes(self):
        custom_chroms = ("A", "B", "C")
        interval = ("B", 10, 20)
        arr = interval_to_array(interval, chromosomes=custom_chroms)
        expected = jnp.array([1, 10, 20], dtype=jnp.int32)
        assert jnp.array_equal(arr, expected)

    def test_output_dtype(self):
        interval = ("chr1", 100, 200)
        arr = interval_to_array(interval)
        assert arr.dtype == jnp.int32


class TestLoadBedArray:
    """Tests for load_bed_array function."""

    def test_loads_as_array(self, simple_bed_file: Path):
        arr = load_bed_array(simple_bed_file)
        expected = jnp.array(
            [[0, 100, 200], [1, 300, 400], [2, 500, 600]], dtype=jnp.int32
        )
        assert jnp.array_equal(arr, expected)

    def test_shape(self, simple_bed_file: Path):
        arr = load_bed_array(simple_bed_file)
        assert arr.shape == (3, 3)

    def test_dtype(self, simple_bed_file: Path):
        arr = load_bed_array(simple_bed_file)
        assert arr.dtype == jnp.int32

    def test_gzipped(self, gzipped_bed_file: Path):
        arr = load_bed_array(gzipped_bed_file)
        assert arr.shape == (2, 3)

    def test_custom_chromosomes(self, tmp_path: Path):
        bed_path = tmp_path / "custom.bed"
        bed_path.write_text("A\t10\t20\nB\t30\t40\n")
        custom_chroms = ("A", "B", "C")
        arr = load_bed_array(bed_path, chromosomes=custom_chroms)
        expected = jnp.array([[0, 10, 20], [1, 30, 40]], dtype=jnp.int32)
        assert jnp.array_equal(arr, expected)


class TestCropIntervals:
    """Tests for crop_intervals function."""

    def test_crop_without_centers(self):
        intervals = [("chr1", 100, 500), ("chr2", 200, 600)]
        cropped = list(crop_intervals(intervals, size=100))
        assert cropped == [("chr1", 100, 200), ("chr2", 200, 300)]

    def test_crop_respects_end_boundary(self):
        intervals = [("chr1", 100, 150)]  # Only 50bp available
        cropped = list(crop_intervals(intervals, size=100))
        # Should be capped at original end
        assert cropped == [("chr1", 100, 150)]

    def test_crop_with_centers(self):
        intervals = [("chr1", 0, 1000), ("chr2", 0, 1000)]
        centers = [500, 600]
        cropped = list(crop_intervals(intervals, size=100, centers=centers))
        # size=100, half=50, so [center-50, center-50+100]
        assert cropped == [("chr1", 450, 550), ("chr2", 550, 650)]

    def test_crop_with_centers_respects_start(self):
        intervals = [("chr1", 100, 1000)]
        centers = [120]  # center - half = 70, but start is 100
        cropped = list(crop_intervals(intervals, size=100, centers=centers))
        # new_start = max(120-50, 100) = 100
        assert cropped[0][1] == 100

    def test_crop_with_centers_respects_end(self):
        intervals = [("chr1", 0, 200)]
        centers = [180]  # Would want [130, 230] but end is 200
        cropped = list(crop_intervals(intervals, size=100, centers=centers))
        # new_start = 130, new_end = min(130+100, 200) = 200
        assert cropped == [("chr1", 130, 200)]

    def test_is_generator(self):
        intervals = [("chr1", 0, 100)]
        result = crop_intervals(intervals, size=50)
        assert hasattr(result, "__next__")

    def test_empty_input(self):
        cropped = list(crop_intervals([], size=100))
        assert cropped == []

    def test_preserves_chromosome(self):
        intervals = [("chrX", 0, 1000), ("chrY", 0, 1000)]
        cropped = list(crop_intervals(intervals, size=100))
        assert cropped[0][0] == "chrX"
        assert cropped[1][0] == "chrY"
