"""Tests for giggleml.data.contrastive_memmap module."""

from pathlib import Path

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import zarr

from giggleml.data.contrastive_memmap import (
    CURRENT_VERSION,
    ContrastiveMemmapData,
    ContrastiveMemmapMetadata,
)
from giggleml.train.contrastive_data_loader import BedFileCache


class TestContrastiveMemmapMetadata:
    """Tests for ContrastiveMemmapMetadata dataclass."""

    def test_to_dict_roundtrip(self):
        metadata = ContrastiveMemmapMetadata(
            version=1,
            embedding_dim=128,
            total_intervals=1000,
            num_files=3,
            bed_names=["a", "b", "c"],
            offsets=[0, 300, 700],
            lengths=[300, 400, 300],
        )

        data = metadata.to_dict()
        restored = ContrastiveMemmapMetadata.from_dict(data)

        assert restored.version == metadata.version
        assert restored.embedding_dim == metadata.embedding_dim
        assert restored.total_intervals == metadata.total_intervals
        assert restored.num_files == metadata.num_files
        assert restored.bed_names == metadata.bed_names
        assert restored.offsets == metadata.offsets
        assert restored.lengths == metadata.lengths


class TestContrastiveMemmapData:
    """Tests for ContrastiveMemmapData class."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample zarr embeddings and BED files for testing."""
        embedding_dir = tmp_path / "embeddings"
        bed_dir = tmp_path / "beds"
        embedding_dir.mkdir()
        bed_dir.mkdir()

        bed_names = ["file_a", "file_b", "file_c"]
        embedding_dim = 32

        # Create test data with varying sizes
        sizes = [10, 25, 15]

        for name, size in zip(bed_names, sizes):
            # Create zarr array with unique values for testing
            zarr_path = embedding_dir / f"{name}.zarr"
            arr = zarr.open_array(zarr_path, mode="w", shape=(size, embedding_dim))
            # Fill with values based on name and position
            data = np.arange(size * embedding_dim).reshape(size, embedding_dim)
            data = data.astype(np.float32) + ord(name[5]) * 1000  # Unique per file
            arr[:] = data

            # Create BED file with unique intervals
            bed_path = bed_dir / f"{name}.bed"
            with open(bed_path, "w") as f:
                for i in range(size):
                    # Use chr index matching file to make unique
                    chrom_idx = ord(name[5]) - ord("a") + 1
                    start = i * 1000
                    end = start + 500
                    f.write(f"chr{chrom_idx}\t{start}\t{end}\n")

        return {
            "embedding_dir": embedding_dir,
            "bed_dir": bed_dir,
            "bed_names": bed_names,
            "sizes": sizes,
            "embedding_dim": embedding_dim,
        }

    def test_build_creates_files(self, tmp_path, sample_data):
        output_dir = tmp_path / "memmap_output"

        memmap = ContrastiveMemmapData.build_from_files(
            bed_names=sample_data["bed_names"],
            embedding_dir=sample_data["embedding_dir"],
            bed_dir=sample_data["bed_dir"],
            output_dir=output_dir,
        )

        # Verify files were created
        assert (output_dir / "data.mmap").exists()
        assert (output_dir / "metadata.json").exists()

        # Verify metadata
        assert memmap.metadata.version == CURRENT_VERSION
        assert memmap.metadata.embedding_dim == sample_data["embedding_dim"]
        assert memmap.metadata.total_intervals == sum(sample_data["sizes"])
        assert memmap.metadata.num_files == len(sample_data["bed_names"])

    def test_build_sorts_bed_names(self, tmp_path, sample_data):
        output_dir = tmp_path / "memmap_output"

        # Pass unsorted bed names
        memmap = ContrastiveMemmapData.build_from_files(
            bed_names=["file_c", "file_a", "file_b"],  # Unsorted
            embedding_dir=sample_data["embedding_dir"],
            bed_dir=sample_data["bed_dir"],
            output_dir=output_dir,
        )

        # Should be sorted in metadata
        assert memmap.bed_names == ["file_a", "file_b", "file_c"]

    def test_read_after_build(self, tmp_path, sample_data):
        output_dir = tmp_path / "memmap_output"

        ContrastiveMemmapData.build_from_files(
            bed_names=sample_data["bed_names"],
            embedding_dir=sample_data["embedding_dir"],
            bed_dir=sample_data["bed_dir"],
            output_dir=output_dir,
        )

        # Re-open in read mode
        memmap = ContrastiveMemmapData(output_dir, mode="r")

        assert memmap.metadata.total_intervals == sum(sample_data["sizes"])
        assert memmap.bed_names == sorted(sample_data["bed_names"])

    def test_get_embeddings_matches_original(self, tmp_path, sample_data):
        output_dir = tmp_path / "memmap_output"

        ContrastiveMemmapData.build_from_files(
            bed_names=sample_data["bed_names"],
            embedding_dir=sample_data["embedding_dir"],
            bed_dir=sample_data["bed_dir"],
            output_dir=output_dir,
        )

        memmap = ContrastiveMemmapData(output_dir, mode="r")

        # Check each file's embeddings match original zarr
        for idx, name in enumerate(sorted(sample_data["bed_names"])):
            zarr_path = sample_data["embedding_dir"] / f"{name}.zarr"
            zarr_array = zarr.open_array(zarr_path, mode="r")
            original = zarr_array[:].astype(ml_dtypes.bfloat16)

            memmap_emb = memmap.get_embeddings(idx)

            assert memmap_emb.shape == original.shape
            assert np.allclose(
                memmap_emb.astype(np.float32),
                original.astype(np.float32),
                rtol=1e-2,
            )

    def test_get_intervals_matches_original(self, tmp_path, sample_data):
        output_dir = tmp_path / "memmap_output"

        ContrastiveMemmapData.build_from_files(
            bed_names=sample_data["bed_names"],
            embedding_dir=sample_data["embedding_dir"],
            bed_dir=sample_data["bed_dir"],
            output_dir=output_dir,
        )

        memmap = ContrastiveMemmapData(output_dir, mode="r")

        # Check file_a (idx 0 after sorting, chr1)
        intervals = memmap.get_intervals(0)
        assert intervals.shape == (10, 3)
        assert intervals[0, 0] == 0  # chr1 -> idx 0
        assert intervals[0, 1] == 0  # start
        assert intervals[0, 2] == 500  # end

        # Check file_b (idx 1 after sorting, chr2)
        intervals = memmap.get_intervals(1)
        assert intervals.shape == (25, 3)
        assert intervals[0, 0] == 1  # chr2 -> idx 1

        # Check file_c (idx 2 after sorting, chr3)
        intervals = memmap.get_intervals(2)
        assert intervals.shape == (15, 3)
        assert intervals[0, 0] == 2  # chr3 -> idx 2

    def test_offsets_and_lengths(self, tmp_path, sample_data):
        output_dir = tmp_path / "memmap_output"

        memmap = ContrastiveMemmapData.build_from_files(
            bed_names=sample_data["bed_names"],
            embedding_dir=sample_data["embedding_dir"],
            bed_dir=sample_data["bed_dir"],
            output_dir=output_dir,
        )

        # Sorted order: file_a (10), file_b (25), file_c (15)
        assert memmap.metadata.lengths == [10, 25, 15]
        assert memmap.metadata.offsets == [0, 10, 35]


class TestBedFileCacheWithMemmap:
    """Tests for BedFileCache using memmap storage."""

    @pytest.fixture
    def memmap_setup(self, tmp_path):
        """Create memmap and corresponding zarr/BED files for comparison."""
        embedding_dir = tmp_path / "embeddings"
        bed_dir = tmp_path / "beds"
        memmap_dir = tmp_path / "memmap"
        embedding_dir.mkdir()
        bed_dir.mkdir()

        bed_names = ["bed0", "bed1", "bed2"]
        embedding_dim = 16
        sizes = [5, 8, 6]

        # Create test data
        for name, size in zip(bed_names, sizes):
            zarr_path = embedding_dir / f"{name}.zarr"
            arr = zarr.open_array(zarr_path, mode="w", shape=(size, embedding_dim))
            arr[:] = np.random.randn(size, embedding_dim).astype(np.float32)

            bed_path = bed_dir / f"{name}.bed"
            with open(bed_path, "w") as f:
                for i in range(size):
                    f.write(f"chr1\t{i * 100}\t{i * 100 + 50}\n")

        # Build memmap
        ContrastiveMemmapData.build_from_files(
            bed_names=bed_names,
            embedding_dir=embedding_dir,
            bed_dir=bed_dir,
            output_dir=memmap_dir,
        )

        return {
            "embedding_dir": embedding_dir,
            "bed_dir": bed_dir,
            "memmap_dir": memmap_dir,
            "bed_names": bed_names,
        }

    def test_cache_with_memmap_produces_same_data(self, memmap_setup):
        """BedFileCache with memmap should produce same data as without."""
        # Cache without memmap
        cache_no_memmap = BedFileCache(
            bed_names=memmap_setup["bed_names"],
            embedding_dir=memmap_setup["embedding_dir"],
            bed_dir=memmap_setup["bed_dir"],
        )

        # Cache with memmap
        cache_with_memmap = BedFileCache(
            bed_names=memmap_setup["bed_names"],
            embedding_dir=memmap_setup["embedding_dir"],
            bed_dir=memmap_setup["bed_dir"],
            memmap_dir=memmap_setup["memmap_dir"],
        )

        # Load same file with both caches
        for idx in range(3):
            data_no_memmap = cache_no_memmap._load_uncached(idx)
            data_with_memmap = cache_with_memmap._load_uncached(idx)

            assert data_no_memmap.node_idx == data_with_memmap.node_idx
            assert data_no_memmap.embeddings.shape == data_with_memmap.embeddings.shape
            assert data_no_memmap.intervals.shape == data_with_memmap.intervals.shape

            # Compare values (allow for bfloat16 precision differences)
            assert jnp.allclose(
                data_no_memmap.embeddings,
                data_with_memmap.embeddings,
                rtol=1e-2,
            )
            assert jnp.array_equal(
                data_no_memmap.intervals,
                data_with_memmap.intervals,
            )

    def test_cache_memmap_bed_names_mismatch_raises(self, memmap_setup):
        """BedFileCache should raise error if bed_names don't match memmap."""
        with pytest.raises(ValueError, match="bed_names mismatch"):
            BedFileCache(
                bed_names=["bed0", "bed1", "different_name"],  # Wrong name
                embedding_dir=memmap_setup["embedding_dir"],
                bed_dir=memmap_setup["bed_dir"],
                memmap_dir=memmap_setup["memmap_dir"],
            )

    def test_cache_memmap_validates_sorted_order(self, memmap_setup):
        """BedFileCache should validate sorted bed_names match memmap."""
        # This should work because bed_names are sorted internally
        cache = BedFileCache(
            bed_names=["bed2", "bed0", "bed1"],  # Unsorted input
            embedding_dir=memmap_setup["embedding_dir"],
            bed_dir=memmap_setup["bed_dir"],
            memmap_dir=memmap_setup["memmap_dir"],
        )

        # Should have sorted internally and match memmap
        assert cache.bed_names == ["bed0", "bed1", "bed2"]


class TestContrastiveMemmapErrors:
    """Tests for error handling in ContrastiveMemmapData."""

    def test_inconsistent_embedding_dim_raises(self, tmp_path):
        """Should raise error if zarr files have different embedding dimensions."""
        embedding_dir = tmp_path / "embeddings"
        bed_dir = tmp_path / "beds"
        output_dir = tmp_path / "memmap"
        embedding_dir.mkdir()
        bed_dir.mkdir()

        # Create zarr with different dimensions
        arr1 = zarr.open_array(embedding_dir / "file_a.zarr", mode="w", shape=(5, 32))
        arr1[:] = np.zeros((5, 32))

        arr2 = zarr.open_array(embedding_dir / "file_b.zarr", mode="w", shape=(5, 64))
        arr2[:] = np.zeros((5, 64))

        # Create BED files
        for name in ["file_a", "file_b"]:
            with open(bed_dir / f"{name}.bed", "w") as f:
                for i in range(5):
                    f.write(f"chr1\t{i * 100}\t{i * 100 + 50}\n")

        with pytest.raises(ValueError, match="Inconsistent embedding dim"):
            ContrastiveMemmapData.build_from_files(
                bed_names=["file_a", "file_b"],
                embedding_dir=embedding_dir,
                bed_dir=bed_dir,
                output_dir=output_dir,
            )

    def test_empty_bed_names_raises(self, tmp_path):
        """Should raise error if no files to process."""
        with pytest.raises(ValueError, match="No files to process"):
            ContrastiveMemmapData.build_from_files(
                bed_names=[],
                embedding_dir=tmp_path,
                bed_dir=tmp_path,
                output_dir=tmp_path / "output",
            )
