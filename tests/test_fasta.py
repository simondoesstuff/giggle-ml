"""Tests for giggleml.data.fasta module."""

import os
from pathlib import Path

import jax.numpy as jnp
import pytest

from giggleml.data.fasta import fasta_keys, fasta_map, load_fasta


@pytest.fixture
def simple_fasta_file(tmp_path: Path) -> Path:
    """Create a simple FASTA file for testing."""
    fasta_path = tmp_path / "test.fa"
    fasta_path.write_text(">chr1\nACGTACGTACGT\n>chr2\nGGGGCCCCAAAA\n")
    return fasta_path


@pytest.fixture
def multiline_fasta_file(tmp_path: Path) -> Path:
    """Create a FASTA file with multi-line sequences."""
    fasta_path = tmp_path / "multiline.fa"
    fasta_path.write_text(">chr1\nACGT\nACGT\n>chr2\nGGGG\nCCCC\n")
    return fasta_path


@pytest.fixture
def loaded_fasta(simple_fasta_file: Path) -> dict[str, str]:
    """Load a simple FASTA file."""
    return load_fasta(simple_fasta_file)


class TestLoadFasta:
    """Tests for load_fasta function."""

    def test_loads_sequences(self, simple_fasta_file: Path):
        fasta = load_fasta(simple_fasta_file)
        assert "chr1" in fasta
        assert "chr2" in fasta

    def test_sequence_content(self, simple_fasta_file: Path):
        fasta = load_fasta(simple_fasta_file)
        assert fasta["chr1"] == "ACGTACGTACGT"
        assert fasta["chr2"] == "GGGGCCCCAAAA"

    def test_multiline_sequences(self, multiline_fasta_file: Path):
        fasta = load_fasta(multiline_fasta_file)
        assert fasta["chr1"] == "ACGTACGT"
        assert fasta["chr2"] == "GGGGCCCC"

    def test_returns_dict(self, simple_fasta_file: Path):
        fasta = load_fasta(simple_fasta_file)
        assert isinstance(fasta, dict)

    def test_caching(self, simple_fasta_file: Path):
        # Load twice, should be cached
        fasta1 = load_fasta(simple_fasta_file)
        fasta2 = load_fasta(simple_fasta_file)
        # Same object due to caching
        assert fasta1 is fasta2

    def test_creates_index_file(self, simple_fasta_file: Path):
        load_fasta(simple_fasta_file)
        index_path = simple_fasta_file.with_suffix(".fa.fxi")
        assert index_path.exists()


class TestFastaKeys:
    """Tests for fasta_keys function."""

    def test_returns_keys(self, loaded_fasta: dict[str, str]):
        keys = fasta_keys(loaded_fasta)
        assert "chr1" in keys
        assert "chr2" in keys

    def test_returns_list(self, loaded_fasta: dict[str, str]):
        keys = fasta_keys(loaded_fasta)
        assert isinstance(keys, list)

    def test_preserves_order(self, simple_fasta_file: Path):
        fasta = load_fasta(simple_fasta_file)
        keys = fasta_keys(fasta)
        assert keys == ["chr1", "chr2"]


class TestFastaMap:
    """Tests for fasta_map function."""

    def test_extracts_from_tuple_intervals(self, loaded_fasta: dict[str, str]):
        intervals = [("chr1", 0, 4), ("chr2", 0, 4)]
        seqs = list(fasta_map(loaded_fasta, intervals))
        assert seqs == ["ACGT", "GGGG"]

    def test_extracts_substring(self, loaded_fasta: dict[str, str]):
        intervals = [("chr1", 4, 8)]
        seqs = list(fasta_map(loaded_fasta, intervals))
        assert seqs == ["ACGT"]

    def test_extracts_from_array_intervals(self, loaded_fasta: dict[str, str]):
        # Using custom chromosomes matching our fasta
        custom_chroms = ("chr1", "chr2")
        intervals = jnp.array([[0, 0, 4], [1, 0, 4]], dtype=jnp.int32)
        seqs = list(fasta_map(loaded_fasta, intervals, chromosomes=custom_chroms))
        assert seqs == ["ACGT", "GGGG"]

    def test_empty_intervals(self, loaded_fasta: dict[str, str]):
        seqs = list(fasta_map(loaded_fasta, []))
        assert seqs == []

    def test_is_generator(self, loaded_fasta: dict[str, str]):
        intervals = [("chr1", 0, 4)]
        result = fasta_map(loaded_fasta, intervals)
        assert hasattr(result, "__next__")

    def test_full_sequence(self, loaded_fasta: dict[str, str]):
        intervals = [("chr1", 0, 12)]
        seqs = list(fasta_map(loaded_fasta, intervals))
        assert seqs == ["ACGTACGTACGT"]

    def test_with_generator_intervals(self, loaded_fasta: dict[str, str]):
        def gen_intervals():
            yield ("chr1", 0, 4)
            yield ("chr2", 4, 8)

        seqs = list(fasta_map(loaded_fasta, gen_intervals()))
        assert seqs == ["ACGT", "CCCC"]

    def test_array_intervals_with_default_chromosomes(self, tmp_path: Path):
        # Create a FASTA with standard chromosome names
        fasta_path = tmp_path / "standard.fa"
        fasta_path.write_text(">chr1\nACGTACGTACGT\n>chr2\nGGGGCCCCAAAA\n")
        fasta = load_fasta(fasta_path)

        # chr1 = index 0, chr2 = index 1 in DEFAULT_CHROMOSOMES
        intervals = jnp.array([[0, 0, 4], [1, 0, 4]], dtype=jnp.int32)
        seqs = list(fasta_map(fasta, intervals))
        assert seqs == ["ACGT", "GGGG"]
