"""Tests for giggleml.utils.file_utils module."""

from pathlib import Path

from giggleml.utils.file_utils import file_ext


class TestFileExt:
    """Tests for file_ext function."""

    def test_simple_extension(self):
        assert file_ext("test.txt") == "txt"

    def test_double_extension(self):
        assert file_ext("test.tar.gz") == "gz"

    def test_no_extension(self):
        assert file_ext("README") == ""

    def test_path_with_directories(self):
        assert file_ext("/path/to/file.bed") == "bed"

    def test_path_object(self):
        assert file_ext(Path("/some/path/file.fasta")) == "fasta"

    def test_hidden_file_no_extension(self):
        # Hidden files like .gitignore have no suffix per Path semantics
        assert file_ext(".gitignore") == ""

    def test_hidden_file_with_actual_extension(self):
        assert file_ext(".config.json") == "json"

    def test_gz_compressed(self):
        assert file_ext("intervals.bed.gz") == "gz"

    def test_fasta_extension(self):
        assert file_ext("genome.fa") == "fa"

    def test_uppercase_extension(self):
        assert file_ext("data.BED") == "BED"
