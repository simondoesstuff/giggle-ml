"""Tests for giggleml.data.similarity_matrix module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from giggleml.data.giggle import GiggleResult
from giggleml.data.similarity_matrix import SimilarityMatrix


class TestSimilarityMatrix:
    """Tests for SimilarityMatrix class."""

    def test_create_new_matrix(self, tmp_path):
        path = tmp_path / "matrix.mmap"
        matrix = SimilarityMatrix(path, n=10, mode="w+")

        assert matrix.path == path
        assert matrix.n == 10
        assert matrix.array.shape == (10, 10)
        assert matrix.array.dtype == np.float16
        assert np.all(matrix.array == 0)

    def test_open_existing_matrix(self, tmp_path):
        path = tmp_path / "matrix.mmap"

        # Create matrix
        matrix1 = SimilarityMatrix(path, n=5, mode="w+")
        matrix1[0, 1] = 1.5
        matrix1.flush()
        del matrix1

        # Reopen
        matrix2 = SimilarityMatrix(path, n=5, mode="r+")
        assert float(matrix2[0, 1]) == pytest.approx(1.5, rel=1e-2)

    def test_read_only_mode(self, tmp_path):
        path = tmp_path / "matrix.mmap"

        # Create matrix
        matrix1 = SimilarityMatrix(path, n=5, mode="w+")
        matrix1[2, 3] = 2.5
        matrix1.flush()
        del matrix1

        # Open read-only
        matrix2 = SimilarityMatrix(path, n=5, mode="r")
        assert float(matrix2[2, 3]) == pytest.approx(2.5, rel=1e-2)

    def test_getitem_setitem(self, tmp_path):
        path = tmp_path / "matrix.mmap"
        matrix = SimilarityMatrix(path, n=5, mode="w+")

        matrix[1, 2] = 3.14
        assert float(matrix[1, 2]) == pytest.approx(3.14, rel=1e-2)

    def test_array_property(self, tmp_path):
        path = tmp_path / "matrix.mmap"
        matrix = SimilarityMatrix(path, n=5, mode="w+")

        arr = matrix.array
        arr[0, 0] = 1.0
        assert float(matrix[0, 0]) == pytest.approx(1.0, rel=1e-2)

    def test_flush(self, tmp_path):
        path = tmp_path / "matrix.mmap"
        matrix = SimilarityMatrix(path, n=3, mode="w+")
        matrix[0, 0] = 5.0
        matrix.flush()

        # Verify file exists and has content
        assert path.exists()
        assert path.stat().st_size == 3 * 3 * 2  # 3x3 matrix * 2 bytes per float16


class TestBuildFromGiggle:
    """Tests for build_from_giggle utility."""

    @pytest.fixture
    def mock_index(self):
        """Create a mock GiggleIndex."""
        index = MagicMock()
        index.list_beds = {"a.bed.gz", "b.bed.gz", "c.bed.gz"}
        return index

    def test_builds_matrix_with_correct_size(self, tmp_path, mock_index):
        path = tmp_path / "matrix.mmap"

        # self_query returns results for each file
        mock_index.self_query.return_value = []

        matrix = SimilarityMatrix.build_from_giggle(mock_index, path)

        assert matrix.n == 3
        assert mock_index.self_query.call_count == 3

    def test_populates_combo_scores(self, tmp_path, mock_index):
        path = tmp_path / "matrix.mmap"

        def mock_self_query(bed_name):
            # Return similarity scores
            if bed_name == "a.bed.gz":
                return [
                    GiggleResult(
                        file="a.bed.gz",
                        file_size=100,
                        overlaps=10,
                        odds_ratio=1.0,
                        fishers_two_tail=1.0,
                        fishers_left_tail=1.0,
                        fishers_right_tail=1.0,
                        combo_score=100.0,
                    ),
                    GiggleResult(
                        file="b.bed.gz",
                        file_size=100,
                        overlaps=5,
                        odds_ratio=1.5,
                        fishers_two_tail=0.5,
                        fishers_left_tail=0.5,
                        fishers_right_tail=0.5,
                        combo_score=25.0,
                    ),
                ]
            return []

        mock_index.self_query.side_effect = mock_self_query

        matrix = SimilarityMatrix.build_from_giggle(mock_index, path, symmetric=False)

        # Beds are sorted: a, b, c -> indices 0, 1, 2
        assert float(matrix[0, 0]) == pytest.approx(100.0, rel=1e-2)
        assert float(matrix[0, 1]) == pytest.approx(25.0, rel=1e-2)

    def test_symmetric_mode_takes_max(self, tmp_path, mock_index):
        path = tmp_path / "matrix.mmap"

        def mock_self_query(bed_name):
            # a -> b has score 25, b -> a has score 30
            if bed_name == "a.bed.gz":
                return [
                    GiggleResult(
                        file="b.bed.gz",
                        file_size=100,
                        overlaps=5,
                        odds_ratio=1.5,
                        fishers_two_tail=0.5,
                        fishers_left_tail=0.5,
                        fishers_right_tail=0.5,
                        combo_score=25.0,
                    ),
                ]
            elif bed_name == "b.bed.gz":
                return [
                    GiggleResult(
                        file="a.bed.gz",
                        file_size=100,
                        overlaps=8,
                        odds_ratio=2.0,
                        fishers_two_tail=0.3,
                        fishers_left_tail=0.3,
                        fishers_right_tail=0.3,
                        combo_score=30.0,
                    ),
                ]
            return []

        mock_index.self_query.side_effect = mock_self_query

        matrix = SimilarityMatrix.build_from_giggle(mock_index, path, symmetric=True)

        # Both directions should have max value (30.0)
        # a=0, b=1
        assert float(matrix[0, 1]) == pytest.approx(30.0, rel=1e-2)
        assert float(matrix[1, 0]) == pytest.approx(30.0, rel=1e-2)

    def test_symmetric_mode_with_negative_scores(self, tmp_path, mock_index):
        """Verify negative scores are handled correctly (not zeroed out)."""
        path = tmp_path / "matrix.mmap"

        def mock_self_query(bed_name):
            # a -> b has score -10, b -> a has score -5
            if bed_name == "a.bed.gz":
                return [
                    GiggleResult(
                        file="b.bed.gz",
                        file_size=100,
                        overlaps=5,
                        odds_ratio=0.5,
                        fishers_two_tail=0.9,
                        fishers_left_tail=0.9,
                        fishers_right_tail=0.9,
                        combo_score=-10.0,
                    ),
                ]
            elif bed_name == "b.bed.gz":
                return [
                    GiggleResult(
                        file="a.bed.gz",
                        file_size=100,
                        overlaps=3,
                        odds_ratio=0.6,
                        fishers_two_tail=0.8,
                        fishers_left_tail=0.8,
                        fishers_right_tail=0.8,
                        combo_score=-5.0,
                    ),
                ]
            return []

        mock_index.self_query.side_effect = mock_self_query

        matrix = SimilarityMatrix.build_from_giggle(mock_index, path, symmetric=True)

        # Both directions should have max value (-5.0, not 0)
        # a=0, b=1
        assert float(matrix[0, 1]) == pytest.approx(-5.0, rel=1e-2)
        assert float(matrix[1, 0]) == pytest.approx(-5.0, rel=1e-2)

    def test_asymmetric_mode_keeps_direction(self, tmp_path, mock_index):
        path = tmp_path / "matrix.mmap"

        def mock_self_query(bed_name):
            if bed_name == "a.bed.gz":
                return [
                    GiggleResult(
                        file="b.bed.gz",
                        file_size=100,
                        overlaps=5,
                        odds_ratio=1.5,
                        fishers_two_tail=0.5,
                        fishers_left_tail=0.5,
                        fishers_right_tail=0.5,
                        combo_score=25.0,
                    ),
                ]
            elif bed_name == "b.bed.gz":
                return [
                    GiggleResult(
                        file="a.bed.gz",
                        file_size=100,
                        overlaps=8,
                        odds_ratio=2.0,
                        fishers_two_tail=0.3,
                        fishers_left_tail=0.3,
                        fishers_right_tail=0.3,
                        combo_score=30.0,
                    ),
                ]
            return []

        mock_index.self_query.side_effect = mock_self_query

        matrix = SimilarityMatrix.build_from_giggle(mock_index, path, symmetric=False)

        # a=0, b=1 - each direction keeps its own value
        assert float(matrix[0, 1]) == pytest.approx(25.0, rel=1e-2)
        assert float(matrix[1, 0]) == pytest.approx(30.0, rel=1e-2)

    def test_ignores_unknown_files_in_results(self, tmp_path, mock_index):
        path = tmp_path / "matrix.mmap"

        def mock_self_query(bed_name):
            return [
                GiggleResult(
                    file="unknown.bed.gz",
                    file_size=100,
                    overlaps=5,
                    odds_ratio=1.5,
                    fishers_two_tail=0.5,
                    fishers_left_tail=0.5,
                    fishers_right_tail=0.5,
                    combo_score=50.0,
                ),
            ]

        mock_index.self_query.side_effect = mock_self_query

        matrix = SimilarityMatrix.build_from_giggle(mock_index, path)

        # Matrix should be all zeros since unknown file is ignored
        assert np.all(matrix.array == 0)

    def test_sorts_beds_for_consistent_ordering(self, tmp_path):
        """Verify beds are sorted for consistent matrix indexing."""
        index = MagicMock()
        index.list_beds = {"z.bed.gz", "a.bed.gz", "m.bed.gz"}
        index.self_query.return_value = []

        path = tmp_path / "matrix.mmap"
        SimilarityMatrix.build_from_giggle(index, path)

        # Verify calls were made in sorted order
        calls = [call[0][0] for call in index.self_query.call_args_list]
        assert calls == ["a.bed.gz", "m.bed.gz", "z.bed.gz"]
