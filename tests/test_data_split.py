"""Tests for data_split utilities."""

import pytest

from giggleml.utils.data_split import DataSplit, train_test_val_split


class TestDataSplit:
    def test_properties(self) -> None:
        split = DataSplit(train=[0, 1, 2], val=[3, 4], test=[5])
        assert split.n_train == 3
        assert split.n_val == 2
        assert split.n_test == 1

    def test_frozen(self) -> None:
        split = DataSplit(train=[0], val=[1], test=[2])
        with pytest.raises(Exception):
            split.train = [3]  # type: ignore


class TestTrainTestValSplit:
    def test_splits_correct_sizes(self) -> None:
        split = train_test_val_split(100, test_fraction=0.1, val_fraction=0.1, seed=42)
        assert split.n_test == 10
        assert split.n_val == 10
        assert split.n_train == 80

    def test_no_overlap(self) -> None:
        split = train_test_val_split(100, test_fraction=0.2, val_fraction=0.1, seed=42)
        all_indices = set(split.train) | set(split.val) | set(split.test)
        assert len(all_indices) == 100
        assert len(split.train) + len(split.val) + len(split.test) == 100

    def test_reproducible(self) -> None:
        split1 = train_test_val_split(100, test_fraction=0.1, val_fraction=0.1, seed=42)
        split2 = train_test_val_split(100, test_fraction=0.1, val_fraction=0.1, seed=42)
        assert split1.train == split2.train
        assert split1.val == split2.val
        assert split1.test == split2.test

    def test_different_seeds_different_splits(self) -> None:
        split1 = train_test_val_split(100, test_fraction=0.1, val_fraction=0.1, seed=42)
        split2 = train_test_val_split(100, test_fraction=0.1, val_fraction=0.1, seed=123)
        assert split1.train != split2.train

    def test_indices_sorted(self) -> None:
        split = train_test_val_split(100, test_fraction=0.1, val_fraction=0.1, seed=42)
        assert split.train == sorted(split.train)
        assert split.val == sorted(split.val)
        assert split.test == sorted(split.test)

    def test_invalid_fractions_raise(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            train_test_val_split(100, test_fraction=-0.1, val_fraction=0.1)

        with pytest.raises(ValueError, match="< 1.0"):
            train_test_val_split(100, test_fraction=0.5, val_fraction=0.6)

    def test_zero_test_fraction(self) -> None:
        split = train_test_val_split(100, test_fraction=0.0, val_fraction=0.1, seed=42)
        assert split.n_test == 0
        assert split.n_val == 10
        assert split.n_train == 90

    def test_zero_val_fraction(self) -> None:
        split = train_test_val_split(100, test_fraction=0.1, val_fraction=0.0, seed=42)
        assert split.n_test == 10
        assert split.n_val == 0
        assert split.n_train == 90
