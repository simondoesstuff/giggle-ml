"""Reproducible train/test/val data splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DataSplit:
    """Container for train/test/val split indices.

    Attributes:
        train: Indices for training set.
        val: Indices for validation set.
        test: Indices for test set.
    """

    train: list[int]
    val: list[int]
    test: list[int]

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_val(self) -> int:
        return len(self.val)

    @property
    def n_test(self) -> int:
        return len(self.test)


def train_test_val_split(
    n: int,
    *,
    test_fraction: float = 0.1,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> DataSplit:
    """Split indices into train, test, and validation sets.

    The split is deterministic given the same n and seed, allowing
    separate scripts to reproduce the exact same split.

    Args:
        n: Total number of samples.
        test_fraction: Fraction of samples for test set.
        val_fraction: Fraction of samples for validation set.
        seed: Random seed for reproducibility.

    Returns:
        DataSplit containing train, val, and test indices.

    Raises:
        ValueError: If fractions are invalid or sum to >= 1.
    """
    if test_fraction < 0 or val_fraction < 0:
        raise ValueError("Fractions must be non-negative")
    if test_fraction + val_fraction >= 1.0:
        raise ValueError("test_fraction + val_fraction must be < 1.0")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    test_size = int(n * test_fraction)
    val_size = int(n * val_fraction)

    test_indices = sorted(indices[:test_size].tolist())
    val_indices = sorted(indices[test_size : test_size + val_size].tolist())
    train_indices = sorted(indices[test_size + val_size :].tolist())

    return DataSplit(train=train_indices, val=val_indices, test=test_indices)
