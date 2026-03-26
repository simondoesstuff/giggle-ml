from __future__ import annotations

import numpy as np


def softmax_with_temperature(
    data: np.ndarray, temperature: float, axis: int = 0
) -> np.ndarray:
    """Apply softmax with temperature along the specified axis, preserving NaNs."""
    scaled = data / temperature
    # Subtract max for numerical stability (per-column when axis=0)
    scaled = scaled - np.nanmax(scaled, axis=axis, keepdims=True)
    exp_vals = np.exp(scaled)
    # Set NaN positions to 0 for sum, then restore NaN
    exp_vals_masked = np.where(np.isnan(data), 0, exp_vals)
    softmax_vals = exp_vals_masked / np.nansum(
        exp_vals_masked, axis=axis, keepdims=True
    )
    # Restore NaNs
    softmax_vals = np.where(np.isnan(data), np.nan, softmax_vals)
    return softmax_vals


def sigmoid_with_temperature(
    data: np.ndarray, temperature: float, midpoint: float = 0.0
) -> np.ndarray:
    """Apply sigmoid with temperature, preserving NaNs.

    Args:
        data: Input array.
        temperature: Controls steepness. Lower = sharper transition.
        midpoint: Value where sigmoid outputs 0.5.

    Returns:
        Sigmoid-transformed array with values in [0, 1].
    """
    scaled = (data - midpoint) / temperature
    result = 1.0 / (1.0 + np.exp(-scaled))
    return np.where(np.isnan(data), np.nan, result)
