"""VRAM estimation utilities for dynamic batching."""

from dataclasses import dataclass


def next_power_of_2(x: int) -> int:
    """Return the smallest power of 2 >= x."""
    return 1 << (x - 1).bit_length()


def fft_length(k: int) -> int:
    """Effective FFT length for sequence length k.

    FFT convolutions pad to next power of 2 of (2k-1) to avoid circular artifacts.
    """
    return next_power_of_2(k << 1 - 1)


@dataclass
class VRAMCoeffs:
    """VRAM model coefficients for V(n, k) = a*n*k_fft + b*n + c.

    Attributes:
        a: Per-FFT-token cost (bytes per FFT token per sequence).
        b: Per-sequence overhead (bytes per sequence).
        c: Fixed model overhead (bytes).
    """

    a: float
    b: float
    c: float

    def estimate(self, n: int, k_max: int) -> float:
        """Estimate VRAM for a batch of n sequences with max length k_max."""
        return self.a * n * fft_length(k_max) + self.b * n + self.c
