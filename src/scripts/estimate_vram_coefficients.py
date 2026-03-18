"""Estimate VRAM coefficients for HyenaDNA models.

Models VRAM usage as V(n, k) = a*n*k_fft + b*n + c where:
- n: batch size (number of sequences)
- k: sequence length (tokens)
- k_fft: effective FFT length = next_pow2(2k-1), accounting for FFT padding
- a: per-FFT-token memory cost (bytes per FFT token per sequence)
- b: per-sequence overhead (bytes per sequence)
- c: fixed model overhead (bytes)

Uses least squares regression across multiple (n, k) configurations,
repeated over several trials to compute 95% confidence intervals.

Usage:
    uv run python src/scripts/estimate_vram_coefficients.py [model_size]

    model_size: One of 1k, 16k, 32k, 160k, 450k, 1m (default: 1k)
"""

import gc
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats

from giggleml.models.hyena_dna import HyenaDNA
from giggleml.utils.vram import fft_length

NUCLEOTIDES = "ACGT"


def random_sequence(length: int) -> str:
    """Generate a random nucleotide sequence."""
    return "".join(random.choices(NUCLEOTIDES, k=length))


def reset_cuda_state(model: HyenaDNA, device: torch.device) -> None:
    """Reset CUDA state to prevent trial leakage."""
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def measure_vram(
    model: HyenaDNA,
    n: int,
    k: int,
    device: torch.device,
) -> int:
    """Measure peak reserved VRAM for a forward pass.

    Clears CUDA caches, runs a forward pass with random sequences,
    and returns the peak reserved memory above baseline. Reserved memory
    is more accurate for capacity planning than allocated memory.

    Args:
        model: HyenaDNA model (already on device).
        n: Batch size (number of sequences).
        k: Sequence length (tokens).
        device: CUDA device.

    Returns:
        Peak reserved VRAM in bytes (excluding model parameters).
    """
    reset_cuda_state(model, device)

    # Baseline reserved memory (model params + cuda context)
    baseline = torch.cuda.memory_reserved(device)

    # Generate random sequences and run forward pass
    sequences = [random_sequence(k) for _ in range(n)]
    batch = model.collate(sequences)
    batch = {key: val.to(device) for key, val in batch.items()}

    with torch.inference_mode():
        _ = model(batch)
        torch.cuda.synchronize(device)

    peak = torch.cuda.max_memory_reserved(device)

    # Clean up batch tensors before next measurement
    del batch, sequences

    return peak - baseline


@dataclass
class VRAMCoefficients:
    """Estimated VRAM coefficients with 95% confidence intervals.

    For V(n, k) = a*n*k_fft + b*n + c where n=batch_size, k_fft=next_pow2(2k-1).

    Attributes:
        a: Per-FFT-token coefficient (bytes per FFT token per sequence).
        a_ci: 95% confidence interval for a.
        b: Per-sequence coefficient (bytes per sequence).
        b_ci: 95% confidence interval for b.
        c: Fixed model overhead (bytes).
        c_ci: 95% confidence interval for c.
    """

    a: float
    a_ci: tuple[float, float]
    b: float
    b_ci: tuple[float, float]
    c: float
    c_ci: tuple[float, float]


def estimate_coefficients(
    model: HyenaDNA,
    device: torch.device,
    n_trials: int = 10,
    warmup_trials: int = 2,
) -> VRAMCoefficients:
    """Estimate VRAM coefficients a, b, c with 95% confidence intervals.

    Runs multiple trials of VRAM measurements across randomized (n, k) configurations,
    then fits V(n,k) = a*n*k_fft + b*n + c using least squares regression,
    where k_fft = next_pow2(2k-1) accounts for FFT convolution padding.

    CIs are computed from the OLS standard errors of the regression coefficients,
    pooled across trials.

    Args:
        model: HyenaDNA model to measure.
        device: CUDA device to run on.
        n_trials: Number of trials for statistics (after warmup).
        warmup_trials: Number of warmup trials to discard.

    Returns:
        VRAMCoefficients with point estimates and 95% CIs for a, b, c.
    """
    # Randomized configurations to get variation in measurements
    # Sample batch sizes and sequence lengths from ranges
    batch_sizes = [1, 2, 4, 8, 16]
    seq_lengths = [128, 256, 512, 768, 1024]

    # Collect all measurements across trials
    all_measurements: list[tuple[int, int, int]] = []

    for trial in range(warmup_trials + n_trials):
        # Generate random configs for this trial
        configs = [
            (random.choice(batch_sizes), random.choice(seq_lengths))
            for _ in range(10)
        ]

        for n, k in configs:
            vram = measure_vram(model, n, k, device)
            # Only collect after warmup
            if trial >= warmup_trials:
                all_measurements.append((n, k, vram))

        if trial >= warmup_trials:
            print(f"Trial {trial - warmup_trials + 1}/{n_trials}: collected {len(configs)} measurements")

    # Build design matrix for least squares: V = a*n*k_fft + b*n + c
    # where k_fft = next_pow2(2k-1) accounts for FFT padding
    X = np.array([[n * fft_length(k), n, 1.0] for n, k, _ in all_measurements])
    y = np.array([float(v) for _, _, v in all_measurements])

    # Solve least squares and compute standard errors
    # Using numpy for OLS: coeffs = (X'X)^-1 X'y
    # Var(coeffs) = sigma^2 (X'X)^-1
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    coeffs = XtX_inv @ X.T @ y
    a, b, c = coeffs

    # Compute residuals and estimate sigma^2
    y_pred = X @ coeffs
    residuals = y - y_pred
    n_obs = len(y)
    n_params = 3
    dof = n_obs - n_params
    sigma2 = float(np.sum(residuals**2) / dof)

    # Standard errors of coefficients
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    se_a, se_b, se_c = se

    # 95% CI using t-distribution
    t_crit = float(stats.t.ppf(0.975, dof))

    print(f"\nPooled regression on {n_obs} measurements (dof={dof})")
    print(f"  a = {a:.2f} ± {se_a:.2f}")
    print(f"  b = {b:.2f} ± {se_b:.2f}")
    print(f"  c = {c:.0f} ± {se_c:.0f}")
    print(f"  R² = {1 - np.sum(residuals**2) / np.sum((y - y.mean())**2):.6f}")

    return VRAMCoefficients(
        a=float(a),
        a_ci=(float(a - t_crit * se_a), float(a + t_crit * se_a)),
        b=float(b),
        b_ci=(float(b - t_crit * se_b), float(b + t_crit * se_b)),
        c=float(c),
        c_ci=(float(c - t_crit * se_c), float(c + t_crit * se_c)),
    )


def measure_hyena_dna(model_size: str = "1k") -> None:
    """Load a HyenaDNA model and estimate its VRAM coefficients.

    Args:
        model_size: HyenaDNA size variant (1k, 16k, 32k, 160k, 450k, 1m).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VRAM estimation")

    device = torch.device("cuda:0")
    print(f"Loading HyenaDNA({model_size})...")
    model = HyenaDNA(model_size)
    model = model.to(device)
    model.eval()

    print(f"Model loaded. seq_max={model.seq_max}, edim={model.edim}")
    print()

    print("Estimating VRAM coefficients...")
    print("V(n, k) = a*n*k_fft + b*n + c  where k_fft = next_pow2(2k-1)")
    print()

    result = estimate_coefficients(model, device, n_trials=10, warmup_trials=2)

    print()
    print("=" * 60)
    print("Results (95% CI):")
    print("=" * 60)
    print(
        f"a (per-fft-tok):  {result.a:>12.2f} bytes  [{result.a_ci[0]:.2f}, {result.a_ci[1]:.2f}]"
    )
    print(
        f"b (per-sequence): {result.b:>12.2f} bytes  [{result.b_ci[0]:.2f}, {result.b_ci[1]:.2f}]"
    )
    print(
        f"c (fixed):        {result.c:>12.0f} bytes  [{result.c_ci[0]:.0f}, {result.c_ci[1]:.0f}]"
    )
    print()

    # Human-readable summary
    a_mb = result.a / (1024 * 1024)
    b_mb = result.b / (1024 * 1024)
    c_mb = result.c / (1024 * 1024)
    print("Human-readable:")
    print(f"  Per FFT token: {a_mb * 1000:.3f} KB")
    print(f"  Per sequence: {b_mb:.3f} MB")
    print(f"  Fixed:        {c_mb:.1f} MB")


if __name__ == "__main__":
    size = sys.argv[1] if len(sys.argv) > 1 else "1k"
    measure_hyena_dna(size)
