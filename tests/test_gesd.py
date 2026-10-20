from __future__ import annotations

import numpy as np
import pytest
import torch

from gpustacker.stacking import StackSettings, combine_tensor, gesd_critical_table


def test_critical_values_match_rosner():
    # Rosner (1983) / NIST handbook example: n=54, alpha=0.05 -> lambda_1 = 3.158, lambda_10 = 3.085
    table = gesd_critical_table(54, 10, 0.05)
    assert table[0, 54] == pytest.approx(3.158, abs=2e-3)
    assert table[9, 54] == pytest.approx(3.085, abs=2e-3)


def test_gesd_rejects_planted_outliers():
    rng = np.random.default_rng(3)
    stack = torch.from_numpy(rng.normal(100, 2, (40, 1, 8, 8)).astype(np.float32))
    stack[5, 0, 2, 2] = 400.0  # satellite
    stack[9, 0, 2, 2] = 350.0
    stack[17, 0, 5, 5] = 20.0  # dead-ish pixel in one frame
    out, low, high = combine_tensor(stack, StackSettings(method="gesd"))
    assert high >= 2 and low >= 1
    assert abs(float(out[0, 2, 2]) - 100) < 2.5
    assert abs(float(out[0, 5, 5]) - 100) < 2.5


def test_gesd_low_false_positive_rate_on_noise():
    rng = np.random.default_rng(11)
    stack = torch.from_numpy(rng.normal(100, 2, (60, 1, 64, 64)).astype(np.float32))
    _, low, high = combine_tensor(stack, StackSettings(method="gesd", gesd_low_relax=1.0))
    frac = (low + high) / stack.numel()
    assert frac < 0.01


def test_gesd_tolerates_nan_and_small_stacks():
    stack = torch.full((30, 1, 6, 6), 50.0) + torch.randn(30, 1, 6, 6) * 0.5
    stack[0:7] = float("nan")  # uncovered in 7 frames
    stack[12, 0, 1, 1] = 900.0
    out, _, high = combine_tensor(stack, StackSettings(method="gesd"))
    assert torch.isfinite(out).all() and high >= 1 and abs(float(out[0, 1, 1]) - 50) < 1.0
    tiny = torch.randn(3, 1, 4, 4)
    out, low, high = combine_tensor(tiny, StackSettings(method="gesd"))
    assert torch.isfinite(out).all() and low == high == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gesd_cpu_gpu_agree(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(5)
    stack = torch.from_numpy(rng.normal(200, 5, (25, 3, 10, 10)).astype(np.float32))
    stack[3, :, 4, 4] = 2000.0
    ref, rl, rh = combine_tensor(stack, StackSettings(method="gesd"))
    got, gl, gh = combine_tensor(stack.to(device), StackSettings(method="gesd"))
    assert torch.allclose(ref, got.cpu(), atol=1e-3) and (rl, rh) == (gl, gh)


def _gesd_reference(x: torch.Tensor, max_frac: float, alpha: float, low_relax: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The original float64 (k, pixels)-buffer implementation, kept as the oracle for the lean rewrite."""

    n = x.shape[0]
    k = int(max(1, min(n - 3, np.floor(n * max_frac))))
    flat = x.reshape(n, -1)
    pixels = flat.shape[1]
    median = torch.nanmedian(flat, dim=0).values
    shifted = flat - median
    sorted_vals, sort_idx = torch.sort(shifted, dim=0)
    valid = ~torch.isnan(shifted)
    count = valid.sum(dim=0).to(torch.float64)
    n_valid = count.to(torch.long)
    s1 = torch.where(valid, shifted, torch.zeros_like(shifted)).sum(dim=0, dtype=torch.float64)
    s2 = torch.where(valid, shifted * shifted, torch.zeros_like(shifted)).sum(dim=0, dtype=torch.float64)
    cols = torch.arange(pixels)
    lo = torch.zeros(pixels, dtype=torch.long)
    hi = (n_valid - 1).clamp_min(0)
    stats = torch.full((k, pixels), float("-inf"))
    removed = torch.zeros((k, pixels), dtype=torch.long)
    for i in range(k):
        active = count >= 3
        c = count.clamp_min(1)
        mean = s1 / c
        std = ((s2 - s1 * s1 / c) / (c - 1).clamp_min(1)).clamp_min(1e-24).sqrt()
        vlo = sorted_vals[lo, cols].double()
        vhi = sorted_vals[hi, cols].double()
        dlo = (mean - vlo) / std / low_relax
        dhi = (vhi - mean) / std
        pick_hi = dhi >= dlo
        r_i = torch.where(pick_hi, dhi, dlo)
        stats[i] = torch.where(active, r_i, torch.full_like(r_i, float("-inf"))).float()
        removed[i] = sort_idx[torch.where(pick_hi, hi, lo), cols]
        v = torch.where(pick_hi, vhi, vlo)
        s1 = torch.where(active, s1 - v, s1)
        s2 = torch.where(active, s2 - v * v, s2)
        count = torch.where(active, count - 1, count)
        hi = torch.where(active & pick_hi, hi - 1, hi).clamp_min(0)
        lo = torch.where(active & ~pick_hi, lo + 1, lo).clamp_max(n - 1)
    table = torch.from_numpy(gesd_critical_table(n, k, alpha))
    lam = table[:, n_valid]
    steps = torch.arange(1, k + 1).view(k, 1)
    n_out = torch.where(stats > lam, steps, torch.zeros_like(steps)).max(dim=0).values
    reject = torch.zeros((n, pixels), dtype=torch.bool)
    for i in range(k):
        sel = n_out > i
        reject[removed[i, sel], cols[sel]] = True
    return (reject & (flat < median)).reshape(x.shape), (reject & ~(flat < median)).reshape(x.shape)


@pytest.mark.parametrize("low_relax", [1.0, 1.5])
@pytest.mark.parametrize("n", [48, 80])  # k = 14 (eager) and k = 24 (CUDA-graph path on GPU)
def test_gesd_matches_reference_implementation(low_relax, n):
    """Rejection masks of the memory-lean float32 walk must equal the original float64 implementation."""

    from gpustacker.stacking import _gesd_clip

    rng = np.random.default_rng(21)
    pixels = 600
    data = rng.normal(1500, 100, (n, pixels)).astype(np.float32)
    # planted high outliers, dark outliers, partial coverage (NaN tails of varying length) and ties
    data[rng.integers(0, n, 200), rng.integers(0, pixels, 200)] += rng.uniform(500, 20000, 200).astype(np.float32)
    data[rng.integers(0, n, 60), rng.integers(0, pixels, 60)] -= rng.uniform(400, 1200, 60).astype(np.float32)
    for col in range(0, pixels, 7):
        data[: int(rng.integers(0, n - 3)), col] = np.nan
    data[:, 500] = np.nan  # nothing valid
    data[:, 501] = 1500.0  # all equal
    data[:5, 502] = 9000.0  # tied outliers
    x = torch.from_numpy(data.reshape(n, 3, 10, 20))
    _, low, high = _gesd_clip(x, 0.3, 0.05, low_relax)
    ref_low, ref_high = _gesd_reference(x, 0.3, 0.05, low_relax)
    assert torch.equal(low, ref_low) and torch.equal(high, ref_high)
    assert int(high.sum()) > 150 and int(low.sum()) > 30
    if torch.cuda.is_available():
        _, glow, ghigh = _gesd_clip(x.cuda(), 0.3, 0.05, low_relax)
        assert torch.equal(glow.cpu(), ref_low) and torch.equal(ghigh.cpu(), ref_high)
