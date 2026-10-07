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
