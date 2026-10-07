from __future__ import annotations

import numpy as np
import torch

from gpustacker.debayer import debayer_bilinear
from gpustacker.io import load_frame, normalize_layout, save_fits
from gpustacker.stacking import FrameNorm, FrameStore, StackSettings, combine_tensor, stack_store


def test_stack_store_per_channel_normalisation(tmp_path):
    """Frames with a different sky colour must be matched to the reference channel by channel."""

    from gpustacker.backend import select_backend

    backend = select_backend("cpu")
    base = np.full((3, 16, 16), 1000.0, np.float32) + np.random.default_rng(1).normal(0, 2, (3, 16, 16)).astype(np.float32)
    tint = np.array([80.0, 0.0, -60.0], np.float32)[:, None, None]  # different night: redder sky, bluer deficit
    store = FrameStore(tmp_path, (3, 16, 16))
    store.add("ref", base, FrameNorm(offset=np.array([1000.0, 1000.0, 1000.0]), scale=np.ones(3)))
    store.add("other", base + tint, FrameNorm(offset=np.array([1080.0, 1000.0, 940.0]), scale=np.ones(3)))
    out = stack_store(store, backend, StackSettings(method="none", normalize=True)).image
    assert np.allclose(out, base, atol=1e-3)  # colour cast of the second frame fully removed


def test_normalize_layout_variants():
    assert normalize_layout(np.zeros((10, 12))).shape == (1, 10, 12)
    assert normalize_layout(np.zeros((3, 10, 12))).shape == (3, 10, 12)
    assert normalize_layout(np.zeros((10, 12, 3))).shape == (3, 10, 12)


def test_fits_roundtrip(tmp_path):
    data = np.random.default_rng(0).random((3, 8, 9)).astype(np.float32)
    path = save_fits(tmp_path / "x.fit", data, [("NFRAMES", 3, "n")])
    loaded, meta = load_frame(path)
    assert loaded.shape == (3, 8, 9)
    assert np.allclose(loaded, data)
    assert meta.header["NFRAMES"] == 3


def test_debayer_flat_field_is_flat():
    cfa = torch.full((1, 16, 16), 100.0)
    rgb = debayer_bilinear(cfa, "RGGB")
    assert rgb.shape == (3, 16, 16)
    assert torch.allclose(rgb, torch.full_like(rgb, 100.0), atol=1e-4)


def test_debayer_recovers_pure_red():
    cfa = torch.zeros((1, 16, 16))
    cfa[0, 0::2, 0::2] = 200.0  # R sites only for RGGB
    rgb = debayer_bilinear(cfa, "RGGB")
    assert torch.allclose(rgb[0, 2:-2, 2:-2], torch.full((12, 12), 200.0), atol=1e-3)
    assert torch.all(rgb[1:] == 0)


def test_vng_flat_and_colour_field():
    from gpustacker.debayer import debayer_vng

    assert torch.allclose(debayer_vng(torch.full((1, 16, 16), 100.0), "RGGB"), torch.full((3, 16, 16), 100.0), atol=1e-3)
    # uniform colour (R=300, G=100, B=50) must come back exactly at every pixel
    cfa = torch.full((1, 32, 32), 100.0)
    cfa[0, 0::2, 0::2] = 300.0
    cfa[0, 1::2, 1::2] = 50.0
    rgb = debayer_vng(cfa, "RGGB")[:, 4:-4, 4:-4]
    for c, v in enumerate((300.0, 100.0, 50.0)):
        assert torch.allclose(rgb[c], torch.full_like(rgb[c], v), atol=1e-3)


def test_vng_keeps_star_sharper_than_bilinear():
    from gpustacker.debayer import debayer_vng

    yy, xx = np.mgrid[0:48, 0:48].astype(np.float32)
    star = 2000.0 * np.exp(-((xx - 24.3) ** 2 + (yy - 23.6) ** 2) / (2 * 0.9**2)) + 100.0
    cfa = torch.from_numpy(star)[None]  # grey star: every colour equals the scene
    truth_peak = float(star.max())
    vng = debayer_vng(cfa, "RGGB").mean(dim=0)
    bil = debayer_bilinear(cfa, "RGGB").mean(dim=0)
    assert float(vng.max()) > float(bil.max())
    assert abs(float(vng.max()) - truth_peak) < abs(float(bil.max()) - truth_peak)


def test_sigma_clip_rejects_outlier():
    rng = np.random.default_rng(1)
    stack = torch.from_numpy(rng.normal(100, 2, (12, 1, 6, 6)).astype(np.float32))
    stack[3, 0, 2, 2] = 5000.0
    out, low, high = combine_tensor(stack, StackSettings(method="sigma", sigma_low=3, sigma_high=3))
    assert high >= 1 and abs(float(out[0, 2, 2]) - 100) < 5


def test_winsorized_and_median_ignore_nan():
    stack = torch.full((5, 1, 4, 4), 10.0)
    stack[0] = float("nan")
    stack[1, 0, 0, 0] = 90.0
    for method in ("winsorized", "median", "none", "percentile"):
        out, _, _ = combine_tensor(stack, StackSettings(method=method))
        assert torch.isfinite(out).all()
        if method != "none":
            assert abs(float(out[0, 0, 0]) - 10.0) < 1e-3, method
