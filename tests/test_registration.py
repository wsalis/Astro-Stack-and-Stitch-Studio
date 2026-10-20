from __future__ import annotations

import numpy as np
import torch

from gpustacker.detection import build_star_psf, choose_psf_size, detect_stars
from gpustacker.registration import estimate_alignment, warp_to_reference


def test_detects_most_stars(scene):
    image, xy = scene
    stars = detect_stars(image, thresh_sigma=5)
    assert len(stars) >= int(0.8 * len(xy))
    assert 2.0 < stars.median_fwhm() < 4.5


def test_psf_size_and_star_psf(scene):
    image, _ = scene
    stars = detect_stars(image)
    k = choose_psf_size(stars.median_fwhm())
    assert k % 2 == 1 and 11 <= k <= 51
    psf = build_star_psf(image, stars, 15)
    assert psf is not None and psf.shape == (15, 15)
    assert abs(psf.sum() - 1.0) < 1e-5
    assert psf.argmax() == (15 * 15) // 2


def test_alignment_recovers_shift_and_warp(scene):
    from scipy import ndimage

    image, _ = scene
    ref_stars = detect_stars(image)
    dx, dy = 4.3, -2.6
    moved = ndimage.shift(image, (dy, dx), order=3, mode="nearest").astype(np.float32)
    src_stars = detect_stars(moved)
    align = estimate_alignment(src_stars, ref_stars)
    assert abs(align.shift[0] + dx) < 0.15 and abs(align.shift[1] + dy) < 0.15
    assert abs(align.rotation_deg) < 0.05 and abs(align.scale - 1) < 1e-3

    warped = warp_to_reference(torch.from_numpy(moved)[None], align)
    inner = (slice(None), slice(16, -16), slice(16, -16))
    diff = (warped[inner] - torch.from_numpy(image)[None][inner]).abs()
    assert torch.nanmedian(diff) < 2.0
    assert torch.isnan(warped).any()  # uncovered border is NaN


def test_lanczos_keeps_star_sharpness_and_blocks_dark_halos():
    """Half-pixel Lanczos resampling must not broaden a star (the old clamp did, ~7%) nor ring below the local sky."""

    from gpustacker.registration import _warp_lanczos

    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    sigma = 2.9 / 2.3548
    sky = 3000.0
    star = 20000 * np.exp(-((xx - 31.5) ** 2 + (yy - 31.5) ** 2) / (2 * sigma**2))
    img = torch.from_numpy((sky + star).astype(np.float32))[None]
    sx, sy = torch.from_numpy(xx + 0.5), torch.from_numpy(yy + 0.5)
    out = _warp_lanczos(img, sx, sy, 3)[0].numpy() - sky
    truth = 20000 * np.exp(-((xx + 0.5 - 31.5) ** 2 + (yy + 0.5 - 31.5) ** 2) / (2 * sigma**2))
    inner = (slice(8, -8), slice(8, -8))
    # second moment (width) of the resampled star vs the analytic truth at the shifted position
    def width(a):
        a = np.clip(a[inner], 0, None); yy_, xx_ = np.mgrid[0 : a.shape[0], 0 : a.shape[1]]
        cx, cy = (a * xx_).sum() / a.sum(), (a * yy_).sum() / a.sum()
        return np.sqrt((a * ((xx_ - cx) ** 2 + (yy_ - cy) ** 2)).sum() / a.sum())
    assert abs(width(out) / width(truth) - 1.0) < 0.01
    assert abs(out[inner].sum() / truth[inner].sum() - 1.0) < 0.01  # flux conserved
    assert out.min() >= -1e-3  # no dark ring below the sky


def _warp_lanczos_reference(image: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, a: int = 3) -> torch.Tensor:
    """Plain single-pass Lanczos used as the oracle for the tiled implementation."""

    from gpustacker.registration import _lanczos_weights

    channels, height, width = image.shape
    out_shape = tuple(sx.shape)
    x0, y0 = torch.floor(sx), torch.floor(sy)
    offs = torch.arange(-a + 1, a + 1, device=image.device, dtype=torch.float32)
    wx = _lanczos_weights((sx - x0).unsqueeze(0) - offs.view(-1, 1, 1))
    wy = _lanczos_weights((sy - y0).unsqueeze(0) - offs.view(-1, 1, 1))
    xi_all = (x0.unsqueeze(0) + offs.view(-1, 1, 1)).clamp(0, width - 1).long()
    yi_all = (y0.unsqueeze(0) + offs.view(-1, 1, 1)).clamp(0, height - 1).long()
    flat = image.reshape(channels, -1)
    acc = torch.zeros((channels, *out_shape), device=image.device, dtype=image.dtype)
    wsum = torch.zeros(out_shape, device=image.device)
    floor = torch.full((channels, *out_shape), float("inf"), device=image.device, dtype=image.dtype)
    for iy in range(2 * a):
        row = yi_all[iy] * width
        for ix in range(2 * a):
            w = wy[iy] * wx[ix]
            vals = flat[:, (row + xi_all[ix]).reshape(-1)].reshape(channels, *out_shape)
            acc += vals * w
            wsum += w
            floor = torch.minimum(floor, vals)
    return torch.maximum(acc / wsum.clamp_min(1e-6), floor)


def test_lanczos_tiled_matches_reference():
    from gpustacker.registration import Alignment, _warp_lanczos, source_coords

    rng = np.random.default_rng(5)
    height, width = 150, 70
    img = torch.from_numpy(rng.normal(1000, 20, (3, height, width)).astype(np.float32))
    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(12):
        cy, cx = rng.uniform(5, height - 5), rng.uniform(5, width - 5)
        img += torch.from_numpy((rng.uniform(500, 8000) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.2**2))).astype(np.float32))
    theta = np.deg2rad(1.7)
    matrix = np.array([[np.cos(theta), -np.sin(theta), 2.3], [np.sin(theta), np.cos(theta), -1.4], [0, 0, 1]])
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        x = img.to(device)
        sx, sy = source_coords(Alignment(matrix=matrix, matched=0, rotation_deg=1.7, scale=1.0, shift=(2.3, -1.4)), height, width, x.device)
        expected = _warp_lanczos_reference(x, sx, sy, 3)
        for tile_rows in (7, 64, 1000):  # ragged tiles, several tiles, single tile
            actual = _warp_lanczos(x, sx, sy, 3, tile_rows=tile_rows)
            assert torch.allclose(actual, expected, rtol=1e-5, atol=5e-2), f"{device} tile_rows={tile_rows}: {(actual - expected).abs().max():.4g}"
