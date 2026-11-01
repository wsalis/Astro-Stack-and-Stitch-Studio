from __future__ import annotations

import numpy as np
import pytest
import torch

from gpustacker.detection import build_star_psf, choose_psf_size, detect_stars
from gpustacker.registration import Alignment, estimate_alignment, reference_overlap_fraction, union_reference_grid, warp_to_reference


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


def test_union_grid_includes_shifted_frame_outside_reference():
    reference = Alignment.identity()
    shifted = Alignment(
        np.array([[1.0, 0.0, -4.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]),
        0,
        0.0,
        1.0,
        (-4.0, 3.0),
    )
    shape, origin = union_reference_grid([(reference, (4, 5)), (shifted, (4, 5))])
    assert shape == (7, 9)
    assert origin == (-4, 0)

    image = torch.arange(20, dtype=torch.float32).reshape(1, 4, 5)
    warped = warp_to_reference(image, shifted, output_shape=shape, origin=origin)
    assert torch.allclose(warped[:, 3:7, :5], image, atol=1e-5)
    assert torch.isnan(warped[:, :3]).all()


def test_reference_overlap_fraction_measures_shifted_footprint():
    shifted = Alignment(
        np.array([[1.0, 0.0, 80.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        0,
        0.0,
        1.0,
        (80.0, 0.0),
    )

    assert reference_overlap_fraction(shifted, (100, 100), (100, 100)) == 0.2


def _mixed_star_fields(seed: int = 3):
    """Wide reference catalogue and a narrower, coarser, rotated frame covering ~22% of it."""

    from gpustacker.detection import Stars

    rng = np.random.default_rng(seed)
    ref_shape, src_shape = (2400, 1350), (1200, 680)
    n = 900
    ref_xy = np.column_stack([rng.uniform(0, ref_shape[1] - 1, n), rng.uniform(0, ref_shape[0] - 1, n)])
    flux = rng.pareto(1.2, n) + 1.0
    scale, theta, offset = 1.05, np.radians(17.0), np.array([660.0, 1250.0])
    c, s = np.cos(theta), np.sin(theta)
    matrix = np.array([[scale * c, -scale * s, 0.0], [scale * s, scale * c, 0.0], [0.0, 0.0, 1.0]])
    centre = np.array([(src_shape[1] - 1) / 2, (src_shape[0] - 1) / 2])
    matrix[:2, 2] = offset - matrix[:2, :2] @ centre  # source centre lands on `offset` in the reference
    src_xy = (ref_xy - matrix[:2, 2]) @ np.linalg.inv(matrix[:2, :2]).T
    inside = (src_xy[:, 0] >= 0) & (src_xy[:, 0] <= src_shape[1] - 1) & (src_xy[:, 1] >= 0) & (src_xy[:, 1] <= src_shape[0] - 1)
    keep = np.flatnonzero(inside)
    keep = keep[np.argsort(-flux[keep])][:90]  # shallower sensor: only its brightest stars
    src_xy = src_xy[keep] + rng.normal(0, 0.05, (len(keep), 2))

    def stars(xy: np.ndarray, f: np.ndarray) -> Stars:
        order = np.argsort(-f)
        ones = np.ones(len(xy))
        return Stars(xy[order, 0], xy[order, 1], f[order], ones, ones, f[order], total_detected=len(xy))

    return stars(src_xy, flux[keep]), stars(ref_xy, flux), src_shape, ref_shape, matrix


def test_mixed_alignment_matches_narrow_coarse_frame_to_wide_reference():
    from gpustacker.registration import confirm_alignment, estimate_alignment_mixed, refine_alignment

    src, ref, src_shape, ref_shape, truth = _mixed_star_fields()
    align = estimate_alignment_mixed(src, ref, src_shape, ref_shape, scale_hint=1.04)
    assert abs(align.scale - 1.05) < 2e-3 and abs(align.rotation_deg - 17.0) < 0.1
    corners = np.array([[0, 0, 1], [src_shape[1] - 1, src_shape[0] - 1, 1]], dtype=np.float64)
    assert np.abs((corners @ align.matrix.T)[:, :2] - (corners @ truth.T)[:, :2]).max() < 0.5
    confirmed, tested = confirm_alignment(align.matrix, src, ref, ref_shape)
    assert tested >= 40 and confirmed == tested

    refined = refine_alignment(align, src, ref, shapes=(src_shape, ref_shape))
    assert refined.inverse_poly is not None and refined.refined_matches >= 60

    # reverse roles: wide frame registered onto the narrow reference
    reverse = estimate_alignment_mixed(ref, src, ref_shape, src_shape, scale_hint=1 / 1.04)
    assert abs(reverse.scale - 1 / 1.05) < 2e-3 and abs(reverse.rotation_deg + 17.0) < 0.1


def test_mixed_alignment_rejects_scale_contradicting_metadata():
    from gpustacker.registration import estimate_alignment_mixed

    src, ref, src_shape, ref_shape, _ = _mixed_star_fields()
    with pytest.raises(ValueError, match="mixed-setup alignment failed"):
        estimate_alignment_mixed(src, ref, src_shape, ref_shape, scale_hint=1.6)


def test_mixed_hint_only_for_differing_grid_or_scale():
    from types import SimpleNamespace

    from gpustacker.pipeline import _mixed_align_hint

    def meta(shape, focal):
        return SimpleNamespace(shape=shape, header={"XPIXSZ": 2.9, "YPIXSZ": 2.9, "FOCALLEN": focal})

    assert _mixed_align_hint(meta((1920, 1080), 250.0), meta((1920, 1080), 250.0)) is None
    assert _mixed_align_hint(SimpleNamespace(shape=(100, 80), header={}), SimpleNamespace(shape=(100, 80), header={})) is None
    hint = _mixed_align_hint(meta((1920, 1080), 250.0), meta((3840, 2160), 260.0))
    assert hint is not None and hint[:2] == ((1920, 1080), (3840, 2160)) and abs(hint[2] - 1.04) < 1e-3
    same_grid = _mixed_align_hint(meta((1920, 1080), 150.0), meta((1920, 1080), 250.0))
    assert same_grid is not None and abs(same_grid[2] - 250 / 150) < 1e-3
    unknown_scale = _mixed_align_hint(SimpleNamespace(shape=(100, 80), header={}), SimpleNamespace(shape=(200, 160), header={}))
    assert unknown_scale == ((100, 80), (200, 160), None)


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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("tile_rows", [1, 128])
def test_lanczos_nonfinite_coordinates_are_missing_data(device, tile_rows):
    from gpustacker.registration import _warp_lanczos

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    image = torch.arange(48, dtype=torch.float32, device=device).reshape(3, 4, 4)
    sx = torch.tensor([[float("nan"), float("inf"), -float("inf"), 1.0], [1.0, 1.0, 1.0, 2.0]], device=device)
    sy = torch.tensor([[1.0, 1.0, 1.0, 1.0], [float("nan"), float("inf"), -float("inf"), 2.0]], device=device)

    actual = _warp_lanczos(image, sx, sy, tile_rows=tile_rows)

    assert torch.isnan(actual[:, :, :3]).all()
    torch.testing.assert_close(actual[:, 0, 3], image[:, 1, 1])
    torch.testing.assert_close(actual[:, 1, 3], image[:, 2, 2])


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
