from __future__ import annotations

import numpy as np
import pytest
import torch

from gpustacker.backend import select_backend
from gpustacker.detection import Stars, detect_stars, fwhm_grid, fwhm_grid_with_counts
from gpustacker.diagnostics import crop_box_for, effective_frames, full_depth_box, stack_noise
from gpustacker.normalization import LocalNormSettings, block_stats, fit_norm_maps
from gpustacker.quality import FilterSettings, StarSet, aperture_flux, compute_weights, field_weight_factors, sample_field_grid, select_photometry_stars, transparency
from gpustacker.registration import Alignment, estimate_alignment, refine_alignment, warp_to_reference
from gpustacker.stacking import FrameNorm, FrameStore, StackSettings, combine_full, combine_tensor, stack_store
from gpustacker.tilt import build_map, describe, frame_scatter, session_median


# ----------------------------------------------------------------------------- autocrop


def test_full_depth_box_finds_inner_rectangle():
    cov = np.zeros((200, 300), dtype=np.uint16)
    cov[20:180, 40:260] = 10  # full depth
    cov[10:20, :] = 7  # partial border
    cov[:, 260:280] = 5
    box = full_depth_box(cov, 10)
    assert box == (20, 180, 40, 260)


def test_full_depth_box_skips_corner_notch():
    cov = np.full((120, 120), 8, dtype=np.uint16)
    cov[:30, :30] = 3  # a flipped frame missed this corner
    y0, y1, x0, x1 = full_depth_box(cov, 8)
    assert (y1 - y0) * (x1 - x0) == 90 * 120
    assert cov[y0:y1, x0:x1].min() == 8


def test_crop_box_fraction():
    cov = np.full((64, 64), 20, dtype=np.uint16)
    cov[:8, :] = 19
    assert crop_box_for(cov, 20, 1.0) == (8, 64, 0, 64)
    assert crop_box_for(cov, 20, 0.9) == (0, 64, 0, 64)
    assert crop_box_for(cov, 20, 0.0) is None


def test_stack_noise_and_effective_frames():
    rng = np.random.default_rng(0)
    img = rng.normal(100, 3.0, (1, 256, 256)).astype(np.float32)
    assert abs(stack_noise(img) - 3.0) < 0.4
    assert effective_frames([1, 1, 1, 1]) == pytest.approx(4.0)
    assert effective_frames([1, 0.05, 0.05]) < 1.25


def test_fwhm_grid_localises_soft_corner():
    rng = np.random.default_rng(3)
    n = 400
    x = rng.uniform(0, 400, n).astype(np.float32)
    y = rng.uniform(0, 200, n).astype(np.float32)
    sigma = np.where((x >= 300) & (y < 50), 2.0, 1.0).astype(np.float32)  # top-right cell is twice as soft
    stars = Stars(x, y, np.ones(n, np.float32), sigma, sigma, np.ones(n, np.float32), n)
    grid = fwhm_grid(stars, (200, 400))
    assert grid.shape == (4, 4)
    assert np.isfinite(grid).all()
    assert grid[0, 3] == pytest.approx(2.0 * 2.3548, rel=1e-3)
    assert grid[3, 0] == pytest.approx(1.0 * 2.3548, rel=1e-3)
    sparse = fwhm_grid(Stars(x[:5], y[:5], np.ones(5, np.float32), sigma[:5], sigma[:5], np.ones(5, np.float32), 5), (200, 400))
    assert np.isnan(sparse).all()


def test_fwhm_grid_with_counts_reports_cell_confidence():
    x = np.array([10, 20, 30, 40, 10, 20, 30, 40], dtype=np.float32)
    y = np.array([10, 20, 30, 40, 60, 70, 80, 90], dtype=np.float32)
    widths = np.arange(1, 9, dtype=np.float32)
    stars = Stars(x, y, np.ones(8, np.float32), widths, widths, np.ones(8, np.float32), 8)

    grid, counts = fwhm_grid_with_counts(stars, (100, 100), cells=2, min_stars=2)

    np.testing.assert_array_equal(counts, [[4, 0], [4, 0]])
    assert grid[0, 0] == pytest.approx(np.median(stars.fwhm[:4]))
    assert np.isnan(grid[0, 1])


def test_tilt_map_plane_points_to_soft_side():
    rng = np.random.default_rng(5)
    n = 2000
    x = rng.uniform(0, 1000, n).astype(np.float32)
    y = rng.uniform(0, 800, n).astype(np.float32)
    # focus error grows toward the right edge: FWHM 3 px on the left, 4 px on the right
    sigma = ((3.0 + 1.0 * x / 1000.0) / 2.3548).astype(np.float32)
    a = sigma * 1.1  # mild elongation at a fixed angle
    b = sigma / 1.1
    theta = np.full(n, np.radians(30.0), np.float32)
    stars = Stars(x, y, np.ones(n, np.float32), a, b, np.ones(n, np.float32), n, theta)
    tm = build_map(stars, (800, 1000), cells=4)
    assert tm.soft_side == "right"
    assert tm.tilt_x == pytest.approx(1.0, abs=0.15)
    assert abs(tm.tilt_y) < 0.1
    assert abs(tm.curvature) < 0.1
    assert np.nanmedian(tm.elongation) == pytest.approx(1.21, abs=0.01)
    assert np.nanmedian(tm.angle_deg) == pytest.approx(30.0, abs=1.0)
    assert np.nanmin(tm.coherence) > 0.95
    sess = session_median([tm, tm])
    assert sess is not None and sess.soft_side == "right"
    assert frame_scatter([tm, tm, tm]) == pytest.approx(0.0, abs=1e-6)
    assert any("RIGHT" in line for line in describe(tm))


# ----------------------------------------------------------------------------- photometry / weights


def _field(seed, dim=256, n=40, amp=4000.0, sky=100.0, noise=5.0, fwhm=3.0, transp=1.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:dim, 0:dim].astype(np.float32)
    layout = np.random.default_rng(7)
    pos = layout.uniform(30, dim - 30, (n, 2))
    amps = layout.uniform(0.3, 1.5, n) * amp
    img = np.full((dim, dim), sky, np.float32)
    s = fwhm / 2.3548
    for (x, y), a in zip(pos, amps):
        img += transp * a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    return img, pos


def test_aperture_flux_tracks_transparency():
    ref, _ = _field(1)
    dim = ref.shape[0]
    stars = detect_stars(ref)
    phot = select_photometry_stars(stars, (dim, dim), 3.0)
    assert len(phot) >= 15
    f_ref = aperture_flux(torch.from_numpy(ref), phot)
    cloudy, _ = _field(2, transp=0.5, noise=4.0)  # dimmer AND smoother
    f_cloud = aperture_flux(torch.from_numpy(cloudy), phot)
    t, n = transparency(f_cloud, f_ref, min_stars=10)
    assert n >= 10 and abs(t - 0.5) < 0.08


def test_psfsw_ranks_cloudy_and_moonlit_frames_low():
    # noise alone would rank the cloudy frame HIGHEST (it is smoother); psfsw must not
    noise = np.array([5.0, 4.0, 9.0, 5.0])  # clear, cloudy(smooth), moonlit(noisy), clear
    fwhm = np.array([3.0, 3.1, 3.0, 4.5])
    transp = np.array([1.0, 0.5, 1.0, 1.0])
    w_noise = compute_weights("noise", noise, fwhm, transp)
    assert w_noise[1] == w_noise.max()
    w = compute_weights("psfsw", noise, fwhm, transp)
    assert w[1] < w[0] and w[2] < w[0] and w[3] == pytest.approx(w[0])
    w2 = compute_weights("psfsw+fwhm", noise, fwhm, transp)
    assert w2[3] < w2[0]
    assert np.allclose(compute_weights("none", noise, fwhm, transp), 1.0)
    # unknown transparency falls back to noise-only for that frame
    w3 = compute_weights("psfsw", noise, fwhm, np.array([1.0, np.nan, 1.0, 1.0]))
    assert np.isfinite(w3).all()


def test_field_weight_factors_ignore_persistent_soft_corner():
    base = np.full((4, 4), 3.0)
    base[0, 3] = 4.0
    grids = [base.copy() for _ in range(3)]
    factors = field_weight_factors(grids)
    for factor in factors:
        np.testing.assert_allclose(factor, 1.0)
    missing = field_weight_factors([np.full((4, 4), np.nan) for _ in range(3)])
    for factor in missing:
        np.testing.assert_allclose(factor, 1.0)

    grids[2][2, 1] *= 1.3
    factors = field_weight_factors(grids)
    assert factors[2][2, 1] < factors[0][2, 1]
    assert factors[2][2, 1] == pytest.approx(1.0 / 1.3**2, rel=1e-5)


def test_field_weight_factors_shrink_low_confidence_toward_neutral():
    base = np.full((4, 4), 3.0)
    soft = base.copy()
    soft[2, 1] *= 1.3
    grids = [base.copy(), base.copy(), soft]
    high_counts = [np.full((4, 4), 80) for _ in grids]
    low_counts = [counts.copy() for counts in high_counts]
    low_counts[2][2, 1] = 8

    high_confidence = field_weight_factors(grids, confidence_counts=high_counts)[2][2, 1]
    low_confidence = field_weight_factors(grids, confidence_counts=low_counts)[2][2, 1]

    assert 1.0 / 1.3**2 < high_confidence < low_confidence < 1.0


def test_sample_field_grid_interpolates_and_handles_missing_cells():
    grid = np.array([[1.0, np.nan], [3.0, 5.0]])
    xs = np.array([0.0, 9.5])
    ys = np.array([0.0, 9.5])
    values = sample_field_grid(grid, xs, ys, (10, 10))
    np.testing.assert_allclose(values, [1.0, 5.0])


def test_combine_full_accepts_spatial_weights():
    frames = torch.tensor([[[[10.0, 100.0]]], [[[30.0, 40.0]]]])
    weights = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])
    result = combine_full(frames, StackSettings(method="none", normalize=False), weights)
    torch.testing.assert_close(result.image, torch.tensor([[[10.0, 40.0]]]))


def test_stack_store_applies_registered_spatial_weight_maps(tmp_path, backend):
    store = FrameStore(tmp_path, (1, 2, 2))
    store.add("sharp-left", np.full((1, 2, 2), 10.0, np.float32), FrameNorm(weight_map=np.array([[1.0, 0.0], [1.0, 0.0]], np.float32)))
    store.add("sharp-right", np.full((1, 2, 2), 30.0, np.float32), FrameNorm(weight_map=np.array([[0.0, 1.0], [0.0, 1.0]], np.float32)))
    result = stack_store(store, backend, StackSettings(method="none", normalize=False))
    np.testing.assert_allclose(result.image[0], [[10.0, 30.0], [10.0, 30.0]])


# ----------------------------------------------------------------------------- local normalisation


def test_local_norm_maps_remove_gradient():
    rng = np.random.default_rng(3)
    dim = 256
    yy, xx = np.mgrid[0:dim, 0:dim].astype(np.float32)
    ref = (100 + rng.normal(0, 4, (dim, dim))).astype(np.float32)[None]
    frame = (100 + 60 * xx / dim + rng.normal(0, 8, (dim, dim))).astype(np.float32)[None]  # gradient + 2x noise
    settings = LocalNormSettings(block=32, smooth_sigma=0.5)
    ref_t, frame_t = torch.from_numpy(ref), torch.from_numpy(frame)
    med, mad, frac = block_stats(ref_t, settings.block)
    scale, offset = fit_norm_maps(frame_t, med, mad, frac, settings)
    assert scale.shape == (1, 8, 8) and 0.4 < scale.mean() < 0.65
    s_full = torch.nn.functional.interpolate(torch.from_numpy(scale)[None], size=(dim, dim), mode="bilinear", align_corners=False)[0]
    o_full = torch.nn.functional.interpolate(torch.from_numpy(offset)[None], size=(dim, dim), mode="bilinear", align_corners=False)[0]
    fixed = (frame_t * s_full + o_full).numpy()[0]
    left, right = fixed[:, 20:60].mean(), fixed[:, -60:-20].mean()
    assert abs(left - right) < 3.0 and abs(fixed.mean() - 100) < 2.0


# ----------------------------------------------------------------------------- registration refinement


def test_refine_alignment_reduces_residual():
    from scipy import ndimage

    ref, _ = _field(4, dim=400, n=120, noise=2.0)
    ref_stars = detect_stars(ref)
    # similarity + a small radial distortion
    yy, xx = np.mgrid[0:400, 0:400].astype(np.float32)
    r2 = ((xx - 200) ** 2 + (yy - 200) ** 2) / 200**2
    map_x = xx + 3.0 + 6.0 * r2 * (xx - 200) / 200
    map_y = yy - 2.0 + 6.0 * r2 * (yy - 200) / 200
    src = ndimage.map_coordinates(ref, [map_y, map_x], order=3, mode="nearest").astype(np.float32)
    src_stars = detect_stars(src)
    base = estimate_alignment(src_stars, ref_stars)
    refined = refine_alignment(base, src_stars, ref_stars)
    assert refined.inverse_poly is not None and refined.refined_matches >= 60
    assert refined.residual_rms < 0.4
    # round-trip: reference grid -> fitted source coords -> true forward map should land back on the grid
    from gpustacker.registration import source_coords

    ys, xs = np.mgrid[40:360:20, 40:360:20].astype(np.float64)

    def roundtrip(align):
        sx, sy = source_coords(align, 400, 400, torch.device("cpu"))
        sx, sy = sx.numpy()[40:360:20, 40:360:20].astype(np.float64), sy.numpy()[40:360:20, 40:360:20].astype(np.float64)
        rr2 = ((sx - 200) ** 2 + (sy - 200) ** 2) / 200**2
        fx = sx + 3.0 + 6.0 * rr2 * (sx - 200) / 200
        fy = sy - 2.0 + 6.0 * rr2 * (sy - 200) / 200
        return float(np.mean(np.hypot(fx - xs, fy - ys)))

    err_base, err_ref = roundtrip(base), roundtrip(refined)
    assert err_ref < 0.5 * err_base and err_ref < 0.35


def test_alignment_flipped_property():
    assert Alignment(np.eye(3), 0, 179.2, 1.0, (0, 0)).flipped
    assert not Alignment(np.eye(3), 0, 0.3, 1.0, (0, 0)).flipped


# ----------------------------------------------------------------------------- large-scale rejection


def test_large_scale_rejection_grows_trail():
    rng = np.random.default_rng(5)
    stack = torch.from_numpy(rng.normal(100, 2, (12, 1, 40, 80)).astype(np.float32))
    # bright satellite trail in frame 3 with faint wings that sigma clipping alone misses
    stack[3, 0, 20, :] += 400.0
    stack[3, 0, 19, :] += 5.0
    stack[3, 0, 21, :] += 5.0
    base = combine_full(stack, StackSettings(method="sigma"))
    grown = combine_full(stack, StackSettings(method="sigma", large_scale=True, large_scale_box=7, large_scale_density=0.1, large_scale_grow=2))
    assert grown.rej_high[3].sum() > base.rej_high[3].sum()
    assert bool(grown.rej_high[3, 0, 19].all()) and bool(grown.rej_high[3, 0, 21].all())
    assert grown.rej_high[[i for i in range(12) if i != 3]].sum() <= base.rej_high[[i for i in range(12) if i != 3]].sum() + 5
    assert abs(float(grown.image[0, 20].mean()) - 100) < 2


def test_combine_full_reports_coverage():
    stack = torch.full((6, 1, 4, 4), 10.0)
    stack[0:2, 0, 0, 0] = float("nan")
    out = combine_full(stack, StackSettings(method="winsorized"))
    assert int(out.valid[:, 0, 0, 0].sum()) == 4 and int(out.valid[:, 0, 1, 1].sum()) == 6
