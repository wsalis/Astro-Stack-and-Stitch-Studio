from __future__ import annotations

import numpy as np
import pytest

from gpustacker.detection import Stars, detect_stars
from gpustacker.mfdeconv import MFDeconvSettings, build_luma_variance_map, build_variance_map, prepare_assets, run_mfdeconv


def test_variance_map_counts_sky_once_and_applies_linear_scale():
    image = np.full((32, 32), 100.0, dtype=np.float32)
    settings = MFDeconvSettings(variance_smooth_sigma=0.0)

    variance = build_variance_map(image, gain=2.0, read_noise=4.0, settings=settings, scale=1.5)

    assert np.allclose(variance, 1.5**2 * (100.0 * 2.0 + 4.0**2) / 2.0**2)


def test_luma_variance_map_applies_each_channel_scale():
    frame = np.full((3, 32, 32), 100.0, dtype=np.float32)
    settings = MFDeconvSettings(variance_smooth_sigma=0.0)
    scales = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    variance = build_luma_variance_map(frame, 2.0, 4.0, settings, scales)
    channel_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    expected = 54.0 * np.sum((channel_weights * scales) ** 2)

    assert np.allclose(variance, expected)


def test_mfdeconv_assets_exclude_integration_rejections(monkeypatch):
    import gpustacker.mfdeconv as mfdeconv_module

    frame = np.full((1, 32, 32), 100.0, dtype=np.float32)
    rejection = np.zeros((32, 32), dtype=bool)
    rejection[8:12, 10:14] = True
    psf = np.full((5, 5), 1.0 / 25.0, dtype=np.float32)
    stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)))
    monkeypatch.setattr(mfdeconv_module, "estimate_psf", lambda *_args: (psf, "synthetic", 3.0, stars))
    settings = MFDeconvSettings(use_star_masks=False, use_variance_maps=False)

    assets = prepare_assets([frame], settings, rejection_masks=[rejection])

    keep = assets[0].keep_mask
    assert keep is not None
    assert np.all(keep[8:12, 10:14] == 0.0)
    assert np.all(keep[~rejection] == 1.0)


def _blurred_frames(scene: np.ndarray, n: int, fwhm_blur: float, seed: int = 7) -> list[np.ndarray]:
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    sigma = fwhm_blur / 2.3548
    frames = []
    for _ in range(n):
        blurred = ndimage.gaussian_filter(scene, sigma)
        noisy = rng.poisson(np.clip(blurred, 0, None)).astype(np.float32) + rng.normal(0, 3, scene.shape).astype(np.float32)
        frames.append(np.clip(noisy, 0, None)[None].astype(np.float32))
    return frames


def test_mfdeconv_per_channel_psfs_avoid_colour_rings(scene):
    """Blue focused tighter than red/green (dual-band through a refractor): one luma PSF over-sharpens
    blue and rings it (coloured halos). Per-channel PSFs must keep every channel's ring shallow."""

    from scipy import ndimage

    from gpustacker.backend import select_backend

    sharp, _ = scene
    rng = np.random.default_rng(3)
    fwhms = (3.4, 3.4, 2.6)  # R, G, B
    frames = []
    for _ in range(4):
        rgb = [ndimage.gaussian_filter(sharp, f / 2.3548) for f in fwhms]
        noisy = [rng.poisson(np.clip(c, 0, None)).astype(np.float32) + rng.normal(0, 3, sharp.shape).astype(np.float32) for c in rgb]
        frames.append(np.clip(np.stack(noisy), 0, None).astype(np.float32))
    res = run_mfdeconv(frames, select_backend("cpu"), MFDeconvSettings(iterations=20, early_stop_tol=0.0, dering_star_sigma=-1.0, color_mode="perchannel"), gains=[1.0] * 4)
    assert res.assets[0].channel_psfs is not None
    out = res.image
    stars = detect_stars(out[1])
    yy, xx = np.mgrid[-10:11, -10:11]
    rr = np.hypot(xx, yy)
    troughs = []
    for c in range(3):
        sky = np.median(out[c])
        noise = 1.4826 * np.median(np.abs(out[c] - sky))
        prof = []
        for x, y in zip(stars.x, stars.y):
            cx, cy = int(round(x)), int(round(y))
            if 12 <= cx < out.shape[2] - 12 and 12 <= cy < out.shape[1] - 12:
                cut = out[c, cy - 10 : cy + 11, cx - 10 : cx + 11] - sky
                prof.append([np.median(cut[(rr >= a) & (rr < a + 1)]) for a in range(3, 9)])
        troughs.append(np.median(np.array(prof), axis=0).min() / noise)
    # blue must not ring much deeper than red/green (the colour-halo signature)
    assert troughs[2] > min(troughs[0], troughs[1]) - 0.5, troughs


def test_lrgb_keeps_star_and_sky_colour():
    """lrgb: luminance is replaced, but signal keeps its colour ratios and the sky keeps its colour."""

    from gpustacker.mfdeconv import _apply_luminance

    rng = np.random.default_rng(0)
    dim = 96
    yy, xx = np.mgrid[0:dim, 0:dim].astype(np.float32)
    star = np.exp(-((xx - 48) ** 2 + (yy - 48) ** 2) / (2 * 1.5**2))
    sky = np.array([300.0, 450.0, 330.0], np.float32)[:, None, None]
    tint = np.array([1.0, 0.6, 0.3], np.float32)[:, None, None]  # an orange star
    colour = sky + 2000.0 * tint * star + rng.normal(0, 3, (3, dim, dim)).astype(np.float32)
    lum = 0.2126 * colour[0] + 0.7152 * colour[1] + 0.0722 * colour[2]
    sharper = lum + 800.0 * (np.exp(-((xx - 48) ** 2 + (yy - 48) ** 2) / (2 * 1.0**2)) - 0.45 * star)
    out = _apply_luminance(colour.astype(np.float32), sharper.astype(np.float32))
    out_l = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]
    assert np.allclose(out_l, sharper, atol=0.5)  # luminance is the deconvolved one
    core = (slice(46, 51), slice(46, 51))
    ratio = [(out[c][core] - sky[c, 0, 0]).mean() / (out[0][core] - sky[0, 0, 0]).mean() for c in range(3)]
    assert np.allclose(ratio, tint[:, 0, 0], atol=0.05)  # star keeps its colour
    corner = (slice(0, 20), slice(0, 20))
    assert np.allclose([out[c][corner].mean() for c in range(3)], sky[:, 0, 0], atol=1.5)  # sky colour untouched


def _ring_deficit(out: np.ndarray, truth: np.ndarray, cx: float, cy: float, r0: int, r1: int) -> float:
    """Largest azimuthal-median dip of ``out`` below ``truth`` over 1px annuli r0..r1 (a visible ring)."""

    yy, xx = np.mgrid[0 : out.shape[0], 0 : out.shape[1]]
    r = np.hypot(xx - cx, yy - cy)
    return max(float(np.median(truth[(r >= a) & (r < a + 1)] - out[(r >= a) & (r < a + 1)])) for a in range(r0, r1))


def test_mfdeconv_sharpens(scene, backend):
    sharp, _ = scene
    frames = _blurred_frames(sharp, 4, fwhm_blur=2.5)
    settings = MFDeconvSettings(iterations=15, relax=0.7, kappa=2.0)
    result = run_mfdeconv(frames, backend, settings, gains=[1.0] * 4, read_noises=[3.0] * 4)
    assert result.image.shape == frames[0].shape
    assert np.isfinite(result.image).all() and (result.image >= 0).all()
    before = detect_stars(frames[0][0]).median_fwhm()
    after = detect_stars(result.image[0]).median_fwhm()
    assert after < before * 0.9, (before, after)
    assert len(result.assets) == 4 and all(a.psf.sum() == pytest.approx(1.0, abs=1e-4) for a in result.assets)


def test_mfdeconv_tiled_matches_whole(scene):
    from gpustacker.backend import select_backend

    sharp, _ = scene
    frames = _blurred_frames(sharp, 3, fwhm_blur=2.0)
    backend = select_backend("cpu")
    whole = run_mfdeconv(frames, backend, MFDeconvSettings(iterations=6, psf_size=11, use_star_masks=False, use_variance_maps=False)).image
    tiled = run_mfdeconv(frames, backend, MFDeconvSettings(iterations=6, psf_size=11, tile_size=160, tile_overlap=48, use_star_masks=False, use_variance_maps=False)).image
    rel = np.abs(whole - tiled) / (np.abs(whole) + 1.0)
    assert np.median(rel) < 0.02


def test_mfdeconv_uncovered_borders_stay_finite(scene, backend):
    """NaN borders (pixels no frame covers) must not spread through the convolutions."""

    sharp, _ = scene
    frames = _blurred_frames(sharp, 3, fwhm_blur=2.0)
    for i, f in enumerate(frames):
        f[:, : 12 + i, :] = np.nan
        f[:, :, -(12 + i) :] = np.nan
    out = run_mfdeconv(frames, backend, MFDeconvSettings(iterations=8, psf_size=11, tile_size=160, tile_overlap=40)).image
    assert np.isfinite(out).all()
    inner = out[:, 40:-40, 40:-40]
    assert inner.min() > 0 and np.median(inner) > 0.5 * np.median(sharp)


def test_mfdeconv_dering_prevents_dark_rings(scene):
    """The star-ring lift must clearly reduce the coherent dark ring, without adding a halo.

    A halo (lifted mean / flattened texture around stars) is what the earlier smoothed-deficit lift did.
    """

    from gpustacker.backend import select_backend
    from gpustacker.detection import detect_stars

    sharp, _ = scene
    frames = _blurred_frames(sharp, 4, fwhm_blur=2.5)
    backend = select_backend("cpu")
    yy, xx = np.mgrid[-10:11, -10:11]
    rr = np.hypot(xx, yy)

    def rings(star_sigma):
        out = run_mfdeconv(frames, backend, MFDeconvSettings(iterations=30, early_stop_tol=0.0, psf_size=15, dering_star_sigma=star_sigma), gains=[1.0] * 4).image[0]
        stars = detect_stars(out)
        sky = np.median(out)
        noise = 1.4826 * np.median(np.abs(out - sky))
        profiles, outer = [], []
        for x, y in zip(stars.x, stars.y):
            cx, cy = int(round(x)), int(round(y))
            if 12 <= cx < out.shape[1] - 12 and 12 <= cy < out.shape[0] - 12:
                cut = out[cy - 10 : cy + 11, cx - 10 : cx + 11] - sky
                profiles.append([np.median(cut[(rr >= a) & (rr < a + 1)]) for a in range(3, 9)])
                outer.append(cut[(rr >= 9) & (rr <= 10)].mean())
        return np.median(np.array(profiles), axis=0).min() / noise, np.median(outer) / noise

    trough_off, outer_off = rings(-1.0)
    trough_on, outer_on = rings(1.0)
    assert trough_off < -1.0  # the solver does dig a ring here
    assert trough_on > 0.7 * trough_off  # ~40% shallower on this crowded synthetic field
    assert outer_on < outer_off + 0.15  # no added brightness (halo) beyond the ring


def test_mfdeconv_no_dark_ring_inside_bright_nebula():
    """A star embedded in nebula far above the sky must not get a ring dug below the surrounding nebula."""

    from gpustacker.backend import select_backend

    dim = 128
    yy, xx = np.mgrid[0:dim, 0:dim].astype(np.float32)
    sky = 50.0
    nebula = 600.0 * np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / (2 * 30.0**2))  # bright, smooth
    star = 48000.0 * np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / (2 * 0.5**2))  # point-like: PSF measured from it = blur
    scene = (sky + nebula + star).astype(np.float32)
    frames = _blurred_frames(scene, 4, fwhm_blur=2.6, seed=11)
    backend = select_backend("cpu")
    out = run_mfdeconv(frames, backend, MFDeconvSettings(iterations=20, early_stop_tol=0.0, psf_size=13, kappa=2.0), gains=[1.0] * 4, header_fwhms=[2.85] * 4).image[0]
    r = np.hypot(xx - 64, yy - 64)
    noise = 1.4826 * np.median(np.abs(out[r > 40] - np.median(out[r > 40])))
    # Gibbs ringing of a point source is inherent; the nebula-tracking floor must stop it well short
    # of the sky (the old sky-only floor let it dig ~600 ADU down to sky level here)
    assert _ring_deficit(out, sky + nebula, 64, 64, 4, 13) < 0.5 * 600.0
    assert out[np.abs(r - 5) < 1].min() > sky + 0.5 * nebula[64, 64]
    # and the floor must not have flattened the dark outer nebula falloff upward
    outer = (r > 45) & (r < 55)
    assert abs(np.median(out[outer] - (sky + nebula)[outer])) < 2.0 * noise


def test_keep_mask_does_not_dim_unsaturated_stars():
    """With the keep-mask on, an unsaturated star must sharpen, not turn into a dimmed flat disc.

    Masking every detected star removed the core from the data term; the wing residuals then drove
    the core down ~20% on real data (flat disc with a dark rim).
    """

    from gpustacker.backend import select_backend

    dim = 128
    yy, xx = np.mgrid[0:dim, 0:dim].astype(np.float32)
    nebula = 300.0 * np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / (2 * 35.0**2))
    star = 36000.0 * np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / (2 * 0.5**2))  # point-like, like a real star
    scene = (50.0 + nebula + star).astype(np.float32)
    frames = _blurred_frames(scene, 4, fwhm_blur=2.6, seed=21)
    stacked = np.median(np.stack(frames), axis=0)[0]
    backend = select_backend("cpu")
    # one star -> Gaussian PSF fallback; give it the true width (blur 2.6 px on a 0.5 px-sigma star)
    kw = dict(iterations=15, kappa=1.5, psf_size=13, early_stop_tol=0.0)
    out = run_mfdeconv(frames, backend, MFDeconvSettings(use_star_masks=True, **kw), gains=[1.0] * 4, header_fwhms=[2.85] * 4).image[0]
    unmasked = run_mfdeconv(frames, backend, MFDeconvSettings(use_star_masks=False, **kw), gains=[1.0] * 4, header_fwhms=[2.85] * 4).image[0]
    assert out[64, 64] > 1.5 * stacked[64, 64]  # the core sharpens, it is not dimmed
    assert np.allclose(out, unmasked)  # an unsaturated star is not masked at all
