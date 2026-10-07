from __future__ import annotations

import numpy as np
import torch

from gpustacker.cosmetic import CosmeticSettings, cosmetic_correct


def _noisy(shape=(1, 64, 64), seed=0):
    return torch.from_numpy(np.random.default_rng(seed).normal(500, 10, shape).astype(np.float32))


def test_fixes_hot_and_cold_mono():
    img = _noisy()
    img[0, 10, 10] = 60000.0
    img[0, 30, 40] = 0.0
    out, stats = cosmetic_correct(img, CosmeticSettings(cold_sigma=4.0, fix_cold=True))
    assert stats.hot >= 1 and stats.cold >= 1
    assert abs(float(out[0, 10, 10]) - 500) < 40 and abs(float(out[0, 30, 40]) - 500) < 40


def test_star_survives():
    img = _noisy()
    yy, xx = np.mgrid[0:64, 0:64]
    star = 20000 * np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 1.3**2))
    img[0] += torch.from_numpy(star.astype(np.float32))
    out, _ = cosmetic_correct(img, CosmeticSettings())
    assert float(out[0, 32, 32]) > 0.8 * float(img[0, 32, 32])


def test_cfa_faint_stars_keep_flux_while_hot_pixels_go():
    """Star wings/cores on a Bayer mosaic must not be mistaken for hot pixels (they were, at ~10% flux loss)."""

    rng = np.random.default_rng(3)
    size = 256
    sky = np.full((size, size), 3000.0, np.float32)
    sky[0::2, 0::2] -= 130  # R sites darker, as under a dual-band filter
    sky[1::2, 1::2] -= 100  # B sites
    noise = 10.0
    img = sky + rng.normal(0, noise, (size, size)).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    centres = [(y, x) for y in range(20, size - 20, 24) for x in range(20, size - 20, 24)]
    peaks = rng.uniform(6 * noise, 60 * noise, len(centres))  # faint to moderate stars
    clean = np.zeros_like(img)
    for (cy, cx), pk in zip(centres, peaks):
        clean += pk * np.exp(-((xx - cx - 0.3) ** 2 + (yy - cy + 0.2) ** 2) / (2 * 1.15**2))  # FWHM 2.7 px
    img += clean.astype(np.float32)
    hot = [(5, 7), (100, 201), (177, 33), (240, 240), (63, 128)]
    for y, x in hot:
        img[y, x] += 800.0
    t = torch.from_numpy(img)[None]
    out, stats = cosmetic_correct(t, CosmeticSettings(hot_sigma=3.0), cfa=True)
    out = out[0].numpy()
    for y, x in hot:
        assert abs(out[y, x] - sky[y, x]) < 6 * noise  # fixed
    before = np.array([(img[cy - 2 : cy + 3, cx - 2 : cx + 3] - sky[cy - 2 : cy + 3, cx - 2 : cx + 3]).sum() for cy, cx in centres])
    after = np.array([(out[cy - 2 : cy + 3, cx - 2 : cx + 3] - sky[cy - 2 : cy + 3, cx - 2 : cx + 3]).sum() for cy, cx in centres])
    retained = after / before
    assert np.median(retained) > 0.995 and retained.min() > 0.97, (np.median(retained), retained.min())
    assert stats.hot < len(hot) + 0.003 * img.size  # only the 3-sigma noise tail, no mass replacement of star pixels


def test_cfa_uses_same_colour_neighbours():
    # strong red/blue contrast: in a plain 3x3 median the bright R sites look "hot"
    img = torch.full((1, 64, 64), 200.0)
    img[0, 0::2, 0::2] = 4000.0
    img += (_noisy() - 500.0) * 0.1
    hot_r, hot_c = 20, 20  # an R site
    img[0, hot_r, hot_c] = 60000.0
    out, stats = cosmetic_correct(img, CosmeticSettings(hot_sigma=4.0), cfa=True)
    assert stats.hot == 1 and stats.cold == 0
    assert abs(float(out[0, hot_r, hot_c]) - 4000) < 50
    untouched = out.clone()
    untouched[0, hot_r, hot_c] = img[0, hot_r, hot_c]
    assert torch.equal(untouched, img)


def test_disabled_is_identity():
    img = _noisy()
    out, stats = cosmetic_correct(img, CosmeticSettings(fix_hot=False, fix_cold=False))
    assert torch.equal(out, img) and stats.hot == stats.cold == 0
