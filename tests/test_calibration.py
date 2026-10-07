from __future__ import annotations

import numpy as np
import torch

from gpustacker.backend import select_backend
from gpustacker.calibration import AUTO_PEDESTAL_ADU, Masters, calibrate, load_masters
from gpustacker.io import save_fits


def test_normalised_master_dark_is_applied_to_adu_lights(tmp_path):
    """PixInsight saves masters in [0, 1]; subtracting that from 16-bit ADU lights did nothing (real bug)."""

    backend = select_backend("cpu")
    rng = np.random.default_rng(0)
    dark_adu = (2800 + rng.normal(0, 3, (1, 32, 32))).astype(np.float32)
    dark_adu[0, 5, 5] = 9000.0  # hot pixel that must be removed
    save_fits(tmp_path / "dark.fit", dark_adu / 65535.0, [])
    masters = load_masters(backend, dark=tmp_path / "dark.fit")
    sky = 180.0
    light = torch.from_numpy(dark_adu + sky)
    out = calibrate(light, masters).numpy()
    assert np.allclose(out, sky + AUTO_PEDESTAL_ADU, atol=0.5)  # dark (incl. hot pixel) fully removed


def test_dark_sky_not_clipped_after_dark_subtraction():
    """Frames whose sky sits barely above the dark level must not lose their negative noise tail."""

    rng = np.random.default_rng(1)
    dark = torch.full((1, 64, 64), 2800.0)
    noise = rng.normal(0, 12, (1, 64, 64)).astype(np.float32)
    light = dark + 15.0 + torch.from_numpy(noise)
    out = calibrate(light, Masters(dark=dark)).numpy()
    assert (out > 0).all()
    assert abs(float(out.mean()) - (15.0 + AUTO_PEDESTAL_ADU)) < 1.0  # unbiased: no clipping


def test_matching_units_unchanged():
    dark = torch.full((1, 8, 8), 2800.0)
    light = torch.full((1, 8, 8), 3000.0)
    out = calibrate(light, Masters(dark=dark, pedestal=50.0)).numpy()
    assert np.allclose(out, 250.0)
