"""Shared synthetic star-field fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from gpustacker.backend import select_backend


def make_field(height: int = 256, width: int = 320, n_stars: int = 60, fwhm: float = 3.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (noise-free scene, star xy) with a faint nebula and Gaussian stars."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    scene = 50.0 + 40.0 * np.exp(-(((xx - width * 0.6) / 60.0) ** 2 + ((yy - height * 0.4) / 40.0) ** 2))
    xs = rng.uniform(20, width - 20, n_stars)
    ys = rng.uniform(20, height - 20, n_stars)
    amps = rng.uniform(300, 6000, n_stars)
    sigma = fwhm / 2.3548
    for x, y, a in zip(xs, ys, amps):
        scene += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return scene.astype(np.float32), np.column_stack([xs, ys])


def shifted_noisy(scene: np.ndarray, dx: float, dy: float, rng: np.random.Generator, read_noise: float = 4.0) -> np.ndarray:
    from scipy import ndimage

    moved = ndimage.shift(scene, (dy, dx), order=3, mode="nearest")
    noisy = rng.poisson(np.clip(moved, 0, None)).astype(np.float32) + rng.normal(0, read_noise, moved.shape).astype(np.float32)
    return np.clip(noisy, 0, None).astype(np.float32)


@pytest.fixture(scope="session")
def scene() -> tuple[np.ndarray, np.ndarray]:
    return make_field()


@pytest.fixture
def light_dir(tmp_path: Path, scene: tuple[np.ndarray, np.ndarray]) -> Path:
    """Six shifted, noisy FITS lights with a hot column and a satellite streak in one frame."""

    rng = np.random.default_rng(42)
    base, _ = scene
    folder = tmp_path / "lights"
    folder.mkdir()
    shifts = [(0, 0), (3.2, -1.7), (-2.4, 2.9), (5.1, 4.4), (-4.6, -3.3), (1.3, 6.0)]
    for i, (dx, dy) in enumerate(shifts):
        frame = shifted_noisy(base, dx, dy, rng)
        if i == 2:
            frame[100:104, :] += 900.0  # streak
        hdr = fits.Header()
        hdr["EXPTIME"] = 60.0
        hdr["EGAIN"] = 1.0
        hdr["RDNOISE"] = 4.0
        hdr["FILTER"] = "L"
        fits.PrimaryHDU(data=frame, header=hdr).writeto(folder / f"light_{i:02d}.fit")
    return folder


@pytest.fixture(params=["cpu", "auto"], ids=["cpu", "gpu"])
def backend(request):
    import torch

    if request.param == "auto" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return select_backend(request.param)
