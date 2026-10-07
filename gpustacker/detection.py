"""Star detection, background estimation, FWHM and empirical PSF extraction (CPU, NumPy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

try:
    import sep  # type: ignore
except ImportError:  # pragma: no cover
    try:
        import sep_pjw as sep  # type: ignore
    except ImportError:
        sep = None

FWHM_PER_SIGMA = 2.3548


@dataclass
class Stars:
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    a: np.ndarray
    b: np.ndarray
    peak: np.ndarray
    total_detected: int = 0  # before the brightest-N cap
    theta: np.ndarray | None = None  # major-axis position angle, radians, x-axis anticlockwise (sep convention)

    def __len__(self) -> int:
        return int(self.x.size)

    @property
    def fwhm(self) -> np.ndarray:
        return FWHM_PER_SIGMA * np.sqrt(np.maximum(self.a * self.b, 1e-6))

    @property
    def elongation(self) -> np.ndarray:
        return np.maximum(self.a, 1e-6) / np.maximum(self.b, 1e-6)

    def median_fwhm(self) -> float:
        return float(np.median(self.fwhm)) if len(self) else float("nan")

    def xy(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])


def luma(image: np.ndarray) -> np.ndarray:
    """(C, H, W) -> (H, W) float32 brightness."""

    if image.ndim == 2:
        return np.asarray(image, dtype=np.float32)
    if image.shape[0] == 1:
        return np.asarray(image[0], dtype=np.float32)
    return (0.2126 * image[0] + 0.7152 * image[1] + 0.0722 * image[2]).astype(np.float32)


def robust_stats(image2d: np.ndarray, sample_stride: int = 8) -> tuple[float, float]:
    """(median, MAD-sigma) of finite pixels, sub-sampled for speed."""

    sample = np.asarray(image2d[::sample_stride, ::sample_stride], dtype=np.float32).ravel()
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 1.0
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med))) * 1.4826
    return med, max(mad, 1e-9)


def analyse_frame(image2d: np.ndarray, thresh_sigma: float) -> tuple[Stars, float, float]:
    """Star detection + background/noise for one luma frame; safe to run in a worker process."""

    stars = detect_stars(image2d, thresh_sigma=thresh_sigma)
    bg, noise = robust_stats(image2d)
    return stars, bg, noise


def fwhm_grid(stars: Stars, shape: tuple[int, int], cells: int = 4, min_stars: int = 8) -> np.ndarray:
    return fwhm_grid_with_counts(stars, shape, cells, min_stars)[0]


def fwhm_grid_with_counts(stars: Stars, shape: tuple[int, int], cells: int = 4, min_stars: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """(cells, cells) median FWHM per image cell in sensor coordinates (row 0 = top); NaN below min_stars.

    Stars are pre-registration so the map tracks the sensor, not the sky: tilt or field curvature
    shows as a fixed pattern across frames, seeing/focus drift as a pattern that changes. Also return
    the number of detected stars contributing to each median."""

    grid = np.full((cells, cells), np.nan, dtype=np.float32)
    counts = np.zeros((cells, cells), dtype=np.int32)
    if len(stars) == 0:
        return grid, counts
    height, width = shape
    col = np.clip((stars.x * cells / max(width, 1)).astype(int), 0, cells - 1)
    row = np.clip((stars.y * cells / max(height, 1)).astype(int), 0, cells - 1)
    fw = stars.fwhm
    for r in range(cells):
        for c in range(cells):
            sel = (row == r) & (col == c)
            counts[r, c] = int(sel.sum())
            if counts[r, c] >= min_stars:
                grid[r, c] = np.median(fw[sel])
    return grid, counts


def estimate_background(image2d: np.ndarray, box: int = 64) -> tuple[np.ndarray, float]:
    """Return (background map, global RMS) for a 2-D float32 image."""

    img = np.ascontiguousarray(image2d, dtype=np.float32)
    if sep is not None:
        bkg = sep.Background(img, bw=box, bh=box)
        return np.asarray(bkg.back(), dtype=np.float32), float(bkg.globalrms)
    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med))) * 1.4826
    return np.full_like(img, med), max(mad, 1e-6)


def detect_stars(image2d: np.ndarray, thresh_sigma: float = 5.0, min_area: int = 5, max_sources: int = 2000) -> Stars:
    img = np.ascontiguousarray(image2d, dtype=np.float32)
    back, rms = estimate_background(img)
    sub = img - back
    if sep is not None:
        sigma = thresh_sigma
        objs = None
        for _ in range(6):
            try:
                objs = sep.extract(sub, sigma * rms, minarea=min_area)
            except Exception:  # sep raises on buffer overflow with dense fields
                sigma *= 1.6
                continue
            if len(objs) > max_sources * 4:
                sigma *= 1.6  # still too dense: tighten, but keep this result as a fallback
                continue
            break
        if objs is None:
            objs = np.empty(0)
        if len(objs) == 0:
            return Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)))
        order = np.argsort(objs["flux"])[::-1][:max_sources]
        o = objs[order]
        return Stars(o["x"].astype(np.float32), o["y"].astype(np.float32), o["flux"].astype(np.float32), o["a"].astype(np.float32), o["b"].astype(np.float32), o["peak"].astype(np.float32), int(len(objs)), o["theta"].astype(np.float32))
    return _detect_stars_scipy(sub, rms, thresh_sigma, min_area, max_sources)


def _detect_stars_scipy(sub: np.ndarray, rms: float, thresh_sigma: float, min_area: int, max_sources: int) -> Stars:
    mask = sub > thresh_sigma * rms
    labels, count = ndimage.label(mask)
    if count == 0:
        return Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)))
    idx = np.arange(1, count + 1)
    areas = ndimage.sum(mask, labels, idx)
    keep = areas >= min_area
    idx = idx[keep]
    if idx.size == 0:
        return Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)))
    flux = np.asarray(ndimage.sum(sub, labels, idx), dtype=np.float32)
    peak = np.asarray(ndimage.maximum(sub, labels, idx), dtype=np.float32)
    com = np.asarray(ndimage.center_of_mass(sub, labels, idx), dtype=np.float32)
    y, x = com[:, 0], com[:, 1]
    sigma = np.sqrt(np.asarray(areas[keep], dtype=np.float32) / np.pi) / 2.0
    order = np.argsort(flux)[::-1][:max_sources]
    return Stars(x[order], y[order], flux[order], sigma[order], sigma[order], peak[order], int(flux.size))


def choose_psf_size(fwhm: float, lo: int = 11, hi: int = 51, sigmas: float = 4.0) -> int:
    """Odd kernel size k = 2*ceil(sigmas*sigma)+1 clamped to [lo, hi] (Marek 2026, auto-k at 4 sigma)."""

    sigma = max(float(fwhm), 0.5) / FWHM_PER_SIGMA
    k = 2 * int(np.ceil(sigmas * sigma)) + 1
    return int(min(hi, max(lo, k | 1)))


def psf_support_radius(psf: np.ndarray, floor_frac: float) -> int:
    """Outermost radius whose azimuthal mean still exceeds ``floor_frac`` of the peak.

    Real PSFs have Moffat wings far beyond 4 Gaussian sigmas; a kernel that stops short of them
    cannot gather that light, so bright stars keep a flat halo around a sharpened core.
    """

    r = psf.shape[-1] // 2
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    rr = np.rint(np.hypot(yy, xx)).astype(int)
    counts = np.bincount(rr.ravel(), minlength=r + 1)[: r + 1]
    sums = np.bincount(rr.ravel(), weights=psf.ravel(), minlength=r + 1)[: r + 1]
    profile = sums / np.maximum(counts, 1)
    above = np.nonzero(profile >= floor_frac * float(psf.max()))[0]
    return int(above.max()) if above.size else 0


def crop_kernel(kernel: np.ndarray, size: int) -> np.ndarray:
    """Central ``size`` x ``size`` window of an odd kernel, renormalised to unit sum."""

    m = (kernel.shape[-1] - size) // 2
    if m <= 0:
        return kernel
    out = kernel[..., m:-m, m:-m]
    return (out / max(float(out.sum()), 1e-12)).astype(np.float32)


def gaussian_kernel(size: int, fwhm: float) -> np.ndarray:
    sigma = max(float(fwhm), 0.5) / FWHM_PER_SIGMA
    r = size // 2
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)).astype(np.float32)
    return k / k.sum()


def soften_kernel(kernel: np.ndarray, sigma: float = 0.25) -> np.ndarray:
    out = ndimage.gaussian_filter(kernel.astype(np.float32), sigma, mode="constant")
    out = np.clip(out, 0.0, None)
    return out / max(out.sum(), 1e-12)


def build_star_psf(image2d: np.ndarray, stars: Stars, size: int, max_stars: int = 60, saturation: float | None = None) -> np.ndarray | None:
    """Median-combine recentred cutouts of isolated, unsaturated stars into a unit-sum kernel."""

    if len(stars) < 5:
        return None
    img = np.ascontiguousarray(image2d, dtype=np.float32)
    back, _ = estimate_background(img)
    sub = img - back
    height, width = sub.shape
    r = size // 2
    if saturation is None:
        saturation = float(np.percentile(img, 99.995))
    xy = stars.xy()
    cutouts: list[np.ndarray] = []
    for i in range(len(stars)):
        if len(cutouts) >= max_stars:
            break
        if stars.peak[i] >= saturation:
            continue
        x, y = float(stars.x[i]), float(stars.y[i])
        if not (r + 2 <= x < width - r - 3 and r + 2 <= y < height - r - 3):
            continue
        # a neighbour whose own stamp-radius disc clears this stamp cannot leak into it; stars below
        # 1% of this one's flux are left to the median (requiring 2*size clearance found no stars at large k)
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        d[i] = np.inf
        if np.any((d < size) & (stars.flux > 0.01 * stars.flux[i])):
            continue
        shift = (round(y) - y, round(x) - x)
        cx, cy = int(round(x)), int(round(y))
        patch = sub[cy - r - 1 : cy + r + 2, cx - r - 1 : cx + r + 2]
        patch = ndimage.shift(patch, shift, order=3, mode="nearest")[1:-1, 1:-1]
        total = float(patch.sum())
        if total <= 0:
            continue
        cutouts.append(patch / total)
    if len(cutouts) < 5:
        return None
    psf = np.median(np.stack(cutouts), axis=0)
    psf = np.clip(psf, 0.0, None)
    psf[psf < psf.max() * 1e-4] = 0.0
    return (psf / max(psf.sum(), 1e-12)).astype(np.float32)
