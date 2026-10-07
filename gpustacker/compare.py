"""Benchmark a GPUStacker master against a reference stack of the same field (e.g. PixInsight).

The reference is registered onto the GPUStacker grid, both are background-subtracted, and the two
are compared on identical pixels. Every SNR uses noise measured at the scale of its own signal
(matched-filter for detection, empirical aperture noise for stars, 15 px means for nebula): per-pixel
noise would reward any process that smooths neighbouring pixels (e.g. bilinear demosaicing) and
penalise one that keeps full resolution, without either being deeper.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

from .detection import detect_stars, estimate_background, luma
from .io import load_frame
from .registration import estimate_alignment, refine_alignment, warp_to_reference

StatusCallback = Callable[[str], None]


def _noop(_: str) -> None:
    pass


@dataclass
class CompareResult:
    gpu: str
    reference: str
    alignment_rms_px: float
    common_fraction: float
    stars_gpu: int
    stars_ref: int
    stars_matched: int
    fwhm_gpu: float
    fwhm_ref: float
    faint_peak_snr_gpu: float  # 10th percentile of star peak SNR
    faint_peak_snr_ref: float
    peak_snr_ratio: float  # matched stars, GPU / ref (median)
    peak_snr_ratio_faint: float  # faintest quintile of matched stars
    star_to_nebula_flux_ratio: float  # GPU / ref after calibrating on diffuse pixels; 1 = same balance
    nebula_snr_ratio: dict[str, float] = field(default_factory=dict)  # by reference SNR band
    ref_only_star_median_snr: float = float("nan")  # how bright the stars GPU misses are (in the reference)
    matched_peak_snr_gpu: float = float("nan")  # median peak SNR of matched stars, each side
    matched_peak_snr_ref: float = float("nan")
    matched_peak_snr_faint_gpu: float = float("nan")
    matched_peak_snr_faint_ref: float = float("nan")
    nebula_snr_gpu: dict[str, float] = field(default_factory=dict)  # median smoothed SNR per band, each side
    nebula_snr_ref: dict[str, float] = field(default_factory=dict)
    regional_fwhm_gpu: dict[str, float] = field(default_factory=dict)
    regional_fwhm_ref: dict[str, float] = field(default_factory=dict)
    regional_star_snr_gpu: dict[str, float] = field(default_factory=dict)
    regional_star_snr_ref: dict[str, float] = field(default_factory=dict)
    regional_matched_stars: dict[str, int] = field(default_factory=dict)

    def summary(self) -> list[str]:
        def better(v: float, hi_good: bool = True) -> str:
            if not np.isfinite(v):
                return ""
            return " (+)" if (v > 1.0) == hi_good and abs(v - 1) > 0.02 else (" (-)" if abs(v - 1) > 0.02 else " (=)")

        lines = [
            f"Stars @5σ (matched filter): GPU {self.stars_gpu} vs ref {self.stars_ref} (matched {self.stars_matched}); ref-only stars have median SNR {self.ref_only_star_median_snr:.1f}",
            f"FWHM: GPU {self.fwhm_gpu:.2f} px vs ref {self.fwhm_ref:.2f} px{better(self.fwhm_ref / self.fwhm_gpu)}",
            f"Faint-star aperture SNR (p10): GPU {self.faint_peak_snr_gpu:.1f} vs ref {self.faint_peak_snr_ref:.1f}",
            f"Matched-star aperture SNR GPU/ref: {self.peak_snr_ratio:.2f} (faint quintile {self.peak_snr_ratio_faint:.2f}){better(self.peak_snr_ratio)}",
            f"Star/nebula flux balance GPU/ref: {self.star_to_nebula_flux_ratio:.2f}{better(self.star_to_nebula_flux_ratio)}  (<1 = GPU loses star flux relative to nebula)",
        ]
        for band, ratio in self.nebula_snr_ratio.items():
            lines.append(f"Nebula SNR GPU/ref where ref SNR in {band}: {ratio:.2f}{better(ratio)}")
        for region in sorted(set(self.regional_fwhm_gpu) | set(self.regional_fwhm_ref)):
            gpu, ref = self.regional_fwhm_gpu.get(region), self.regional_fwhm_ref.get(region)
            if gpu and ref:
                lines.append(f"FWHM {region}: GPU {gpu:.2f} vs ref {ref:.2f}px{better(ref / gpu)}")
        for region in sorted(self.regional_matched_stars):
            gpu, ref = self.regional_star_snr_gpu.get(region), self.regional_star_snr_ref.get(region)
            if gpu is not None and ref is not None:
                lines.append(f"Matched-star SNR {region} (n={self.regional_matched_stars[region]}): GPU/ref {gpu / ref:.2f}{better(gpu / ref)}")
        return lines


def _regional_medians(xs: np.ndarray, ys: np.ndarray, values: np.ndarray, shape: tuple[int, int], minimum_count: int = 5) -> tuple[dict[str, float], dict[str, int]]:
    """Median metric on a 4x4 image grid; omit cells with too few finite samples."""

    height, width = shape
    rows = np.clip((np.asarray(ys) * 4 / height).astype(int), 0, 3)
    cols = np.clip((np.asarray(xs) * 4 / width).astype(int), 0, 3)
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(values)
    labels = ("top", "upper-mid", "lower-mid", "bottom")
    columns = ("left", "centre-left", "centre-right", "right")
    medians: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in range(4):
        for col in range(4):
            selected = valid & (rows == row) & (cols == col)
            count = int(selected.sum())
            if count >= minimum_count:
                key = f"{labels[row]} {columns[col]}"
                medians[key] = float(np.median(values[selected]))
                counts[key] = count
    return medians, counts


def _bgsub(image: np.ndarray, box: int = 64) -> tuple[np.ndarray, float]:
    lum = luma(image)
    back, rms = estimate_background(lum, box=box)
    return lum - back, float(rms)


def _scale_noise(sub: np.ndarray, sky: np.ndarray, kernel: np.ndarray) -> float:
    """Robust sigma of ``sub`` convolved with ``kernel`` over sky pixels (large-scale structure removed)."""

    filtered = ndimage.convolve(sub, kernel, mode="nearest") - ndimage.uniform_filter(sub, 65)
    v = filtered[sky]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _aperture_flux(sub: np.ndarray, xs: np.ndarray, ys: np.ndarray, r: float) -> np.ndarray:
    """Background-annulus-corrected circular aperture sums at (xs, ys)."""

    import sep

    data = np.ascontiguousarray(sub, dtype=np.float64)
    flux, _, _ = sep.sum_circle(data, xs.astype(np.float64), ys.astype(np.float64), r, bkgann=(2.0 * r, 3.0 * r), subpix=5)
    return np.asarray(flux)


def compare_stacks(gpu_path: Path, ref_path: Path, status_cb: StatusCallback = _noop, thresh_sigma: float = 5.0) -> CompareResult:
    g, _ = load_frame(gpu_path)
    r, _ = load_frame(ref_path)
    status_cb(f"Comparing {gpu_path.name} {tuple(g.shape)} against {ref_path.name} {tuple(r.shape)}")
    g_sub, g_rms = _bgsub(g)
    r_lum = luma(r)
    r_sub, _ = _bgsub(r)

    gs = detect_stars(g_sub, thresh_sigma=8)
    rs = detect_stars(r_sub, thresh_sigma=8)
    align = refine_alignment(estimate_alignment(rs, gs), rs, gs)
    status_cb(f"Reference -> GPU grid: scale {align.scale:.5f} rot {align.rotation_deg:+.3f}° shift ({align.shift[0]:+.1f}, {align.shift[1]:+.1f}) rms {align.residual_rms:.2f} px")
    if abs(align.scale - 1.0) > 0.05:
        status_cb(f"WARNING: images differ in pixel scale by {align.scale:.2f}x (e.g. drizzle vs non-drizzle) — results are not meaningful; pick a reference at the same scale")
    warped = warp_to_reference(torch.from_numpy(np.ascontiguousarray(r_lum))[None], align, mode="bilinear")[0].numpy()
    h, w = min(warped.shape[0], g_sub.shape[0]), min(warped.shape[1], g_sub.shape[1])
    warped, g_sub = warped[:h, :w], g_sub[:h, :w]
    common = np.isfinite(warped) & np.isfinite(g_sub)
    fill = float(np.nanmedian(warped))
    r_back, r_rms = estimate_background(np.nan_to_num(warped, nan=fill), box=64)
    r_sub = np.where(common, warped - r_back, np.nan)
    g0, r0 = np.nan_to_num(g_sub), np.nan_to_num(r_sub)

    # sky = common pixels well away from anything bright in either image (rough cut, refined below)
    rough = (np.abs(g0) < 3 * g_rms) & (np.abs(r0) < 3 * r_rms)
    sky = common & ndimage.binary_erosion(rough, iterations=3)
    detect_k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16.0  # sep's default filter
    g_det_noise, r_det_noise = _scale_noise(g0, sky, detect_k), _scale_noise(r0, sky, detect_k)
    # matched-filter units: same false-detection rate whatever the pixel correlation
    g_snr = ndimage.convolve(g0, detect_k, mode="nearest") / g_det_noise
    r_snr = ndimage.convolve(r0, detect_k, mode="nearest") / r_det_noise

    def stars(snr: np.ndarray):
        s = detect_stars(snr, thresh_sigma=thresh_sigma, max_sources=50000)
        yi = np.clip(s.y.round().astype(int), 0, h - 1)
        xi = np.clip(s.x.round().astype(int), 0, w - 1)
        keep = common[yi, xi]
        return s.x[keep], s.y[keep], s.flux[keep], s.peak[keep], s.fwhm[keep]

    gx, gy, gf, gp, gw = stars(g_snr)
    rx, ry, rf, rp, rw = stars(r_snr)
    # FWHM from the unfiltered images at the usual 5-sigma per-pixel threshold (comparable to earlier runs)
    g_widths = detect_stars(g0 / g_rms, thresh_sigma=thresh_sigma, max_sources=50000)
    r_widths = detect_stars(r0 / r_rms, thresh_sigma=thresh_sigma, max_sources=50000)
    gw, rw = g_widths.fwhm, r_widths.fwhm
    regional_fwhm_gpu, _ = _regional_medians(g_widths.x, g_widths.y, gw, (h, w))
    regional_fwhm_ref, _ = _regional_medians(r_widths.x, r_widths.y, rw, (h, w))
    tree = cKDTree(np.column_stack([gx, gy]))
    dist, j = tree.query(np.column_stack([rx, ry]), distance_upper_bound=2.5)
    matched = np.isfinite(dist)

    star_mask = np.zeros((h, w), dtype=bool)
    fw_ref = float(np.median(rw)) if len(rw) else 3.0
    for xs, ys in ((gx, gy), (rx, ry)):
        for x, y in zip(xs, ys):
            rad = int(max(4, 3.0 * fw_ref))
            yi, xi = int(round(y)), int(round(x))
            star_mask[max(0, yi - rad) : yi + rad + 1, max(0, xi - rad) : xi + rad + 1] = True
    sky &= ~star_mask
    diffuse = common & ~star_mask

    # star SNR = fixed-aperture flux / empirical aperture noise (apertures dropped on blank sky)
    ap_r = max(2.0, 1.0 * fw_ref)
    rng = np.random.default_rng(0)
    sy, sx = np.nonzero(sky[int(4 * ap_r) : h - int(4 * ap_r), int(4 * ap_r) : w - int(4 * ap_r)])
    pick = rng.choice(len(sx), size=min(4000, len(sx)), replace=False)
    bx, by = sx[pick] + int(4 * ap_r), sy[pick] + int(4 * ap_r)

    def ap_noise(sub):
        f = _aperture_flux(sub, bx, by, ap_r)
        return float(1.4826 * np.median(np.abs(f - np.median(f))))

    g_apn, r_apn = ap_noise(g0), ap_noise(r0)
    mx, my = rx[matched], ry[matched]
    g_apf = _aperture_flux(g0, mx, my, ap_r)
    r_apf = _aperture_flux(r0, mx, my, ap_r)
    gp_m, rp_m = g_apf / g_apn, r_apf / r_apn
    peak_ratio = gp_m / rp_m if matched.any() else np.array([np.nan])
    faint = rp_m < np.percentile(rp_m, 20) if matched.sum() > 10 else np.ones(matched.sum(), bool)
    regional_star_snr_gpu, regional_matched_stars = _regional_medians(rx[matched], ry[matched], gp_m, (h, w))
    regional_star_snr_ref, _ = _regional_medians(rx[matched], ry[matched], rp_m, (h, w))
    # faint-star SNR (p10) over each image's own detections, same aperture metric
    gp = _aperture_flux(g0, gx, gy, ap_r) / g_apn
    rp = _aperture_flux(r0, rx, ry, ap_r) / r_apn

    box = np.ones((15, 15), np.float32) / 225.0
    g_neb_noise, r_neb_noise = _scale_noise(g0, sky, box), _scale_noise(r0, sky, box)
    g_smooth = ndimage.uniform_filter(g0, 15) / g_neb_noise
    r_smooth = ndimage.uniform_filter(r0, 15) / r_neb_noise
    r_select = ndimage.uniform_filter(r0, 45) / r_neb_noise  # band choice independent of the 15px noise being measured
    nebula: dict[str, float] = {}
    nebula_g: dict[str, float] = {}
    nebula_r: dict[str, float] = {}
    for lo, hi in ((2, 5), (5, 15), (15, 50), (50, float("inf"))):
        sel = diffuse & (r_select > lo) & (r_select <= hi)
        if sel.sum() > 1000:
            band = f"({lo}, {hi if np.isfinite(hi) else 'inf'}]"
            nebula[band] = float(np.median(g_smooth[sel] / r_smooth[sel]))
            nebula_g[band] = float(np.median(g_smooth[sel]))
            nebula_r[band] = float(np.median(r_smooth[sel]))
    # balance: matched-star aperture flux ratio over smooth-nebula flux ratio (both in raw image units)
    sel = diffuse & (r_select > 3)
    neb_scale = float(np.median(ndimage.uniform_filter(g0, 15)[sel] / ndimage.uniform_filter(r0, 15)[sel])) if sel.sum() > 1000 else float("nan")
    tot_r = 3.0 * fw_ref  # total-flux aperture: independent of PSF width
    g_tot, r_tot = _aperture_flux(g0, mx, my, tot_r), _aperture_flux(r0, mx, my, tot_r)
    good = (r_apf > 20 * r_apn) & (g_tot > 0) & (r_tot > 0)
    balance = float(np.median(g_tot[good] / r_tot[good]) / neb_scale) if good.sum() > 10 and np.isfinite(neb_scale) else float("nan")

    return CompareResult(
        gpu=str(gpu_path),
        reference=str(ref_path),
        alignment_rms_px=float(align.residual_rms),
        common_fraction=float(common.mean()),
        stars_gpu=int(len(gx)),
        stars_ref=int(len(rx)),
        stars_matched=int(matched.sum()),
        fwhm_gpu=float(np.median(gw)) if len(gw) else float("nan"),
        fwhm_ref=float(np.median(rw)) if len(rw) else float("nan"),
        faint_peak_snr_gpu=float(np.percentile(gp, 10)) if len(gp) else float("nan"),
        faint_peak_snr_ref=float(np.percentile(rp, 10)) if len(rp) else float("nan"),
        peak_snr_ratio=float(np.median(peak_ratio)),
        peak_snr_ratio_faint=float(np.median(peak_ratio[faint])) if matched.any() else float("nan"),
        star_to_nebula_flux_ratio=balance,
        nebula_snr_ratio=nebula,
        ref_only_star_median_snr=float(np.median(rp[~matched])) if (~matched).any() else float("nan"),
        matched_peak_snr_gpu=float(np.median(gp_m)) if matched.any() else float("nan"),
        matched_peak_snr_ref=float(np.median(rp_m)) if matched.any() else float("nan"),
        matched_peak_snr_faint_gpu=float(np.median(gp_m[faint])) if matched.any() else float("nan"),
        matched_peak_snr_faint_ref=float(np.median(rp_m[faint])) if matched.any() else float("nan"),
        nebula_snr_gpu=nebula_g,
        nebula_snr_ref=nebula_r,
        regional_fwhm_gpu=regional_fwhm_gpu,
        regional_fwhm_ref=regional_fwhm_ref,
        regional_star_snr_gpu=regional_star_snr_gpu,
        regional_star_snr_ref=regional_star_snr_ref,
        regional_matched_stars=regional_matched_stars,
    )


def compare_many(gpu_paths: Sequence[Path], ref_path: Path, status_cb: StatusCallback = _noop) -> list[CompareResult]:
    """Compare several candidate masters independently against the same reference."""

    results = []
    for index, gpu_path in enumerate(gpu_paths):
        status_cb(f"Candidate {index + 1}/{len(gpu_paths)}: {gpu_path.name}")
        results.append(compare_stacks(gpu_path, ref_path, lambda message: status_cb(f"[{gpu_path.name}] {message}")))
    return results


def write_report(result: CompareResult | Sequence[CompareResult], path: Path) -> Path:
    payload = asdict(result) if isinstance(result, CompareResult) else {"comparisons": [asdict(item) for item in result]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
