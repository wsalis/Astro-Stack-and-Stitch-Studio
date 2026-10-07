"""Frame quality: aperture photometry on matched stars, transparency, PSF-signal weights, filters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .detection import Stars

Weighting = str  # "psfsw" | "psfsw+fwhm" | "noise" | "fwhm" | "both" | "none"
WEIGHTING_CHOICES = ("psfsw", "psfsw+fwhm", "psfsw+field", "noise", "fwhm", "both", "none")


def field_weight_factors(fwhm_grids: list[np.ndarray], minimum: float = 0.5, maximum: float = 2.0, confidence_counts: list[np.ndarray] | None = None, confidence_prior: float = 16.0) -> list[np.ndarray]:
    """Return local factors, shrinking sparsely sampled cells toward neutral weight 1."""

    if not fwhm_grids:
        return []
    grids = np.asarray(fwhm_grids, dtype=np.float64)
    valid = np.isfinite(grids) & (grids > 0)
    values = np.where(valid, grids, np.nan)
    frame_medians = np.full(len(grids), np.nan)
    for index, frame in enumerate(values):
        finite = frame[np.isfinite(frame)]
        if finite.size:
            frame_medians[index] = np.median(finite)
    relative = values / frame_medians[:, None, None]
    session_pattern = np.full(grids.shape[1:], np.nan)
    for row in range(session_pattern.shape[0]):
        for col in range(session_pattern.shape[1]):
            finite = relative[:, row, col]
            finite = finite[np.isfinite(finite)]
            if finite.size:
                session_pattern[row, col] = np.median(finite)
    ratio = relative / session_pattern[None, :, :]
    factors = np.clip(1.0 / np.square(ratio), minimum, maximum)
    factors[~np.isfinite(factors)] = 1.0
    if confidence_counts is not None:
        counts = np.maximum(np.asarray(confidence_counts, dtype=np.float64), 0.0)
        confidence = counts / (counts + max(confidence_prior, 0.0))
        factors = np.exp(np.log(factors) * confidence)
    return [factor.astype(np.float32) for factor in factors]


def sample_field_grid(grid: np.ndarray, xs: np.ndarray, ys: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly sample a coarse sensor-cell map at sensor pixel coordinates."""

    height, width = shape
    rows, cols = grid.shape
    gx = np.clip((xs + 0.5) * cols / max(width, 1) - 0.5, 0, cols - 1)
    gy = np.clip((ys + 0.5) * rows / max(height, 1) - 0.5, 0, rows - 1)
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    x1, y1 = np.minimum(x0 + 1, cols - 1), np.minimum(y0 + 1, rows - 1)
    tx, ty = gx - x0, gy - y0
    numerator = np.zeros_like(gx, dtype=np.float64)
    denominator = np.zeros_like(gx, dtype=np.float64)
    for row, col, weight in (
        (y0, x0, (1 - ty) * (1 - tx)),
        (y0, x1, (1 - ty) * tx),
        (y1, x0, ty * (1 - tx)),
        (y1, x1, ty * tx),
    ):
        values = grid[row, col]
        valid = np.isfinite(values)
        numerator += np.where(valid, values, 0.0) * weight
        denominator += valid * weight
    return np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 0).astype(np.float32)


@dataclass
class FilterSettings:
    """Frame exclusion thresholds; 0 disables a filter."""

    transparency_min: float = 0.6  # star flux vs reference
    stars_min_ratio: float = 0.6  # detected stars vs session median
    background_max_ratio: float = 2.0  # sky level vs session median
    fwhm_max_ratio: float = 1.4  # FWHM vs session median
    enabled: bool = True


@dataclass
class StarSet:
    """Reference stars used for photometry on every registered frame (reference pixel coords)."""

    x: np.ndarray
    y: np.ndarray
    radius: int  # aperture radius [px]
    ref_flux: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.x.size)


def select_photometry_stars(stars: Stars, shape: tuple[int, int], fwhm: float, max_stars: int = 250, saturation: float | None = None) -> StarSet:
    """Bright, isolated, unsaturated stars away from the edges; aperture radius from FWHM."""

    height, width = shape
    radius = int(np.clip(round(2.5 * (fwhm if np.isfinite(fwhm) and fwhm > 0 else 3.0)), 4, 12))
    margin = 3 * radius + 2
    if len(stars) == 0:
        return StarSet(np.empty(0, np.float32), np.empty(0, np.float32), radius)
    if saturation is None:
        # a saturation plateau: several of the brightest stars share (almost) the same peak
        peak_max = float(stars.peak.max())
        plateau = int((stars.peak >= 0.98 * peak_max).sum())
        saturation = 0.98 * peak_max if plateau >= 3 and len(stars) > 10 else float("inf")
    xy = stars.xy()
    keep: list[int] = []
    for i in range(len(stars)):
        if len(keep) >= max_stars:
            break
        x, y = float(stars.x[i]), float(stars.y[i])
        if stars.peak[i] >= saturation or not (margin <= x < width - margin and margin <= y < height - margin):
            continue
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        d[i] = np.inf
        if d.min() < 2.0 * radius:
            continue
        keep.append(i)
    idx = np.asarray(keep, dtype=np.int64)
    return StarSet(stars.x[idx].astype(np.float32), stars.y[idx].astype(np.float32), radius)


def aperture_flux(luma: torch.Tensor, stars: StarSet) -> np.ndarray:
    """Background-subtracted aperture flux per star on a (H, W) device tensor (NaN = no data)."""

    if len(stars) == 0:
        return np.empty(0, dtype=np.float32)
    device = luma.device
    r = stars.radius
    r_out = 3 * r
    k = 2 * r_out + 1
    cx = torch.from_numpy(np.round(stars.x).astype(np.int64)).to(device)
    cy = torch.from_numpy(np.round(stars.y).astype(np.int64)).to(device)
    off = torch.arange(-r_out, r_out + 1, device=device)
    yy = (cy.view(-1, 1, 1) + off.view(1, k, 1)).clamp(0, luma.shape[0] - 1)
    xx = (cx.view(-1, 1, 1) + off.view(1, 1, k)).clamp(0, luma.shape[1] - 1)
    cut = luma[yy, xx]  # (S, k, k)
    dist = torch.sqrt((off.view(k, 1).float() ** 2 + off.view(1, k).float() ** 2))
    ap = (dist <= r).unsqueeze(0)
    ann = ((dist > 1.5 * r) & (dist <= r_out)).unsqueeze(0)
    ann_vals = torch.where(ann.expand_as(cut), cut, torch.full_like(cut, float("nan")))
    sky = torch.nanmedian(ann_vals.reshape(cut.shape[0], -1), dim=1).values
    ap_vals = torch.where(ap.expand_as(cut), cut - sky.view(-1, 1, 1), torch.zeros_like(cut))
    has_nan = torch.isnan(torch.where(ap.expand_as(cut), cut, torch.zeros_like(cut))).flatten(1).any(dim=1) | torch.isnan(sky)
    flux = ap_vals.nansum(dim=(1, 2))
    flux = torch.where(has_nan, torch.full_like(flux, float("nan")), flux)
    return flux.cpu().numpy().astype(np.float32)


def transparency(flux: np.ndarray, ref_flux: np.ndarray, min_stars: int = 12) -> tuple[float, int]:
    """Median flux ratio to the reference over stars valid in both; (ratio, n_used). NaN if too few."""

    ok = np.isfinite(flux) & np.isfinite(ref_flux) & (ref_flux > 0) & (flux > 0)
    n = int(ok.sum())
    if n < min_stars:
        return float("nan"), n
    return float(np.median(flux[ok] / ref_flux[ok])), n


def compute_weights(mode: str, noise: np.ndarray, fwhm: np.ndarray, transp: np.ndarray) -> np.ndarray:
    """Per-frame weights normalised to a median of 1 and clipped to [0.05, 20].

    psfsw: (transparency / noise)^2 — PSF-signal-weight style; frames with unknown transparency fall
    back to the noise-only term.
    """

    noise = np.asarray(noise, dtype=np.float64)
    fwhm = np.asarray(fwhm, dtype=np.float64)
    transp = np.asarray(transp, dtype=np.float64)
    n = noise.size
    if n == 0:
        return np.empty(0)
    if mode == "none":
        return np.ones(n)
    ref_noise = np.nanmedian(noise)
    ref_fwhm = np.nanmedian(fwhm) if np.isfinite(fwhm).any() else np.nan
    t = np.where(np.isfinite(transp), transp, 1.0)
    w = np.ones(n)
    if mode in ("psfsw", "psfsw+fwhm", "psfsw+field"):
        w *= (t * ref_noise / np.maximum(noise, 1e-9)) ** 2
    if mode in ("noise", "both"):
        w *= (ref_noise / np.maximum(noise, 1e-9)) ** 2
    if mode in ("fwhm", "both", "psfsw+fwhm") and np.isfinite(ref_fwhm):
        f = np.where(np.isfinite(fwhm), fwhm, ref_fwhm)
        w *= (ref_fwhm / np.maximum(f, 0.5)) ** 2
    med = np.nanmedian(w)
    if np.isfinite(med) and med > 0:
        w = w / med
    return np.clip(np.nan_to_num(w, nan=1.0), 0.05, 20.0)
