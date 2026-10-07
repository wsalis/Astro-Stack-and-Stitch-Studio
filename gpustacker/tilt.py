"""Sensor tilt / field curvature diagnostics from per-region star shapes (CPU, NumPy).

Everything is measured in sensor coordinates on unregistered frames, so an optical or
mechanical defect shows as a pattern that is the same in every frame, while seeing or focus
drift shows as a pattern that moves.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from .debayer import debayer
from .detection import Stars, detect_stars, luma
from .io import load_frame

StatusCallback = Callable[[str], None]
COMPASS = ("right", "bottom-right", "bottom", "bottom-left", "left", "top-left", "top", "top-right")


@dataclass
class TiltMap:
    path: Path | None
    shape: tuple[int, int]
    cells: int
    fwhm: np.ndarray  # (cells, cells) median FWHM px, NaN where too few stars; row 0 = top of sensor
    elongation: np.ndarray  # median major/minor axis ratio (1.0 = round)
    angle_deg: np.ndarray  # dominant major-axis angle 0..180, from +x anticlockwise in array coords (y down)
    coherence: np.ndarray  # 0..1: how consistently stars in the cell point the same way
    count: np.ndarray
    stars: int
    median_fwhm: float
    tilt_x: float  # FWHM change left -> right across the full width (px); > 0 = right side softer
    tilt_y: float  # FWHM change top -> bottom (px); > 0 = bottom softer
    curvature: float  # corner excess over centre from the symmetric r^2 term (px); > 0 = corners softer

    @property
    def tilt_px(self) -> float:
        return float(math.hypot(self.tilt_x, self.tilt_y))

    @property
    def tilt_angle_deg(self) -> float:
        """Direction of the soft side, degrees from +x (right) towards +y (down)."""

        return float(math.degrees(math.atan2(self.tilt_y, self.tilt_x))) % 360.0

    @property
    def soft_side(self) -> str:
        return COMPASS[int(((self.tilt_angle_deg + 22.5) % 360.0) // 45.0)]

    @property
    def name(self) -> str:
        return self.path.name if self.path is not None else "session median"

    def corners(self) -> dict[str, float]:
        """Corner and centre FWHM (centre = mean of the central cell block for even grids)."""

        f, n = self.fwhm, self.cells
        mid = slice((n - 1) // 2, n // 2 + 1)
        return {
            "TL": float(f[0, 0]),
            "TR": float(f[0, n - 1]),
            "BL": float(f[n - 1, 0]),
            "BR": float(f[n - 1, n - 1]),
            "C": float(np.nanmean(f[mid, mid])) if np.isfinite(f[mid, mid]).any() else float("nan"),
        }


def frame_luma(path: Path | str) -> tuple[np.ndarray, tuple[int, int]]:
    """Luma of a raw light (bilinear debayer on CPU for CFA data; shapes are sensor pixels)."""

    data, meta = load_frame(path)
    arr = np.asarray(data, dtype=np.float32)
    if meta.is_cfa and arr.shape[0] == 1:
        arr = debayer(torch.from_numpy(np.ascontiguousarray(arr)), meta.bayer_pattern or "RGGB", "bilinear").numpy()
    lum = luma(arr)
    return lum, (int(lum.shape[0]), int(lum.shape[1]))


def cell_stats(stars: Stars, shape: tuple[int, int], cells: int = 4, min_stars: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell (fwhm, elongation, angle_deg, coherence, count) in sensor coordinates."""

    fw_g = np.full((cells, cells), np.nan, dtype=np.float64)
    el_g = fw_g.copy()
    an_g = fw_g.copy()
    co_g = fw_g.copy()
    n_g = np.zeros((cells, cells), dtype=np.int64)
    if len(stars) == 0:
        return fw_g, el_g, an_g, co_g, n_g
    height, width = shape
    col = np.clip((stars.x * cells / max(width, 1)).astype(int), 0, cells - 1)
    row = np.clip((stars.y * cells / max(height, 1)).astype(int), 0, cells - 1)
    fwhm = stars.fwhm.astype(np.float64)
    elong = stars.elongation.astype(np.float64)
    theta = None if stars.theta is None else stars.theta.astype(np.float64)
    for r in range(cells):
        for c in range(cells):
            sel = (row == r) & (col == c)
            n = int(sel.sum())
            n_g[r, c] = n
            if n < min_stars:
                continue
            fw_g[r, c] = np.median(fwhm[sel])
            el_g[r, c] = np.median(elong[sel])
            if theta is not None:
                # doubled-angle mean so 0 and 180 degrees agree; round stars get no vote
                w = np.maximum(elong[sel] - 1.0, 0.0)
                if w.sum() > 0:
                    cx = float(np.sum(w * np.cos(2.0 * theta[sel])) / w.sum())
                    cy = float(np.sum(w * np.sin(2.0 * theta[sel])) / w.sum())
                    an_g[r, c] = math.degrees(0.5 * math.atan2(cy, cx)) % 180.0
                    co_g[r, c] = math.hypot(cx, cy)
    return fw_g, el_g, an_g, co_g, n_g


def fit_plane(fwhm_grid: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares FWHM = c0 + tx*x + ty*y + k*r^2 over cell centres with x, y in [-0.5, 0.5].

    Returns (c0, tilt_x, tilt_y, curvature) where tilt_* is the full-width FWHM change and
    curvature is the corner excess over the centre (k * 0.5). NaN when under-determined.
    """

    g = np.asarray(fwhm_grid, dtype=np.float64)
    n = g.shape[0]
    centres = (np.arange(n) + 0.5) / n - 0.5
    yy, xx = np.meshgrid(centres, centres, indexing="ij")
    ok = np.isfinite(g)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    x, y, z = xx[ok], yy[ok], g[ok]
    cols = [np.ones_like(x), x, y]
    if ok.sum() >= 5:
        cols.append(x * x + y * y)
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    curvature = float(coef[3] * 0.5) if len(coef) > 3 else float("nan")
    return float(coef[0]), float(coef[1]), float(coef[2]), curvature


def build_map(stars: Stars, shape: tuple[int, int], cells: int = 4, path: Path | None = None) -> TiltMap:
    if len(stars) > 20:
        # flat-topped saturated cores inflate the second moments; they all share (near) the same peak
        keep = stars.peak < 0.97 * float(stars.peak.max())
        if keep.sum() >= 20:
            stars = Stars(stars.x[keep], stars.y[keep], stars.flux[keep], stars.a[keep], stars.b[keep], stars.peak[keep], stars.total_detected, None if stars.theta is None else stars.theta[keep])
    fw, el, an, co, n = cell_stats(stars, shape, cells)
    _, tx, ty, curv = fit_plane(fw)
    return TiltMap(path, shape, cells, fw, el, an, co, n, len(stars), stars.median_fwhm(), tx, ty, curv)


def analyse_tilt(path: Path | str, cells: int = 4, thresh_sigma: float = 5.0) -> TiltMap:
    lum, shape = frame_luma(path)
    stars = detect_stars(lum, thresh_sigma=thresh_sigma)
    return build_map(stars, shape, cells, Path(path))


def session_median(maps: Sequence[TiltMap]) -> TiltMap | None:
    """Cell-wise median over frames; the part of the pattern every frame shares."""

    maps = [m for m in maps if m is not None]
    if not maps:
        return None
    cells = maps[0].cells
    same = [m for m in maps if m.cells == cells]
    fw = np.nanmedian(np.stack([m.fwhm for m in same]), axis=0)
    el = np.nanmedian(np.stack([m.elongation for m in same]), axis=0)
    # orientation: coherence-weighted doubled-angle mean across frames
    ang = np.stack([np.radians(m.angle_deg) * 2.0 for m in same])
    wgt = np.stack([np.nan_to_num(m.coherence) for m in same])
    cx = np.nansum(wgt * np.cos(ang), axis=0)
    cy = np.nansum(wgt * np.sin(ang), axis=0)
    an = np.where(wgt.sum(axis=0) > 0, np.degrees(0.5 * np.arctan2(cy, cx)) % 180.0, np.nan)
    co = np.where(wgt.sum(axis=0) > 0, np.hypot(cx, cy) / np.maximum(wgt.sum(axis=0), 1e-9), np.nan)
    n = np.stack([m.count for m in same]).sum(axis=0)
    _, tx, ty, curv = fit_plane(fw)
    return TiltMap(None, same[0].shape, cells, fw, el, an, co, n, int(sum(m.stars for m in same)), float(np.nanmedian([m.median_fwhm for m in same])), tx, ty, curv)


def frame_scatter(maps: Sequence[TiltMap]) -> float:
    """Median per-cell std of the frame-normalised FWHM maps (0 = identical pattern in every frame)."""

    same = [m for m in maps if m is not None]
    if len(same) < 3:
        return float("nan")
    g = np.stack([m.fwhm for m in same])
    rel = g / np.nanmedian(g.reshape(len(g), -1), axis=1)[:, None, None]
    return float(np.nanmedian(np.nanstd(rel, axis=0)))


def describe(tm: TiltMap, scatter: float = float("nan")) -> list[str]:
    """Plain-language reading of one map, with hints on what to adjust."""

    lines: list[str] = []
    fw = tm.fwhm
    if not np.isfinite(fw).any():
        return ["No cell had enough stars to measure."]
    best = float(np.nanmin(fw))
    worst = float(np.nanmax(fw))
    ratio = worst / best if best > 0 else float("nan")
    c = tm.corners()
    lines.append(f"{tm.name}: {tm.stars} stars, median FWHM {tm.median_fwhm:.2f} px; cells {best:.2f}-{worst:.2f} px (ratio {ratio:.2f})")
    lines.append(f"Corners TL {c['TL']:.2f}  TR {c['TR']:.2f}  BL {c['BL']:.2f}  BR {c['BR']:.2f}   centre {c['C']:.2f} px")
    if np.isfinite(tm.tilt_px):
        lines.append(f"Tilt plane: {tm.tilt_px:.2f} px across the sensor, soft side {tm.soft_side.upper()} (left-right {tm.tilt_x:+.2f}, top-bottom {tm.tilt_y:+.2f})")
    if np.isfinite(tm.curvature):
        lines.append(f"Curvature: corners {tm.curvature:+.2f} px vs centre (symmetric part, not tilt)")
    el_med = float(np.nanmedian(tm.elongation)) if np.isfinite(tm.elongation).any() else float("nan")
    el_max = float(np.nanmax(tm.elongation)) if np.isfinite(tm.elongation).any() else float("nan")
    if np.isfinite(el_med):
        lines.append(f"Elongation: median {el_med:.3f}, worst cell {el_max:.3f}; orientation coherence {np.nanmedian(tm.coherence):.2f}")
    if np.isfinite(scatter):
        lines.append(f"Frame-to-frame scatter of the pattern: +/-{100 * scatter:.0f}%")

    lines.append("")
    tilt_ok = np.isfinite(tm.tilt_px) and np.isfinite(tm.median_fwhm) and tm.median_fwhm > 0
    tilt_frac = tm.tilt_px / tm.median_fwhm if tilt_ok else 0.0
    curv_frac = tm.curvature / tm.median_fwhm if np.isfinite(tm.curvature) and tm.median_fwhm > 0 else 0.0
    if np.isfinite(scatter) and scatter > 0.5 * max(tilt_frac, 1e-9):
        lines.append("Reading: the pattern moves from frame to frame about as much as the tilt itself - focus/seeing drift dominates; check the trend before shimming anything.")
    if tilt_frac < 0.06:
        lines.append("Reading: tilt is below 6% of the FWHM - not worth chasing mechanically.")
    else:
        lines.append(
            f"Reading: the {tm.soft_side.upper()} side sits out of the focal plane. One frame cannot tell whether it is too far in or out: "
            "rack the focuser slightly inward and re-shoot - if that side sharpens it is too far OUT (shim it toward the mirror), otherwise the reverse."
        )
    if abs(curv_frac) >= 0.06:
        where = "corners softer than centre" if tm.curvature > 0 else "centre softer than corners"
        lines.append(f"Reading: {where} by {abs(tm.curvature):.2f} px uniformly - that is field curvature / coma-corrector spacing, shims will not fix it.")
    if np.isfinite(el_med) and el_max >= 1.15:
        coh = float(np.nanmedian(tm.coherence)) if np.isfinite(tm.coherence).any() else 0.0
        if coh >= 0.5:
            lines.append("Reading: stars in the soft cells are elongated in a consistent direction - astigmatism from tilt (axis flips 90 deg across the sensor), tracking (same axis everywhere), or coma (axes point at the field centre). Compare the ellipse orientations on the map.")
    return lines


def write_tilt_csv(maps: Sequence[TiltMap], path: Path | str) -> Path:
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "stars", "median_fwhm", "tilt_px", "tilt_x", "tilt_y", "soft_side", "curvature", "TL", "TR", "BL", "BR", "C", "fwhm_grid", "elongation_grid"])
        for m in maps:
            c = m.corners()
            writer.writerow(
                [
                    m.name,
                    m.stars,
                    f"{m.median_fwhm:.3f}",
                    f"{m.tilt_px:.3f}",
                    f"{m.tilt_x:.3f}",
                    f"{m.tilt_y:.3f}",
                    m.soft_side,
                    f"{m.curvature:.3f}",
                    *(f"{c[k]:.3f}" for k in ("TL", "TR", "BL", "BR", "C")),
                    " / ".join(" ".join(f"{v:.2f}" for v in row) for row in m.fwhm),
                    " / ".join(" ".join(f"{v:.3f}" for v in row) for row in m.elongation),
                ]
            )
    return path
