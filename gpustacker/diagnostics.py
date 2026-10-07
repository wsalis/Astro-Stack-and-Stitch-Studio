"""Stack diagnostics: coverage-based autocrop, noise/SNR estimate, per-frame CSV."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _max_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Largest all-True axis-aligned rectangle (y0, y1, x0, x1) via the histogram/stack method."""

    height, width = mask.shape
    heights = np.zeros(width + 1, dtype=np.int64)  # trailing sentinel stays 0
    best = (0, 0, 0, 0)
    best_area = 0
    for y in range(height):
        heights[:width] = np.where(mask[y], heights[:width] + 1, 0)
        stack: list[int] = []
        for x in range(width + 1):
            while stack and heights[stack[-1]] >= heights[x]:
                top = stack.pop()
                h = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                area = h * (x - left)
                if area > best_area:
                    best_area = area
                    best = (y - h + 1, y + 1, left, x)
            stack.append(x)
    return best


def full_depth_box(coverage: np.ndarray, min_frames: int, coarse: int = 4) -> tuple[int, int, int, int] | None:
    """Largest rectangle where coverage >= min_frames everywhere, as (y0, y1, x0, x1)."""

    mask = coverage >= min_frames
    if not mask.any():
        return None
    height, width = mask.shape
    hc, wc = height // coarse, width // coarse
    if hc < 2 or wc < 2:
        y0, y1, x0, x1 = _max_rectangle(mask)
        return (y0, y1, x0, x1) if y1 > y0 and x1 > x0 else None
    # a coarse cell counts only if every pixel inside it is covered, so the result is conservative
    coarse_mask = mask[: hc * coarse, : wc * coarse].reshape(hc, coarse, wc, coarse).all(axis=(1, 3))
    y0, y1, x0, x1 = _max_rectangle(coarse_mask)
    if y1 <= y0 or x1 <= x0:
        return None
    y0, y1, x0, x1 = y0 * coarse, y1 * coarse, x0 * coarse, x1 * coarse
    # grow each edge at full resolution while the new edge line is fully covered
    while y0 > 0 and mask[y0 - 1, x0:x1].all():
        y0 -= 1
    while y1 < height and mask[y1, x0:x1].all():
        y1 += 1
    while x0 > 0 and mask[y0:y1, x0 - 1].all():
        x0 -= 1
    while x1 < width and mask[y0:y1, x1].all():
        x1 += 1
    return (y0, y1, x0, x1)


def crop_box_for(coverage: np.ndarray, n_frames: int, fraction: float) -> tuple[int, int, int, int] | None:
    """Autocrop box for the given depth fraction (1.0 = every frame must cover the pixel)."""

    if fraction <= 0 or n_frames <= 0:
        return None
    min_frames = max(1, int(math.ceil(fraction * n_frames - 1e-9)))
    return full_depth_box(coverage, min_frames)


def apply_box(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = box
    return np.ascontiguousarray(array[..., y0:y1, x0:x1])


def stack_noise(image: np.ndarray, sample_stride: int = 4) -> float:
    """Robust background noise (MAD sigma after 3-sigma clipping) of a (C, H, W) or (H, W) image."""

    arr = np.asarray(image, dtype=np.float32)
    lum = arr if arr.ndim == 2 else (arr[0] if arr.shape[0] == 1 else 0.2126 * arr[0] + 0.7152 * arr[1] + 0.0722 * arr[2])
    sample = lum[::sample_stride, ::sample_stride].ravel()
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return float("nan")
    for _ in range(3):
        med = np.median(sample)
        sigma = np.median(np.abs(sample - med)) * 1.4826
        keep = np.abs(sample - med) < 3.0 * sigma
        if keep.all() or sigma <= 0:
            break
        sample = sample[keep]
    return float(np.median(np.abs(sample - np.median(sample))) * 1.4826)


def effective_frames(weights: Iterable[float]) -> float:
    w = np.asarray(list(weights), dtype=np.float64)
    if w.size == 0 or not np.isfinite(w).all() or (w**2).sum() == 0:
        return 0.0
    return float(w.sum() ** 2 / (w**2).sum())


def write_frames_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        return path
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return path
