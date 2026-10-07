"""Drizzle integration (Fruchter & Hook 2002) on the GPU, with Bayer-aware mode for OSC data.

Each input pixel is shrunk by ``pixfrac``, mapped through the frame's registration transform
onto an output grid ``scale`` times finer than the reference, and its flux spread over the output
pixels it covers. The square "drop" is approximated by a grid of sub-samples splatted with
``index_add_``; this reproduces the square kernel to within the sub-sample spacing and keeps the
whole thing a handful of GPU scatter operations per frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .backend import Backend, StatusCallback
from .debayer import cfa_masks
from .registration import Alignment, reference_coords
from .stacking import FrameNorm, norm_vec

ProgressCallback = Callable[[float, str], None]


def _noop_progress(_: float, __: str) -> None:
    pass


def _noop_status(_: str) -> None:
    pass


@dataclass
class DrizzleSettings:
    enabled: bool = False
    scale: int = 2
    kernel: Literal["square", "point", "gaussian"] = "square"
    pixfrac: float = 0.8
    cfa: bool = True  # drizzle Bayer colours straight from the CFA (OSC only); otherwise debayered frames
    subsamples: int = 0  # per axis; 0 = auto from pixfrac*scale
    min_weight: float = 0.05  # output pixels with less relative weight are set to 0 (uncovered)


@dataclass
class DrizzleAccumulator:
    num: torch.Tensor  # (C, Ho, Wo) weighted flux
    den: torch.Tensor  # (C, Ho, Wo) weight
    scale: int
    frames: int = 0

    def image(self, min_weight: float) -> np.ndarray:
        img = self.num / self.den.clamp_min(1e-12)
        thr = min_weight * torch.quantile(self.den.flatten()[:: max(1, self.den.numel() // 500000)], 0.5)
        img = torch.where(self.den >= thr, img, torch.zeros_like(img))
        return img.cpu().numpy().astype(np.float32)

    def weight_map(self) -> np.ndarray:
        return self.den.mean(dim=0).cpu().numpy().astype(np.float32)


def new_accumulator(channels: int, height: int, width: int, scale: int, device: torch.device) -> DrizzleAccumulator:
    return DrizzleAccumulator(
        torch.zeros((channels, height * scale, width * scale), dtype=torch.float32, device=device),
        torch.zeros((channels, height * scale, width * scale), dtype=torch.float32, device=device),
        scale,
    )


def _subsample_offsets(settings: DrizzleSettings, device: torch.device) -> torch.Tensor:
    k = settings.subsamples if settings.subsamples > 0 else int(max(2, math.ceil(settings.pixfrac * settings.scale) + 1))
    # centres of k x k cells inside the pixfrac-shrunk drop, in source-pixel units
    edges = (torch.arange(k, device=device, dtype=torch.float32) + 0.5) / k - 0.5
    return edges * settings.pixfrac  # (k,)


def _kernel_samples(settings: DrizzleSettings, device: torch.device) -> list[tuple[float, float, float]]:
    """Return (dy, dx, normalized weight) samples for the selected drop kernel."""

    if settings.kernel == "point":
        return [(0.0, 0.0, 1.0)]

    offsets = _subsample_offsets(settings, device)
    offset_values = offsets.tolist()
    if settings.kernel == "square":
        sample_weight = 1.0 / (len(offset_values) ** 2)
        return [(dy, dx, sample_weight) for dy in offset_values for dx in offset_values]
    if settings.kernel == "gaussian":
        sigma = max(abs(settings.pixfrac) / 4.0, 1e-6)
        axis = torch.exp(-0.5 * (offsets / sigma) ** 2)
        weights = axis[:, None] * axis[None, :]
        weights /= weights.sum().clamp_min(1e-12)
        weight_rows = weights.tolist()
        return [
            (dy, dx, weight_rows[row][col])
            for row, dy in enumerate(offset_values)
            for col, dx in enumerate(offset_values)
        ]
    raise ValueError(f"Unknown drizzle kernel {settings.kernel!r}")


def _norm_map_sample(map_np: np.ndarray, rx: torch.Tensor, ry: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Sample a (C, bh, bw) low-res map at reference coords; returns (C, N)."""

    m = torch.from_numpy(map_np).to(rx.device).unsqueeze(0)
    gx = (2.0 * (rx + 0.5) / width - 1.0).view(1, 1, -1, 1)
    gy = (2.0 * (ry + 0.5) / height - 1.0).view(1, 1, -1, 1)
    grid = torch.cat([gx, gy], dim=-1)  # (1, 1, N, 2) -> output (1, C, 1, N)
    return F.grid_sample(m, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0, :]


def drizzle_frame(acc: DrizzleAccumulator, frame: torch.Tensor, alignment: Alignment, norm: FrameNorm, ref_offset: float | np.ndarray, settings: DrizzleSettings, reject_mask: torch.Tensor | None = None, cfa_pattern: str | None = None, ref_shape: tuple[int, int] | None = None) -> None:
    """Add one calibrated, UNREGISTERED frame to the accumulator.

    frame: (C, H, W) on device. With ``cfa_pattern`` the frame must be (1, H, W) raw CFA and the
    accumulator must have 3 channels: each Bayer site contributes only to its own colour.
    reject_mask: (Hr, Wr) bool in REFERENCE coordinates (from the regular stack), or None.
    norm.offset/scale and ref_offset may be scalars or per-colour (C,) vectors.
    """

    device = frame.device
    channels, height, width = frame.shape
    ref_h, ref_w = ref_shape if ref_shape is not None else (height, width)
    scale = acc.scale
    out_h, out_w = acc.num.shape[1], acc.num.shape[2]
    weight = float(norm.weight)
    samples = _kernel_samples(settings, device)
    n_col = acc.num.shape[0]
    g_off = torch.from_numpy(norm_vec(norm.offset, n_col)).to(device).view(-1, 1)
    g_scale = torch.from_numpy(norm_vec(norm.scale, n_col)).to(device).view(-1, 1)
    g_ref = torch.from_numpy(norm_vec(ref_offset, n_col)).to(device).view(-1, 1)

    if cfa_pattern is not None:
        colour_masks = cfa_masks(cfa_pattern, height, width, device)  # (3, H, W)
        colour_of_pixel = colour_masks.argmax(dim=0)  # 0=R,1=G,2=B
    ys, xs = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32), torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    valid = ~torch.isnan(frame).any(dim=0)
    vals = torch.nan_to_num(frame, nan=0.0).reshape(channels, -1)
    xs_f, ys_f = xs.reshape(-1), ys.reshape(-1)
    valid_f = valid.reshape(-1)
    if cfa_pattern is not None:
        colour_f = colour_of_pixel.reshape(-1)

    # Process one kernel sample at a time to bound memory.
    for dy, dx, sample_weight in samples:
        rx, ry = reference_coords(alignment, xs_f + dx, ys_f + dy)
        ox = torch.floor(rx * scale + 0.5 * (scale - 1) + 0.5).long()  # sub-sample centre -> output pixel
        oy = torch.floor(ry * scale + 0.5 * (scale - 1) + 0.5).long()
        inside = valid_f & (ox >= 0) & (ox < out_w) & (oy >= 0) & (oy < out_h)
        if reject_mask is not None:
            rxi = rx.round().long().clamp(0, ref_w - 1)
            ryi = ry.round().long().clamp(0, ref_h - 1)
            inside &= ~reject_mask[ryi, rxi]
        if not bool(inside.any()):
            continue
        idx = (oy[inside] * out_w + ox[inside])
        v = vals[:, inside]
        # normalisation: local maps if present (sampled at the reference position), else global
        if norm.scale_map is not None and norm.offset_map is not None:
            s = _norm_map_sample(norm.scale_map, rx[inside], ry[inside], ref_h, ref_w)
            o = _norm_map_sample(norm.offset_map, rx[inside], ry[inside], ref_h, ref_w)
            if cfa_pattern is not None:
                c = colour_f[inside]
                s = s[c, torch.arange(c.numel(), device=device)].unsqueeze(0)
                o = o[c, torch.arange(c.numel(), device=device)].unsqueeze(0)
            v = v * s + o
        elif cfa_pattern is not None:
            c = colour_f[inside]
            v = (v - g_off[c, 0]) * g_scale[c, 0] + g_ref[c, 0]
        else:
            v = (v - g_off) * g_scale + g_ref
        w = torch.full((idx.numel(),), weight * sample_weight, device=device)
        if cfa_pattern is not None:
            c = colour_f[inside]
            for ch in range(3):
                sel = c == ch
                if bool(sel.any()):
                    acc.num[ch].view(-1).index_add_(0, idx[sel], v[0, sel] * w[sel])
                    acc.den[ch].view(-1).index_add_(0, idx[sel], w[sel])
        else:
            for ch in range(channels):
                acc.num[ch].view(-1).index_add_(0, idx, v[ch] * w)
                acc.den[ch].view(-1).index_add_(0, idx, w)
    acc.frames += 1


def coverage_from_weight(weight: np.ndarray, per_frame_weight: float) -> np.ndarray:
    """Rough 'frames per pixel' from the accumulated weight map."""

    return (weight / max(per_frame_weight, 1e-12)).astype(np.float32)
