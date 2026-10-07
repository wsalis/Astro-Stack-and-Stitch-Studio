"""Local normalisation: low-resolution per-frame offset maps fitted to a reference.

frame' = scale * frame + offset_map(x, y). The multiplicative ``scale`` is the frame's global
flux scale (from star photometry, see pipeline.register) and is NOT re-fitted per block: a
block's median is dominated by sky + nebula, so fitting scale to it would pin nebulosity to
the reference while leaving stars under-corrected (star/nebula flux imbalance). The offset map
corrects rotating light-pollution gradients (meridian flips), moon-lit halves of a night, and
sky-brightness drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .backend import Backend


@dataclass
class LocalNormSettings:
    enabled: bool = True
    block: int = 128
    smooth_sigma: float = 1.0  # in blocks
    min_valid_fraction: float = 0.3  # blocks with less coverage are filled from neighbours
    scale_limits: tuple[float, float] = (0.5, 2.0)  # used only when no flux scale is supplied (noise-ratio fallback)


def block_stats(image: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-block (median, MAD-sigma, valid fraction) for a (C, H, W) tensor with NaNs. Returns (C, bh, bw) each."""

    channels, height, width = image.shape
    pad_h = (-height) % block
    pad_w = (-width) % block
    x = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), value=float("nan"))[0]
    blocks = x.unfold(1, block, block).unfold(2, block, block)  # (C, bh, bw, block, block)
    bh, bw = blocks.shape[1], blocks.shape[2]
    flat = blocks.reshape(channels, bh, bw, -1)
    valid = ~torch.isnan(flat)
    frac = valid.float().mean(dim=-1)
    med = torch.nanmedian(flat, dim=-1).values
    mad = torch.nanmedian((flat - med.unsqueeze(-1)).abs(), dim=-1).values * 1.4826
    return med, mad, frac


def _fill_and_smooth(arr: np.ndarray, bad: np.ndarray, sigma: float) -> np.ndarray:
    """Fill bad blocks from the nearest good ones, then Gaussian-smooth (per channel)."""

    out = arr.copy()
    for c in range(arr.shape[0]):
        plane = out[c]
        mask = bad[c] | ~np.isfinite(plane)
        if mask.all():
            plane[:] = 0.0 if np.isnan(plane).all() else np.nanmedian(plane)
            continue
        if mask.any():
            idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
            plane[:] = plane[tuple(idx)]
        if sigma > 0:
            plane[:] = ndimage.gaussian_filter(plane, sigma, mode="nearest")
    return out


def fit_norm_maps(frame: torch.Tensor, ref_med: torch.Tensor, ref_mad: torch.Tensor, ref_frac: torch.Tensor, settings: LocalNormSettings, scale: float | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (scale_map, offset_map) as (C, bh, bw) float32 so that frame' matches the reference blocks.

    ``scale``: the frame's global flux scale (scalar or (C,)); the scale map is then constant and only the
    offsets vary. Without it the per-block noise ratio is used (legacy behaviour).
    """

    med, mad, frac = block_stats(frame, settings.block)
    if scale is None:
        lo, hi = settings.scale_limits
        scale_t = (ref_mad / mad.clamp_min(1e-9)).clamp(lo, hi)
    else:
        s = torch.as_tensor(np.asarray(scale, dtype=np.float32).reshape(-1), device=frame.device)
        scale_t = s.view(-1, 1, 1).expand_as(med).clone() if s.numel() > 1 else torch.full_like(med, float(s))
    offset = ref_med - scale_t * med
    bad = ((frac < settings.min_valid_fraction) | (ref_frac < settings.min_valid_fraction) | ~torch.isfinite(scale_t) | ~torch.isfinite(offset)).cpu().numpy()
    scale_np = _fill_and_smooth(scale_t.cpu().numpy().astype(np.float32), bad, settings.smooth_sigma if scale is None else 0.0)
    offset_np = _fill_and_smooth(offset.cpu().numpy().astype(np.float32), bad, settings.smooth_sigma)
    return scale_np, offset_np


def reference_block_stats(reference: np.ndarray, backend: Backend, block: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ref_t = torch.from_numpy(np.ascontiguousarray(reference, dtype=np.float32)).to(backend.device)
    return block_stats(ref_t, block)
