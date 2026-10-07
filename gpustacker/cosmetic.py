"""Automatic hot/cold pixel cosmetic correction (PixInsight-style auto detect).

Each pixel is compared with the median of its 8 neighbours; pixels deviating by more
than ``sigma`` robust standard deviations are replaced by that median. For CFA data the
neighbourhood is sampled at stride 2 so only same-colour pixels are used. A hot pixel is
an isolated event: it is only replaced when none of its 8 immediate neighbours carries a
comparable excess over the local background, which keeps star cores and wings intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CosmeticSettings:
    hot_sigma: float = 3.0
    cold_sigma: float = 0.0
    fix_hot: bool = True
    fix_cold: bool = False
    # a "hot" pixel is skipped if its brightest immediate neighbour's excess exceeds this fraction of its own
    # excess (plus one sigma); a star peak at FWHM >= ~1.5 px always has such a neighbour, a hot pixel never does
    star_protect_frac: float = 0.3


@dataclass
class CosmeticStats:
    hot: int = 0
    cold: int = 0

    def __iadd__(self, other: "CosmeticStats") -> "CosmeticStats":
        self.hot += other.hot
        self.cold += other.cold
        return self


# Optimal 19-comparator sorting network for 8 inputs (Knuth, TAOCP 5.3.4)
_NETWORK_8 = ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (1, 2), (5, 6), (0, 4), (1, 5), (2, 6), (3, 7), (2, 4), (3, 5), (1, 2), (3, 4), (5, 6))


def _taps(image: torch.Tensor, stride: int) -> list[torch.Tensor]:
    pad = stride
    padded = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")[0]
    _, height, width = image.shape
    taps = []
    for dy in (-stride, 0, stride):
        for dx in (-stride, 0, stride):
            if dy == 0 and dx == 0:
                continue
            taps.append(padded[:, pad + dy : pad + dy + height, pad + dx : pad + dx + width])
    return taps


def neighbour_median(image: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """Lower median of the 8 neighbours at ``stride`` for a (C, H, W) tensor."""

    taps = [t.clone() for t in _taps(image, stride)]
    for a, b in _NETWORK_8:  # min/max network is much cheaper than a sort kernel on 9 MP
        lo = torch.minimum(taps[a], taps[b])
        taps[b] = torch.maximum(taps[a], taps[b])
        taps[a] = lo
    return taps[3]


def neighbour_max(image: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """Maximum of the 8 neighbours at ``stride`` for a (C, H, W) tensor."""

    taps = _taps(image, stride)
    out = taps[0]
    for t in taps[1:]:
        out = torch.maximum(out, t)
    return out


def cosmetic_correct(image: torch.Tensor, settings: CosmeticSettings, cfa: bool = False, sample_stride: int = 8) -> tuple[torch.Tensor, CosmeticStats]:
    """Return (corrected image, counts). Robust sigma comes from the MAD of the local deviation."""

    if not (settings.fix_hot or settings.fix_cold):
        return image, CosmeticStats()
    stride = 2 if cfa else 1
    med = neighbour_median(image, stride=stride)
    dev = image - med
    sample = dev[..., ::sample_stride, ::sample_stride].flatten()
    sample = sample[torch.isfinite(sample)]
    sigma = (torch.median(sample.abs()) * 1.4826).clamp_min(1e-6) if sample.numel() else torch.tensor(1.0, device=image.device)
    if settings.fix_hot:
        # dev is each pixel's excess over its own-colour background, so comparing excesses across
        # immediate (other-colour) neighbours is valid on CFA data too
        neigh = neighbour_max(dev, stride=1)
        hot = (dev > settings.hot_sigma * sigma) & (neigh < settings.star_protect_frac * dev + sigma)
    else:
        hot = torch.zeros_like(dev, dtype=torch.bool)
    cold = (dev < -settings.cold_sigma * sigma) if settings.fix_cold else torch.zeros_like(dev, dtype=torch.bool)
    bad = hot | cold
    out = torch.where(bad, med, image)
    return out, CosmeticStats(int(hot.sum()), int(cold.sum()))
