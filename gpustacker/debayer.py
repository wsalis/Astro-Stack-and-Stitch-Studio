"""Bilinear CFA demosaicing on the torch device."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_OFFSETS = {
    # (row, col) of R, G1, G2, B inside the 2x2 CFA cell
    "RGGB": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "BGGR": ((1, 1), (0, 1), (1, 0), (0, 0)),
    "GRBG": ((0, 1), (0, 0), (1, 1), (1, 0)),
    "GBRG": ((1, 0), (0, 0), (1, 1), (0, 1)),
}

_K_RB = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 4.0
_K_G = torch.tensor([[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]]) / 4.0


def cfa_masks(pattern: str, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Return (3, H, W) 0/1 masks selecting R, G, B sites."""

    key = pattern.upper()
    if key not in _OFFSETS:
        raise ValueError(f"Unsupported Bayer pattern {pattern!r}")
    r, g1, g2, b = _OFFSETS[key]
    masks = torch.zeros(3, height, width, device=device)
    masks[0, r[0]::2, r[1]::2] = 1.0
    masks[1, g1[0]::2, g1[1]::2] = 1.0
    masks[1, g2[0]::2, g2[1]::2] = 1.0
    masks[2, b[0]::2, b[1]::2] = 1.0
    return masks


def debayer_bilinear(cfa: torch.Tensor, pattern: str) -> torch.Tensor:
    """(1, H, W) CFA -> (3, H, W) RGB via separable-free bilinear interpolation."""

    if cfa.ndim != 3 or cfa.shape[0] != 1:
        raise ValueError("debayer expects a (1, H, W) tensor")
    _, height, width = cfa.shape
    masks = cfa_masks(pattern, height, width, cfa.device)
    sampled = cfa * masks  # (3, H, W)
    kernels = torch.stack([_K_RB, _K_G, _K_RB]).to(cfa.device, cfa.dtype).unsqueeze(1)  # (3,1,3,3)
    padded = F.pad(sampled.unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    rgb = F.conv2d(padded, kernels, groups=3)
    return rgb[0]


# 8 VNG directions as (dy, dx) unit steps: N, E, S, W, NE, SE, SW, NW
_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1))


def debayer_vng(cfa: torch.Tensor, pattern: str) -> torch.Tensor:
    """(1, H, W) CFA -> (3, H, W) RGB via Variable Number of Gradients (Chang, Cheung & Pang 1999).

    For each pixel, 8 directional gradients are measured on the raw mosaic; only directions with
    gradient <= 1.5*min + 0.5*(max - min) are used, and missing colours are filled from colour
    differences along those directions on top of the pixel's own raw value. Unlike bilinear, this
    keeps the full-resolution luminance of every sampled pixel (sharper stars, no blur of R/B).
    """

    if cfa.ndim != 3 or cfa.shape[0] != 1:
        raise ValueError("debayer expects a (1, H, W) tensor")
    _, height, width = cfa.shape
    raw = cfa[0]
    pad = 3
    rp = F.pad(raw.view(1, 1, height, width), (pad, pad, pad, pad), mode="reflect")[0, 0]

    def at(dy: int, dx: int) -> torch.Tensor:
        return rp[pad + dy : pad + dy + height, pad + dx : pad + dx + width]

    grads = []
    for dy, dx in _DIRS:
        if dy == 0 or dx == 0:  # axial: same-colour pairs along the direction plus the two parallel lines
            py, px = dx, dy  # perpendicular unit step
            g = (at(dy, dx) - at(-dy, -dx)).abs() + (at(2 * dy, 2 * dx) - at(0, 0)).abs()
            g = g + 0.5 * ((at(dy + py, dx + px) - at(-dy + py, -dx + px)).abs() + (at(dy - py, dx - px) - at(-dy - py, -dx - px)).abs())
            g = g + 0.5 * ((at(2 * dy + py, 2 * dx + px) - at(py, px)).abs() + (at(2 * dy - py, 2 * dx - px) - at(-py, -px)).abs())
        else:  # diagonal
            g = (at(dy, dx) - at(-dy, -dx)).abs() + (at(2 * dy, 2 * dx) - at(0, 0)).abs()
            g = g + 0.5 * ((at(dy, 0) - at(0, -dx)).abs() + (at(0, dx) - at(-dy, 0)).abs())
            g = g + 0.5 * ((at(2 * dy, dx) - at(dy, 0)).abs() + (at(dy, 2 * dx) - at(0, dx)).abs())
        grads.append(g)
    grads = torch.stack(grads)  # (8, H, W)
    gmin, gmax = grads.min(dim=0).values, grads.max(dim=0).values
    use = (grads <= 1.5 * gmin + 0.5 * (gmax - gmin)).float()  # always includes the minimum

    masks = cfa_masks(pattern, height, width, cfa.device)  # (3, H, W)
    bil = debayer_bilinear(cfa, pattern)  # neighbour colour estimates
    bp = F.pad(bil.unsqueeze(0), (2, 2, 2, 2), mode="reflect")[0]
    diff_sum = torch.zeros((3, height, width), device=cfa.device, dtype=cfa.dtype)
    for i, (dy, dx) in enumerate(_DIRS):
        for step in (1, 2):  # VNG averages each direction over a short run of pixels, not a single one
            sy, sx = 2 + step * dy, 2 + step * dx
            nb = bp[:, sy : sy + height, sx : sx + width]
            nb_centre_colour = (nb * masks).sum(dim=0)  # neighbour's estimate of THIS pixel's colour
            diff_sum += 0.5 * use[i] * (nb - nb_centre_colour)  # colour difference c - own along this direction
    rgb = raw.unsqueeze(0) + diff_sum / use.sum(dim=0)
    return torch.where(masks > 0, raw.unsqueeze(0).expand_as(rgb), rgb)


def debayer(cfa: torch.Tensor, pattern: str, method: str = "vng") -> torch.Tensor:
    if method == "bilinear":
        return debayer_bilinear(cfa, pattern)
    if method == "vng":
        return debayer_vng(cfa, pattern)
    raise ValueError(f"Unknown debayer method {method!r}")
