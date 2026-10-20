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
_VNG_PAD = 3  # furthest raw sample a VNG gradient reaches
_VNG_HALO = 8  # extra rows per tile so tile seams are invisible (must be even to keep CFA parity)


def _vng_gradients(raw: torch.Tensor) -> torch.Tensor:
    """(h, w) raw -> (8, h, w) VNG gradients, built from six shared |difference| planes."""

    height, width = raw.shape
    pad = _VNG_PAD
    rp = F.pad(raw.view(1, 1, height, width), (pad, pad, pad, pad), mode="reflect")[0, 0]
    hp, wp = rp.shape
    planes: dict[tuple[int, int], tuple[torch.Tensor, int]] = {}

    def absdiff(a: tuple[int, int], b: tuple[int, int]) -> torch.Tensor:
        """|at(a) - at(b)| as a view into one shared |rp(y, x) - rp(y + oy, x + ox)| plane."""

        oy, ox = b[0] - a[0], b[1] - a[1]
        if oy < 0 or (oy == 0 and ox < 0):
            a, (oy, ox) = b, (-oy, -ox)
        if (oy, ox) not in planes:
            lox = max(0, -ox)
            hix = wp - max(0, ox)
            planes[(oy, ox)] = ((rp[: hp - oy, lox:hix] - rp[oy:, lox + ox : hix + ox]).abs_(), lox)
        plane, lox = planes[(oy, ox)]
        y0, x0 = pad + a[0], pad + a[1] - lox
        return plane[y0 : y0 + height, x0 : x0 + width]

    grads = torch.empty((8, height, width), device=raw.device, dtype=raw.dtype)
    for i, (dy, dx) in enumerate(_DIRS):
        g = grads[i]
        if dy == 0 or dx == 0:  # axial: same-colour pairs along the direction plus the two parallel lines
            py, px = dx, dy  # perpendicular unit step
            side = (
                absdiff((dy + py, dx + px), (-dy + py, -dx + px)),
                absdiff((dy - py, dx - px), (-dy - py, -dx - px)),
                absdiff((2 * dy + py, 2 * dx + px), (py, px)),
                absdiff((2 * dy - py, 2 * dx - px), (-py, -px)),
            )
        else:  # diagonal
            side = (
                absdiff((dy, 0), (0, -dx)),
                absdiff((0, dx), (-dy, 0)),
                absdiff((2 * dy, dx), (dy, 0)),
                absdiff((dy, 2 * dx), (0, dx)),
            )
        torch.add(side[0], side[1], out=g)
        g.add_(side[2]).add_(side[3]).mul_(0.5)
        g.add_(absdiff((dy, dx), (-dy, -dx))).add_(absdiff((2 * dy, 2 * dx), (0, 0)))
    return grads


def _vng_core(raw: torch.Tensor, pattern: str) -> torch.Tensor:
    height, width = raw.shape
    grads = _vng_gradients(raw)
    gmin, gmax = torch.aminmax(grads, dim=0)
    use = (grads <= 1.5 * gmin + 0.5 * (gmax - gmin)).to(raw.dtype)  # always includes the minimum
    del grads

    masks = cfa_masks(pattern, height, width, raw.device)  # (3, H, W)
    bil = debayer_bilinear(raw.unsqueeze(0), pattern)  # neighbour colour estimates
    bp = F.pad(bil.unsqueeze(0), (2, 2, 2, 2), mode="reflect")[0]
    hp, wp = bp.shape[1:]
    # colour-difference images on the padded grid: bil[c] minus bil[colour of the CFA site (a, b) away],
    # so a neighbour's "estimate of the centre pixel's colour" is a plain shifted view
    big = cfa_masks(pattern, hp + 2, wp + 2, raw.device).argmax(dim=0)
    cd = {}
    for a in (0, 1):
        for b in (0, 1):
            own = bp.gather(0, big[2 - a : 2 - a + hp, 2 - b : 2 - b + wp].unsqueeze(0))
            cd[(a, b)] = bp - own
    del big
    diff_sum = torch.zeros((3, height, width), device=raw.device, dtype=raw.dtype)
    for i, (dy, dx) in enumerate(_DIRS):
        # VNG averages each direction over a short run of pixels (steps 1 and 2), not a single one
        near = cd[(abs(dy), abs(dx))][:, 2 + dy : 2 + dy + height, 2 + dx : 2 + dx + width]
        far = cd[(0, 0)][:, 2 + 2 * dy : 2 + 2 * dy + height, 2 + 2 * dx : 2 + 2 * dx + width]
        diff_sum.addcmul_(use[i].unsqueeze(0), near + far, value=0.5)
    rgb = raw.unsqueeze(0) + diff_sum / use.sum(dim=0)
    return torch.where(masks > 0, raw.unsqueeze(0).expand_as(rgb), rgb)


def debayer_vng(cfa: torch.Tensor, pattern: str, tile_rows: int = 256) -> torch.Tensor:
    """(1, H, W) CFA -> (3, H, W) RGB via Variable Number of Gradients (Chang, Cheung & Pang 1999).

    For each pixel, 8 directional gradients are measured on the raw mosaic; only directions with
    gradient <= 1.5*min + 0.5*(max - min) are used, and missing colours are filled from colour
    differences along those directions on top of the pixel's own raw value. Unlike bilinear, this
    keeps the full-resolution luminance of every sampled pixel (sharper stars, no blur of R/B).

    Large frames are processed in overlapping row tiles so the many elementwise passes stay in
    the device's L2 cache; the result is independent of the tiling.
    """

    if cfa.ndim != 3 or cfa.shape[0] != 1:
        raise ValueError("debayer expects a (1, H, W) tensor")
    _, height, width = cfa.shape
    raw = cfa[0]
    tile_rows = max(2, tile_rows // 2 * 2)
    if height <= tile_rows + 2 * _VNG_HALO:
        return _vng_core(raw, pattern)
    out = torch.empty((3, height, width), device=cfa.device, dtype=cfa.dtype)
    for y0 in range(0, height, tile_rows):
        y1 = min(height, y0 + tile_rows)
        ys, ye = max(0, y0 - _VNG_HALO), min(height, y1 + _VNG_HALO)
        out[:, y0:y1] = _vng_core(raw[ys:ye], pattern)[:, y0 - ys : y1 - ys]
    return out


def debayer(cfa: torch.Tensor, pattern: str, method: str = "vng") -> torch.Tensor:
    if method == "bilinear":
        return debayer_bilinear(cfa, pattern)
    if method == "vng":
        return debayer_vng(cfa, pattern)
    raise ValueError(f"Unknown debayer method {method!r}")
