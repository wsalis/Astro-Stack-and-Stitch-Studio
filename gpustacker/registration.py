"""Star-pattern registration: transform estimation (astroalign) + polynomial refinement + GPU warping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .detection import Stars, detect_stars

Interpolation = Literal["lanczos3", "bicubic", "bilinear"]


@dataclass
class Alignment:
    matrix: np.ndarray  # 3x3 similarity, maps source pixel coords -> reference pixel coords
    matched: int
    rotation_deg: float
    scale: float
    shift: tuple[float, float]
    # optional 3rd-order polynomial refinement mapping reference -> source coords (what warping needs)
    inverse_poly: np.ndarray | None = None  # (2, 10): [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]
    forward_poly: np.ndarray | None = None  # same basis, source -> reference (used by drizzle)
    refined_matches: int = 0
    residual_rms: float = float("nan")

    @classmethod
    def identity(cls) -> "Alignment":
        return cls(np.eye(3, dtype=np.float64), 0, 0.0, 1.0, (0.0, 0.0))

    @property
    def flipped(self) -> bool:
        return abs(self.rotation_deg) > 90.0


def estimate_alignment(source_stars: Stars, reference_stars: Stars, max_control_points: int = 60) -> Alignment:
    """Estimate a similarity transform from detected star lists via astroalign."""

    import astroalign

    if len(source_stars) < 3 or len(reference_stars) < 3:
        raise ValueError("Not enough stars to align (need at least 3 in each frame)")
    transform, (src_pts, _) = astroalign.find_transform(
        source_stars.xy()[:max_control_points],
        reference_stars.xy()[:max_control_points],
        max_control_points=max_control_points,
    )
    params = np.asarray(transform.params, dtype=np.float64)
    return Alignment(
        matrix=params,
        matched=int(len(src_pts)),
        rotation_deg=float(np.degrees(transform.rotation)),
        scale=float(transform.scale),
        shift=(float(transform.translation[0]), float(transform.translation[1])),
    )


def _poly_design(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y, x**3, x * x * y, x * y * y, y**3])


def refine_alignment(alignment: Alignment, source_stars: Stars, reference_stars: Stars, max_stars: int = 400, match_radius: float = 2.0, min_matches: int = 60, clip_sigma: float = 3.0) -> Alignment:
    """Fit a 3rd-order polynomial (reference -> source) through all stars matched by the similarity transform.

    Cubic terms capture ordinary radial optical distortion. Falls back to the similarity transform when
    too few stars match. Residuals are iteratively sigma clipped.
    """

    if len(source_stars) < min_matches or len(reference_stars) < min_matches:
        return alignment
    src = source_stars.xy()[:max_stars].astype(np.float64)
    ref = reference_stars.xy()[:max_stars].astype(np.float64)
    src_h = np.column_stack([src, np.ones(len(src))]) @ alignment.matrix.T
    pred = src_h[:, :2] / src_h[:, 2:3]
    d2 = ((pred[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
    j = d2.argmin(1)
    dist = np.sqrt(d2[np.arange(len(pred)), j])
    ok = dist < match_radius
    # one-to-one: keep the closest source star for each reference star
    keep = np.zeros(len(pred), dtype=bool)
    seen: set[int] = set()
    for i in np.argsort(dist):
        if ok[i] and int(j[i]) not in seen:
            seen.add(int(j[i]))
            keep[i] = True
    if keep.sum() < min_matches:
        return alignment
    rx, ry = ref[j[keep], 0], ref[j[keep], 1]
    sx, sy = src[keep, 0], src[keep, 1]
    use = np.ones(int(keep.sum()), dtype=bool)
    coef = np.zeros((2, 10))
    for _ in range(4):
        design = _poly_design(rx[use], ry[use])
        cx = np.linalg.lstsq(design, sx[use], rcond=None)[0]
        cy = np.linalg.lstsq(design, sy[use], rcond=None)[0]
        coef = np.stack([cx, cy])
        full = _poly_design(rx, ry)
        res = np.hypot(full @ cx - sx, full @ cy - sy)
        sigma = 1.4826 * np.median(res[use]) + 1e-6
        new_use = res < clip_sigma * sigma
        if new_use.sum() < min_matches or np.array_equal(new_use, use):
            break
        use = new_use
    full = _poly_design(rx[use], ry[use])
    rms = float(np.sqrt(np.mean((full @ coef[0] - sx[use]) ** 2 + (full @ coef[1] - sy[use]) ** 2)))
    fdesign = _poly_design(sx[use], sy[use])
    forward = np.stack([np.linalg.lstsq(fdesign, rx[use], rcond=None)[0], np.linalg.lstsq(fdesign, ry[use], rcond=None)[0]])
    return Alignment(alignment.matrix, alignment.matched, alignment.rotation_deg, alignment.scale, alignment.shift, coef, forward, int(use.sum()), rms)


def alignment_from_image(source_luma: np.ndarray, reference_stars: Stars, thresh_sigma: float = 5.0) -> tuple[Alignment, Stars]:
    stars = detect_stars(source_luma, thresh_sigma=thresh_sigma)
    return estimate_alignment(stars, reference_stars), stars


def _eval_poly(coef: np.ndarray, xs: torch.Tensor, ys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # evaluate in float64: cubic terms of pixel coordinates overflow float32 precision
    c = torch.from_numpy(coef.astype(np.float64)).to(xs.device)
    xd, yd = xs.double(), ys.double()
    terms = [torch.ones_like(xd), xd, yd, xd * xd, xd * yd, yd * yd, xd**3, xd * xd * yd, xd * yd * yd, yd**3]
    ox = sum(c[0, i] * t for i, t in enumerate(terms))
    oy = sum(c[1, i] * t for i, t in enumerate(terms))
    return ox.float(), oy.float()


def source_coords(alignment: Alignment, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Source-pixel coordinates (sx, sy), each (H, W), for every reference pixel."""

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return source_coords_at(alignment, xs, ys)


def source_coords_at(alignment: Alignment, xs: torch.Tensor, ys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map arbitrary reference-grid coordinates to source-pixel coordinates."""

    if alignment.inverse_poly is not None:
        return _eval_poly(alignment.inverse_poly, xs, ys)
    inv = torch.from_numpy(np.linalg.inv(alignment.matrix)).to(device=xs.device, dtype=torch.float32)
    sx = inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]
    sy = inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]
    return sx, sy


def reference_coords(alignment: Alignment, xs: torch.Tensor, ys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map source-pixel coordinates (any shape) to reference coordinates (forward transform)."""

    if alignment.forward_poly is not None:
        return _eval_poly(alignment.forward_poly, xs, ys)
    m = torch.from_numpy(alignment.matrix).to(device=xs.device, dtype=torch.float64)
    xd, yd = xs.double(), ys.double()
    rx = m[0, 0] * xd + m[0, 1] * yd + m[0, 2]
    ry = m[1, 0] * xd + m[1, 1] * yd + m[1, 2]
    return rx.float(), ry.float()


def union_reference_grid(frames: list[tuple[Alignment, tuple[int, int]]]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (height, width) and (x, y) origin for the union of aligned frame footprints."""

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for alignment, (height, width) in frames:
        edge_x = np.linspace(0, width - 1, 65, dtype=np.float64)
        edge_y = np.linspace(0, height - 1, 65, dtype=np.float64)
        xs = np.concatenate([edge_x, edge_x, np.zeros_like(edge_y), np.full_like(edge_y, width - 1)])
        ys = np.concatenate([np.zeros_like(edge_x), np.full_like(edge_x, height - 1), edge_y, edge_y])
        if alignment.forward_poly is not None:
            mapped = _poly_design(xs, ys) @ alignment.forward_poly.T
            rx, ry = mapped[:, 0], mapped[:, 1]
        else:
            points = np.column_stack([xs, ys, np.ones_like(xs)]) @ alignment.matrix.T
            rx, ry = points[:, 0] / points[:, 2], points[:, 1] / points[:, 2]
        all_x.append(rx)
        all_y.append(ry)
    if not all_x:
        raise ValueError("Cannot build an output grid without frames")
    min_x = int(np.floor(min(float(values.min()) for values in all_x)))
    min_y = int(np.floor(min(float(values.min()) for values in all_y)))
    max_x = int(np.ceil(max(float(values.max()) for values in all_x)))
    max_y = int(np.ceil(max(float(values.max()) for values in all_y)))
    return (max_y - min_y + 1, max_x - min_x + 1), (min_x, min_y)


def reference_overlap_fraction(
    alignment: Alignment,
    source_shape: tuple[int, int],
    reference_shape: tuple[int, int],
    samples: int = 65,
) -> float:
    """Estimate the fraction of a source frame that overlaps the reference footprint."""

    source_height, source_width = source_shape
    reference_height, reference_width = reference_shape
    xs, ys = np.meshgrid(
        np.linspace(0, source_width - 1, samples),
        np.linspace(0, source_height - 1, samples),
    )
    flat_x, flat_y = xs.ravel(), ys.ravel()
    if alignment.forward_poly is not None:
        mapped = _poly_design(flat_x, flat_y) @ alignment.forward_poly.T
        reference_x, reference_y = mapped[:, 0], mapped[:, 1]
    else:
        points = np.column_stack([flat_x, flat_y, np.ones_like(flat_x)]) @ alignment.matrix.T
        reference_x = points[:, 0] / points[:, 2]
        reference_y = points[:, 1] / points[:, 2]
    inside = (
        (reference_x >= 0)
        & (reference_x <= reference_width - 1)
        & (reference_y >= 0)
        & (reference_y <= reference_height - 1)
    )
    return float(inside.mean())


def _lanczos_weights(t: torch.Tensor, a: int = 3) -> torch.Tensor:
    pi_t = torch.pi * t
    core = (a * torch.sin(pi_t) * torch.sin(pi_t / a)) / (pi_t * pi_t + 1e-30)
    out = torch.where(t.abs() < 1e-6, torch.ones_like(t), core)
    return torch.where(t.abs() < a, out, torch.zeros_like(t))


def _warp_lanczos(image: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, a: int = 3, tile_rows: int = 128) -> torch.Tensor:
    """Separable Lanczos-a resampling. Ringing is suppressed only where it actually occurs: an output
    pixel is never allowed below the darkest input sample that contributed to it (dark halos around
    stars), while the sharpening from the negative lobes is otherwise kept intact.

    The output is produced in row tiles so the accumulators and the source rows the 36 taps gather
    from stay cache-resident; the result does not depend on the tiling."""

    channels, height, width = image.shape
    out_h, out_w = sx.shape  # output grid may differ from the source image (mosaic reprojection)
    device = image.device
    offs = torch.arange(-a + 1, a + 1, device=device, dtype=torch.float32).view(-1, 1, 1)  # 2a taps
    flat = image.reshape(channels, -1)
    out = torch.empty((channels, out_h, out_w), device=device, dtype=image.dtype)
    tile_rows = max(1, tile_rows)
    for r0 in range(0, out_h, tile_rows):
        r1 = min(out_h, r0 + tile_rows)
        tsx, tsy = sx[r0:r1], sy[r0:r1]
        finite = torch.isfinite(tsx) & torch.isfinite(tsy)
        tsx = torch.nan_to_num(tsx, nan=0.0, posinf=0.0, neginf=0.0)
        tsy = torch.nan_to_num(tsy, nan=0.0, posinf=0.0, neginf=0.0)
        rows = r1 - r0
        x0 = torch.floor(tsx)
        y0 = torch.floor(tsy)
        wx = _lanczos_weights((tsx - x0).unsqueeze(0) - offs)  # (2a, rows, Wo)
        wy = _lanczos_weights((tsy - y0).unsqueeze(0) - offs)
        xi_all = (x0.unsqueeze(0) + offs).clamp_(0, width - 1).to(torch.int32)
        row_all = (y0.unsqueeze(0) + offs).clamp_(0, height - 1).to(torch.int32).mul_(width)
        acc = torch.zeros((channels, rows, out_w), device=device, dtype=image.dtype)
        floor = torch.full((channels, rows, out_w), float("inf"), device=device, dtype=image.dtype)
        xi_taps, wx_taps = xi_all.unbind(0), wx.unsqueeze(1).unbind(0)
        for row, wy_row in zip(row_all.unbind(0), wy.unsqueeze(1).unbind(0)):
            for xi, wx_tap in zip(xi_taps, wx_taps):
                vals = flat.index_select(1, (row + xi).view(-1)).view(channels, rows, out_w)
                acc.addcmul_(vals, wy_row * wx_tap)
                torch.minimum(floor, vals, out=floor)
        wsum = wy.sum(dim=0) * wx.sum(dim=0)  # sum over the 2a x 2a taps factorises
        acc.div_(wsum.clamp_min_(1e-6))
        out[:, r0:r1] = torch.where(finite.unsqueeze(0), torch.maximum(acc, floor), float("nan"))
    return out


def warp_to_reference(
    image: torch.Tensor,
    alignment: Alignment,
    mode: str = "lanczos3",
    output_shape: tuple[int, int] | None = None,
    origin: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Warp a (C, H, W) tensor onto a reference-aligned grid; uncovered pixels become NaN."""

    channels, height, width = image.shape
    out_height, out_width = output_shape or (height, width)
    if alignment.inverse_poly is None and np.allclose(alignment.matrix, np.eye(3)) and origin == (0, 0) and (out_height, out_width) == (height, width):
        return image
    ys, xs = torch.meshgrid(
        torch.arange(out_height, device=image.device, dtype=torch.float32) + origin[1],
        torch.arange(out_width, device=image.device, dtype=torch.float32) + origin[0],
        indexing="ij",
    )
    sx, sy = source_coords_at(alignment, xs, ys)
    inside = (sx >= 0) & (sx <= width - 1) & (sy >= 0) & (sy <= height - 1)
    if mode == "lanczos3":
        warped = _warp_lanczos(image, sx, sy, 3)
    else:
        gx = 2.0 * sx / max(width - 1, 1) - 1.0
        gy = 2.0 * sy / max(height - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        warped = F.grid_sample(image.unsqueeze(0), grid, mode=mode, padding_mode="border", align_corners=True)[0]
    return torch.where(inside.unsqueeze(0), warped.clamp_min(0.0), torch.full_like(warped, float("nan")))
