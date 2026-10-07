"""Mosaic assembly from plate-solved master tiles.

Per tile: a low-order polynomial sky-gradient model is removed, then the tile is reprojected
through its WCS (SIP distortion included) onto a shared tangent-plane canvas with Lanczos-3.
Star positions in the overlaps are compared and small per-tile shifts are solved so the
astrometric solutions agree to sub-pixel accuracy. Tiles are then matched photometrically in the
overlaps (gain, offset and a residual plane per tile, solved jointly against the first tile) and
blended in two bands: the background over a wide feather, stars and fine detail over a few-pixel
seam so slightly misregistered stars are never doubled.
"""

from __future__ import annotations

import math
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .backend import Backend, select_backend, to_numpy, to_tensor
from .calibration import ADU_FULL_SCALE
from .detection import detect_stars, luma
from .diagnostics import largest_valid_box
from .io import WCS_KEY, load_frame, save_fits
from .platesolve import find_astap, solve_fits
from .registration import _warp_lanczos

StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[float, str], None]

ORIENTATIONS = ("first", "north")
INTERPOLATIONS = ("lanczos3", "bicubic", "bilinear")
BLEND_MODES = ("seam", "feather")
MAX_CANVAS_PIXELS = 1_600_000_000  # 40k x 40k


def _noop_status(_: str) -> None:
    pass


def _noop_progress(_: float, __: str) -> None:
    pass


def _fmt(value: float, signed: bool = False) -> str:
    """Readable number for ADU (hundreds) and normalised [0, 1] data alike."""

    if not np.isfinite(value):
        return "nan"
    mag = abs(value)
    digits = 2 if mag >= 10 or mag == 0 else max(2, 3 - int(math.floor(math.log10(mag))))
    return f"{value:{'+' if signed else ''}.{min(digits, 7)}f}"


class MosaicCancelled(Exception):
    pass


@dataclass
class MosaicSettings:
    tiles: list[Path]
    output: Path
    gradient_degree: int = 2  # polynomial sky model removed per tile; 0 = off
    gradient_block: int = 64  # px, block size for the sky samples
    pixel_scale: float = 0.0  # output arcsec/px; 0 = finest tile
    orientation: str = "first"  # "first" keeps the first tile's rotation, "north" = north up
    interpolation: str = "lanczos3"
    refine: bool = True  # per-tile shift from matched stars in the overlaps
    photometric: bool = True  # gain/offset matching in the overlaps
    match_gradient: bool = True  # also solve a residual plane per tile
    feather: float = 100.0  # px, background blend width
    seam_width: float = 4.0  # px, star / detail blend width
    blend_mode: str = "feather"  # "seam" = background feathered, detail from one tile; "feather" = plain feather
    auto_crop: bool = True  # crop to the largest rectangle with any tile coverage
    plate_solve: bool = True  # solve tiles without a WCS using ASTAP
    astap_path: Path | None = None
    save_coverage: bool = True
    device: str = "auto"
    vram_fraction: float = 0.75


@dataclass
class Tile:
    path: Path
    image: np.ndarray  # (C, H, W) float32, NaN where there is no data
    header: dict[str, Any]
    wcs: Any  # astropy.wcs.WCS
    scale: float  # arcsec / px
    gradient_pp: np.ndarray | None = None  # (C,) peak-to-peak of the removed gradient
    bbox: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 on the canvas (exclusive)
    warped: np.ndarray | None = None  # (C, h, w) inside bbox, NaN where invalid
    valid: np.ndarray | None = None  # (h, w) bool
    shift: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))  # canvas px
    gain: np.ndarray | None = None  # (C,)
    offset: np.ndarray | None = None  # (C,)
    plane: np.ndarray | None = None  # (C, 2) per normalised canvas x, y
    weight: np.ndarray | None = None  # (h, w) wide feather weight
    narrow: np.ndarray | None = None  # (h, w) seam weight

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[2]


@dataclass
class OverlapStats:
    i: int
    j: int
    pixels: int
    stars: int
    shift: tuple[float, float]  # median star offset tile j - tile i before refinement [px]
    rms: float  # star position scatter after refinement [px]
    residual_before: float  # median |difference| in overlap blocks before photometric match
    residual_after: float


@dataclass
class MosaicResult:
    output: Path
    coverage_output: Path | None
    shape: tuple[int, int]  # (H, W)
    pixel_scale: float
    tiles: list[Tile]
    overlaps: list[OverlapStats]
    seconds: float = 0.0


# ---------------------------------------------------------------------------- WCS helpers


def _coerce(value: Any) -> Any:
    """XISF FITS keywords arrive as strings; wcslib needs real numbers."""

    if isinstance(value, str):
        text = value.strip()
        if text in ("T", "F"):
            return text == "T"
        try:
            return int(text) if text.lstrip("+-").isdigit() else float(text)
        except ValueError:
            return text.strip("'")
    return value


def fits_header(header: dict[str, Any], shape: tuple[int, int] | None = None):
    from astropy.io import fits

    out = fits.Header()
    if shape is not None:
        out["NAXIS"] = 2
        out["NAXIS1"] = int(shape[1])
        out["NAXIS2"] = int(shape[0])
    for key, value in header.items():
        if key in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "COMMENT", "HISTORY", ""):
            continue
        try:
            out[key] = _coerce(value)
        except (ValueError, TypeError):
            continue
    return out


def wcs_from_header(header: dict[str, Any], shape: tuple[int, int]):
    """astropy WCS with SIP for a header, or None when it carries no celestial solution."""

    import warnings

    from astropy.wcs import WCS, FITSFixedWarning

    if not any(str(header.get(f"CTYPE{i}", "")).startswith(("RA", "DEC")) for i in (1, 2)):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs = WCS(fits_header(header, shape), relax=True)
    except Exception:
        return None
    if not wcs.has_celestial:
        return None
    wcs = wcs.celestial
    wcs.pixel_shape = (int(shape[1]), int(shape[0]))
    return wcs


def wcs_pixel_scale(wcs) -> float:
    from astropy.wcs.utils import proj_plane_pixel_scales

    return float(np.mean(np.abs(proj_plane_pixel_scales(wcs)))) * 3600.0


def wcs_header_items(wcs) -> list[tuple[str, Any, str]]:
    items = []
    for card in wcs.to_header(relax=True).cards:
        if WCS_KEY.match(card.keyword) or card.keyword in ("RADESYS", "EQUINOX"):
            items.append((card.keyword, card.value, card.comment))
    items.append(("PLTSOLVD", True, "WCS from mosaic projection"))
    return items


def _edge_points(shape: tuple[int, int], per_edge: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """0-based pixel coordinates sampled along the image border (corners included)."""

    h, w = shape
    t = np.linspace(0.0, 1.0, per_edge)
    xs = np.concatenate([t * (w - 1), np.full(per_edge, w - 1.0), t * (w - 1), np.zeros(per_edge)])
    ys = np.concatenate([np.zeros(per_edge), t * (h - 1), np.full(per_edge, h - 1.0), t * (h - 1)])
    return xs, ys


# ---------------------------------------------------------------------------- gradient model


def _poly_terms(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones_like(x)]
    for total in range(1, degree + 1):
        for px in range(total, -1, -1):
            cols.append(x**px * y ** (total - px))
    return np.stack(cols, axis=-1)


def block_medians(image: np.ndarray, block: int, min_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(C, bh, bw) nan-median per block and (bh, bw) valid fraction (channel 0)."""

    c, h, w = image.shape
    bh, bw = math.ceil(h / block), math.ceil(w / block)
    padded = np.full((c, bh * block, bw * block), np.nan, dtype=np.float32)
    padded[:, :h, :w] = image
    blocks = padded.reshape(c, bh, block, bw, block).transpose(0, 1, 3, 2, 4).reshape(c, bh, bw, block * block)
    frac = np.isfinite(blocks[0]).mean(axis=-1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks are expected off the tile
        med = np.nanmedian(blocks, axis=-1)
    med[:, frac < min_fraction] = np.nan
    return med.astype(np.float32), frac.astype(np.float32)


def fit_gradient(image: np.ndarray, degree: int, block: int) -> np.ndarray:
    """Per-channel polynomial sky model (C, H, W) fitted to block medians with asymmetric clipping,
    so nebulosity (bright blocks) drops out and the fit follows the darkest, sky-dominated blocks."""

    c, h, w = image.shape
    med, _ = block_medians(image, block)
    bh, bw = med.shape[1:]
    by, bx = np.mgrid[0:bh, 0:bw]
    cx = (bx + 0.5) * block / w * 2 - 1
    cy = (by + 0.5) * block / h * 2 - 1
    terms = _poly_terms(cx.ravel(), cy.ravel(), degree)
    yy, xx = np.mgrid[0:h, 0:w]
    full_terms = _poly_terms((xx.ravel() + 0.5) / w * 2 - 1, (yy.ravel() + 0.5) / h * 2 - 1, degree).astype(np.float32)
    model = np.zeros((c, h, w), dtype=np.float32)
    for ch in range(c):
        values = med[ch].ravel()
        keep = np.isfinite(values)
        if keep.sum() < 3 * terms.shape[1]:
            continue
        coef = np.zeros(terms.shape[1])
        for _ in range(12):
            coef, *_ = np.linalg.lstsq(terms[keep], values[keep], rcond=None)
            resid = values - terms @ coef
            sigma = 1.4826 * np.nanmedian(np.abs(resid[keep] - np.nanmedian(resid[keep]))) + 1e-9
            new_keep = np.isfinite(values) & (resid < 1.0 * sigma) & (resid > -3.0 * sigma)
            if new_keep.sum() < 3 * terms.shape[1] or np.array_equal(new_keep, keep):
                break
            keep = new_keep
        model[ch] = (full_terms @ coef.astype(np.float32)).reshape(h, w)
    return model


def correct_gradient(image: np.ndarray, degree: int, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (image with the gradient removed, per-channel peak-to-peak of the removed model).
    The model's median level is kept so the sky stays at its original ADU."""

    if degree <= 0:
        return image, np.zeros(image.shape[0], dtype=np.float32)
    model = fit_gradient(image, degree, block)
    valid = np.isfinite(image[0])
    out = image.copy()
    pp = np.zeros(image.shape[0], dtype=np.float32)
    for ch in range(image.shape[0]):
        m = model[ch][valid]
        if m.size == 0:
            continue
        level = float(np.median(m))
        out[ch] = image[ch] - model[ch] + level
        pp[ch] = float(m.max() - m.min())
    return out, pp


# ---------------------------------------------------------------------------- GPU helpers


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (C, H, W); large sigmas run on a down-sampled copy."""

    if sigma <= 0:
        return x
    c, h, w = x.shape
    factor = max(1, int(sigma // 4))
    if factor > 1:
        small = F.avg_pool2d(x.unsqueeze(0), factor, ceil_mode=True)[0]
        small = _gaussian_blur(small, sigma / factor)
        return F.interpolate(small.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0]
    radius = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    pad = (radius, radius)
    y = x.unsqueeze(1)  # (C, 1, H, W)
    y = F.pad(y, (radius, radius, 0, 0), mode="reflect" if w > radius else "replicate")
    y = F.conv2d(y, k.view(1, 1, 1, -1))
    y = F.pad(y, (0, 0) + pad, mode="reflect" if h > radius else "replicate")
    y = F.conv2d(y, k.view(1, 1, -1, 1))
    return y[:, 0]


def _masked_blur(x: torch.Tensor, mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Normalised convolution: blur ignoring pixels where mask == 0."""

    num = _gaussian_blur(x * mask, sigma)
    den = _gaussian_blur(mask.expand_as(x), sigma)
    return num / den.clamp_min(1e-3)


def _blur_canvas(image: np.ndarray, mask: np.ndarray, sigma: float, backend: Backend) -> np.ndarray:
    """Masked Gaussian blur of a (C, H, W) host array, run on the device in overlapping row bands."""

    c, h, w = image.shape
    out = np.empty_like(image)
    margin = int(math.ceil(4 * sigma))
    m = mask.astype(np.float32)
    rows = int(max(2 * margin + 64, min(h, backend.budget_bytes // 4 // max(c * w * 4 * 10, 1))))
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        e0, e1 = max(0, r0 - margin), min(h, r1 + margin)
        x = to_tensor(image[:, e0:e1], backend)
        mk = to_tensor(m[e0:e1], backend)
        y = _masked_blur(x, mk, sigma)
        out[:, r0:r1] = to_numpy(y[:, r0 - e0 : r1 - e0])
        del x, mk, y
    backend.empty_cache()
    return out


# ---------------------------------------------------------------------------- builder


class MosaicBuilder:
    def __init__(self, settings: MosaicSettings, status_cb: StatusCallback = _noop_status, progress_cb: ProgressCallback = _noop_progress) -> None:
        if settings.orientation not in ORIENTATIONS:
            raise ValueError(f"orientation must be one of {ORIENTATIONS}")
        if settings.interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {INTERPOLATIONS}")
        if settings.blend_mode not in BLEND_MODES:
            raise ValueError(f"blend_mode must be one of {BLEND_MODES}")
        if len(settings.tiles) < 2:
            raise ValueError("A mosaic needs at least two tiles")
        self.settings = settings
        self.status = status_cb
        self.progress = progress_cb
        self._cancel = False
        self.backend: Backend | None = None
        self.out_wcs = None
        self.canvas: tuple[int, int] = (0, 0)
        self.out_scale = 0.0
        self._unit_range = False

    def cancel(self) -> None:
        self._cancel = True

    def _check(self) -> None:
        if self._cancel:
            raise MosaicCancelled("Cancelled")

    # ------------------------------------------------------------------ run

    def run(self) -> MosaicResult:
        t0 = time.perf_counter()
        s = self.settings
        self.backend = select_backend(s.device, s.vram_fraction, self.status)
        tiles = self.load_tiles()
        self._check()
        self.progress(0.15, "Removing gradients")
        self.remove_gradients(tiles)
        self._check()
        self.progress(0.25, "Planning canvas")
        self.plan_canvas(tiles)
        self.progress(0.3, "Reprojecting tiles")
        self.warp_all(tiles)
        self._check()
        overlaps: list[OverlapStats] = []
        if s.refine:
            self.progress(0.55, "Refining registration")
            overlaps = self.refine_registration(tiles)
            self._check()
        if s.photometric:
            self.progress(0.65, "Matching photometry")
            overlaps = self.match_photometry(tiles, overlaps)
            self._check()
        else:
            for t in tiles:
                c = t.image.shape[0]
                t.gain, t.offset, t.plane = np.ones(c), np.zeros(c), np.zeros((c, 2))
        self.progress(0.75, "Blending")
        image, coverage = self.blend(tiles)
        self._check()
        if s.auto_crop:
            image, coverage, crop_box = self.crop_to_coverage(image, coverage)
        else:
            crop_box = None
        self.progress(0.95, "Saving")
        out, cov_path = self.save(tiles, image, coverage, crop_box)
        seconds = time.perf_counter() - t0
        self.progress(1.0, "Done")
        self.status(f"Mosaic {self.canvas[1]} x {self.canvas[0]} px at {self.out_scale:.3f}\"/px -> {out} ({seconds:.1f}s)")
        return MosaicResult(out, cov_path, self.canvas, self.out_scale, tiles, overlaps, seconds)

    def crop_to_coverage(self, image: np.ndarray, coverage: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int] | None]:
        """Crop image and coverage to the largest rectangle without uncovered pixels."""

        box = largest_valid_box(coverage > 0)
        if box is None:
            self.status("Autocrop: no covered rectangle found; output left uncropped")
            return image, coverage, None
        y0, y1, x0, x1 = box
        if (y0, y1, x0, x1) == (0, coverage.shape[0], 0, coverage.shape[1]):
            self.status("Autocrop: no black borders to remove")
            return image, coverage, None
        old_h, old_w = coverage.shape
        image = np.ascontiguousarray(image[:, y0:y1, x0:x1])
        coverage = np.ascontiguousarray(coverage[y0:y1, x0:x1])
        self.out_wcs.wcs.crpix -= np.array([x0, y0], dtype=np.float64)
        self.canvas = (y1 - y0, x1 - x0)
        self.out_wcs.pixel_shape = (self.canvas[1], self.canvas[0])
        self.status(f"Autocrop: largest fully covered rectangle {x1 - x0}x{y1 - y0} px; trimmed left {x0} right {old_w - x1} top {y0} bottom {old_h - y1}")
        return image, coverage, box

    # ------------------------------------------------------------------ loading

    def load_tiles(self) -> list[Tile]:
        s = self.settings
        tiles: list[Tile] = []
        exe = None
        for i, path in enumerate(s.tiles):
            self._check()
            self.progress(0.12 * i / len(s.tiles), f"Loading {path.name}")
            data, meta = load_frame(path)
            image = data.copy()
            coverage_path = path.with_name(f"{path.stem}_coverage.fit")
            coverage_map = None
            if coverage_path.is_file():
                coverage_data, coverage_meta = load_frame(coverage_path)
                if coverage_meta.shape == meta.shape and coverage_meta.channels == 1:
                    coverage_map = coverage_data[0] > 0
                    self.status(f"{path.name}: using coverage map {coverage_path.name}")
                else:
                    self.status(f"{path.name}: ignoring mismatched coverage map {coverage_path.name}")
            invalid = ~np.isfinite(image).all(axis=0)
            if coverage_map is not None:
                invalid |= ~coverage_map
            else:
                invalid |= (image == 0).all(axis=0)
            image[:, invalid] = np.nan
            # PixInsight masters are normalised [0, 1]; GPUStacker masters are ADU. Bring every tile to
            # tile 1's convention so one gain solve covers the lot (as calibration does for masters).
            unit = bool(np.nanmedian(image[0, ::8, ::8]) < 1.0)
            if i == 0:
                self._unit_range = unit
            elif unit != self._unit_range:
                image *= ADU_FULL_SCALE if unit else 1.0 / ADU_FULL_SCALE
                self.status(f"{path.name}: {'[0, 1] normalised' if unit else 'ADU'} data rescaled to match tile 1")
            wcs = wcs_from_header(meta.header, meta.shape)
            header = dict(meta.header)
            stale = wcs is not None and "GPUSTACK" in header and not header.get("PLTSOLVD")
            if stale:
                # a GPUStacker master written before outputs were solved: its WCS is the uncropped sub's
                self.status(f"{path.name}: WCS was inherited from a sub, not solved on the master" + ("; re-solving" if s.plate_solve else " (enable plate solving to fix)"))
                if s.plate_solve:
                    wcs = None
            if wcs is None and s.plate_solve:
                if exe is None:
                    exe = find_astap(s.astap_path)
                    if exe is None:
                        raise RuntimeError(f"{path.name} has no WCS and ASTAP was not found (install from hnsky.org with a D50/D80 database)")
                if not stale:
                    self.status(f"{path.name}: no WCS in header, plate solving with ASTAP...")
                solved = self._solve_tile(data, header, exe)
                if solved is None:
                    raise RuntimeError(f"ASTAP found no solution for {path.name}")
                for key in [k for k in header if WCS_KEY.match(k)]:
                    del header[key]
                header.update(solved)
                wcs = wcs_from_header(header, meta.shape)
            if wcs is None:
                raise RuntimeError(f"{path.name} has no WCS; plate solve it first (GPUStacker does this when ASTAP is installed)")
            scale = wcs_pixel_scale(wcs)
            tiles.append(Tile(path, image, header, wcs, scale))
            ra, dec = (float(v) for v in wcs.wcs.crval)
            self.status(f"Tile {i + 1}/{len(s.tiles)} {path.name}: {meta.shape[1]}x{meta.shape[0]} {'RGB' if meta.channels == 3 else 'mono'}, {scale:.3f}\"/px, centre RA {ra:.4f} Dec {dec:+.4f}, {100 * (~invalid).mean():.0f}% valid")
        channels = {t.image.shape[0] for t in tiles}
        if len(channels) > 1:
            raise RuntimeError("All tiles must have the same channel count (mix of mono and RGB)")
        return tiles

    @staticmethod
    def _solve_tile(image: np.ndarray, header: dict[str, Any], exe: Path) -> dict[str, Any] | None:
        from astropy.io import fits

        hints = {k: _coerce(v) for k, v in header.items() if k in ("RA", "DEC", "OBJCTRA", "OBJCTDEC", "XPIXSZ", "FOCALLEN", "DRIZSCL")}
        lum = luma(image)
        lum = np.where(np.isfinite(lum), lum, float(np.nanmedian(lum[::8, ::8]))).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix="gpustacker_mosaic_") as tmp:
            tmp_fits = save_fits(Path(tmp) / "tile.fit", lum, base_header=hints)
            if solve_fits(tmp_fits, exe) is None:
                return None
            solved = fits.getheader(tmp_fits)
        return {k: solved[k] for k in solved if WCS_KEY.match(k) or k in ("RADESYS", "EQUINOX")}

    # ------------------------------------------------------------------ gradients

    def remove_gradients(self, tiles: list[Tile]) -> None:
        s = self.settings
        if s.gradient_degree <= 0:
            self.status("Gradient correction: off")
            return
        for i, t in enumerate(tiles):
            self._check()
            self.progress(0.15 + 0.1 * i / len(tiles), f"Gradient model {t.path.name}")
            t.image, t.gradient_pp = correct_gradient(t.image, s.gradient_degree, s.gradient_block)
            sky = float(np.nanmedian(t.image[0, ::8, ::8]))
            pp = ", ".join(_fmt(v) for v in t.gradient_pp)
            self.status(f"Gradient removed from {t.path.name}: degree {s.gradient_degree}, peak-to-peak [{pp}] ({100 * float(t.gradient_pp.max()) / max(sky, 1e-12):.1f}% of sky)")

    # ------------------------------------------------------------------ canvas

    def plan_canvas(self, tiles: list[Tile]) -> None:
        from astropy.wcs import WCS

        s = self.settings
        # centre: mean direction of all tile border points
        vecs = []
        for t in tiles:
            xs, ys = _edge_points(t.shape)
            ra, dec = t.wcs.all_pix2world(xs, ys, 0)
            ra_r, dec_r = np.radians(ra), np.radians(dec)
            vecs.append(np.stack([np.cos(dec_r) * np.cos(ra_r), np.cos(dec_r) * np.sin(ra_r), np.sin(dec_r)], axis=-1))
        mean = np.concatenate(vecs).mean(axis=0)
        mean /= np.linalg.norm(mean)
        ra0 = math.degrees(math.atan2(mean[1], mean[0])) % 360.0
        dec0 = math.degrees(math.asin(np.clip(mean[2], -1, 1)))

        self.out_scale = s.pixel_scale if s.pixel_scale > 0 else min(t.scale for t in tiles)
        deg = self.out_scale / 3600.0
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.cunit = ["deg", "deg"]
        wcs.wcs.radesys = "ICRS"
        if s.orientation == "first":
            # same tangent point, rotation and reference pixel as tile 0 -> its grid lines up with the
            # canvas (an integer shift when the scale matches), so it is copied rather than resampled
            cd = tiles[0].wcs.pixel_scale_matrix
            cd = cd / math.sqrt(abs(np.linalg.det(cd))) * deg
            ra0, dec0 = (float(v) for v in tiles[0].wcs.wcs.crval)
            crpix = [float(v) for v in tiles[0].wcs.wcs.crpix]
        else:
            cd = np.array([[-deg, 0.0], [0.0, deg]])
            crpix = [1.0, 1.0]
        wcs.wcs.crval = [ra0, dec0]
        wcs.wcs.crpix = crpix
        wcs.wcs.cd = cd

        xs_all, ys_all = [], []
        for t in tiles:
            xs, ys = _edge_points(t.shape)
            ra, dec = t.wcs.all_pix2world(xs, ys, 0)
            px, py = wcs.all_world2pix(ra, dec, 0)
            xs_all.append(px)
            ys_all.append(py)
        px = np.concatenate(xs_all)
        py = np.concatenate(ys_all)
        x_min, x_max = math.floor(np.nanmin(px)) - 1, math.ceil(np.nanmax(px)) + 1
        y_min, y_max = math.floor(np.nanmin(py)) - 1, math.ceil(np.nanmax(py)) + 1
        width, height = x_max - x_min + 1, y_max - y_min + 1
        if width * height > MAX_CANVAS_PIXELS:
            raise RuntimeError(f"Mosaic canvas {width} x {height} is too large; check the tiles' plate solutions")
        wcs.wcs.crpix = [crpix[0] - x_min, crpix[1] - y_min]
        wcs.pixel_shape = (width, height)
        self.out_wcs = wcs
        self.canvas = (height, width)
        rot = math.degrees(math.atan2(cd[0, 1], cd[1, 1])) % 360.0
        self.status(f"Canvas {width} x {height} px, {self.out_scale:.3f}\"/px, centre RA {ra0:.4f} Dec {dec0:+.4f}, rotation {rot:.1f} deg ({s.orientation})")

    def _tile_bbox(self, t: Tile) -> tuple[int, int, int, int]:
        xs, ys = _edge_points(t.shape)
        ra, dec = t.wcs.all_pix2world(xs, ys, 0)
        px, py = self.out_wcs.all_world2pix(ra, dec, 0)
        px, py = px + t.shift[0], py + t.shift[1]
        height, width = self.canvas
        x0 = max(0, math.floor(np.nanmin(px)) - 2)
        y0 = max(0, math.floor(np.nanmin(py)) - 2)
        x1 = min(width, math.ceil(np.nanmax(px)) + 3)
        y1 = min(height, math.ceil(np.nanmax(py)) + 3)
        return x0, y0, x1, y1

    def _source_map(self, t: Tile, bbox: tuple[int, int, int, int], step: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """(sy, sx) tile pixel coordinates for every canvas pixel in bbox, from a coarse WCS grid."""

        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        nx, ny = math.ceil((w - 1) / step) + 1, math.ceil((h - 1) / step) + 1
        gx = np.linspace(x0, x1 - 1, nx) - t.shift[0]
        gy = np.linspace(y0, y1 - 1, ny) - t.shift[1]
        X, Y = np.meshgrid(gx, gy)
        ra, dec = self.out_wcs.all_pix2world(X.ravel(), Y.ravel(), 0)
        with np.errstate(all="ignore"):
            try:
                sx, sy = t.wcs.all_world2pix(ra, dec, 0, quiet=True, maxiter=30, tolerance=1e-4)
            except Exception:
                sx, sy = t.wcs.wcs_world2pix(ra, dec, 0)
        sx = np.where(np.isfinite(sx), sx, -1e6).reshape(ny, nx)
        sy = np.where(np.isfinite(sy), sy, -1e6).reshape(ny, nx)
        coarse = torch.from_numpy(np.stack([sx, sy]).astype(np.float32)).to(self.backend.device)
        fine = F.interpolate(coarse.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=True)[0]
        return fine[0], fine[1]

    def _warp(self, image: torch.Tensor, mask: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channels, height, width = image.shape
        inside = (sx >= 0) & (sx <= width - 1) & (sy >= 0) & (sy <= height - 1)
        gx = 2.0 * sx / max(width - 1, 1) - 1.0
        gy = 2.0 * sy / max(height - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        if self.settings.interpolation == "lanczos3":
            warped = _warp_lanczos(image, sx.clamp(-4, width + 3), sy.clamp(-4, height + 3), 3)
        else:
            warped = F.grid_sample(image.unsqueeze(0), grid, mode=self.settings.interpolation, padding_mode="border", align_corners=True)[0]
        m = F.grid_sample(mask.view(1, 1, height, width), grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        # Lanczos taps reach 3 px: only trust pixels whose whole footprint was valid data
        m = -F.max_pool2d(-m, 7, stride=1, padding=3)[0]
        valid = inside & (m > 0.999)
        return warped, valid

    def warp_tile(self, t: Tile) -> None:
        backend = self.backend
        t.bbox = bbox = self._tile_bbox(t)
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        c = t.image.shape[0]
        fill = np.nanmedian(t.image[:, ::8, ::8].reshape(c, -1), axis=1)
        img_np = np.where(np.isfinite(t.image), t.image, fill[:, None, None]).astype(np.float32)
        image = to_tensor(img_np, backend)
        mask = to_tensor(np.isfinite(t.image[0]).astype(np.float32), backend)
        area = (self.out_scale / t.scale) ** 2
        sx_all, sy_all = self._source_map(t, bbox)
        warped = np.full((c, h, w), np.nan, dtype=np.float32)
        valid = np.zeros((h, w), dtype=bool)
        per_row = c * w * 4 * 14 + w * 4 * 6
        rows = int(max(32, min(h, backend.budget_bytes // 3 // max(per_row, 1))))
        for r0 in range(0, h, rows):
            r1 = min(h, r0 + rows)
            out, ok = self._warp(image, mask, sx_all[r0:r1], sy_all[r0:r1])
            out = torch.where(ok.unsqueeze(0), out * area, torch.full_like(out, float("nan")))
            warped[:, r0:r1] = to_numpy(out)
            valid[r0:r1] = to_numpy(ok)
        t.warped, t.valid = warped, valid
        del image, mask, sx_all, sy_all
        backend.empty_cache()

    def warp_all(self, tiles: list[Tile]) -> None:
        for i, t in enumerate(tiles):
            self._check()
            self.progress(0.3 + 0.25 * i / len(tiles), f"Reprojecting {t.path.name}")
            t0 = time.perf_counter()
            self.warp_tile(t)
            x0, y0, x1, y1 = t.bbox
            self.status(f"Reprojected {t.path.name} -> canvas [{x0}:{x1}, {y0}:{y1}] ({t.valid.sum() / 1e6:.1f} Mpx valid, {time.perf_counter() - t0:.1f}s)")

    # ------------------------------------------------------------------ overlaps

    @staticmethod
    def _intersection(a: Tile, b: Tile) -> tuple[int, int, int, int] | None:
        x0, y0 = max(a.bbox[0], b.bbox[0]), max(a.bbox[1], b.bbox[1])
        x1, y1 = min(a.bbox[2], b.bbox[2]), min(a.bbox[3], b.bbox[3])
        return (x0, y0, x1, y1) if x1 - x0 > 8 and y1 - y0 > 8 else None

    @staticmethod
    def _crop(t: Tile, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        x0, y0, x1, y1 = box
        sl = (slice(y0 - t.bbox[1], y1 - t.bbox[1]), slice(x0 - t.bbox[0], x1 - t.bbox[0]))
        return t.warped[(slice(None),) + sl], t.valid[sl]

    def overlap_pairs(self, tiles: list[Tile], min_pixels: int = 2000) -> list[tuple[int, int, tuple[int, int, int, int], np.ndarray]]:
        pairs = []
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                box = self._intersection(tiles[i], tiles[j])
                if box is None:
                    continue
                both = self._crop(tiles[i], box)[1] & self._crop(tiles[j], box)[1]
                if both.sum() >= min_pixels:
                    pairs.append((i, j, box, both))
        return pairs

    # ------------------------------------------------------------------ registration refinement

    @staticmethod
    def _stars_in(t: Tile, box: tuple[int, int, int, int], both: np.ndarray) -> np.ndarray:
        img, _ = MosaicBuilder._crop(t, box)
        lum = luma(img)
        fill = float(np.nanmedian(lum[both])) if both.any() else 0.0
        lum = np.where(both & np.isfinite(lum), lum, fill).astype(np.float32)
        stars = detect_stars(lum, thresh_sigma=8.0, max_sources=1500)
        if len(stars) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        keep = (stars.peak < 0.9 * np.nanmax(lum)) & (stars.elongation < 1.6)
        return np.column_stack([stars.x[keep] + box[0], stars.y[keep] + box[1], stars.flux[keep]])

    @staticmethod
    def _match(a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
        """Mutual nearest-neighbour matches; returns (N, 2) displacement b - a."""

        from scipy.spatial import cKDTree

        if len(a) == 0 or len(b) == 0:
            return np.zeros((0, 2))
        ta, tb = cKDTree(a[:, :2]), cKDTree(b[:, :2])
        d_ab, i_ab = tb.query(a[:, :2], distance_upper_bound=radius)
        d_ba, i_ba = ta.query(b[:, :2], distance_upper_bound=radius)
        ok = np.isfinite(d_ab)
        idx_a = np.nonzero(ok)[0]
        idx_b = i_ab[ok]
        mutual = i_ba[idx_b] == idx_a
        return b[idx_b[mutual], :2] - a[idx_a[mutual], :2]

    def refine_registration(self, tiles: list[Tile]) -> list[OverlapStats]:
        pairs = self.overlap_pairs(tiles)
        if not pairs:
            self.status("Registration check: no overlapping tiles found")
            return []
        measured = []
        stats: list[OverlapStats] = []
        for i, j, box, both in pairs:
            self._check()
            a = self._stars_in(tiles[i], box, both)
            b = self._stars_in(tiles[j], box, both)
            disp = self._match(a, b, radius=4.0)
            if len(disp) >= 5:
                med = np.median(disp, axis=0)
                resid = disp - med
                keep = np.hypot(resid[:, 0], resid[:, 1]) < max(1.0, 3 * 1.4826 * np.median(np.hypot(resid[:, 0], resid[:, 1])))
                disp = disp[keep]
                med = np.median(disp, axis=0)
                rms = float(np.sqrt(np.mean(np.sum((disp - med) ** 2, axis=1))))
                measured.append((i, j, med, len(disp)))
            else:
                med, rms = np.zeros(2), float("nan")
            stats.append(OverlapStats(i, j, int(both.sum()), len(disp), (float(med[0]), float(med[1])), rms, float("nan"), float("nan")))
            self.status(f"Overlap {tiles[i].name} / {tiles[j].name}: {int(both.sum()) / 1e6:.2f} Mpx, {len(disp)} matched stars, offset ({med[0]:+.2f}, {med[1]:+.2f}) px, scatter {rms:.2f} px")
        if not measured:
            self.status("Registration refinement skipped: too few matched stars in the overlaps")
            return stats
        # per-tile shift d with d_i - d_j = (pos_j - pos_i) median, reference tile 0 fixed
        n = len(tiles)
        rows, rhs, wts = [], [], []
        for i, j, med, count in measured:
            row = np.zeros(n)
            row[i], row[j] = 1.0, -1.0
            rows.append(row)
            rhs.append(med)
            wts.append(math.sqrt(count))
        A = np.array(rows)[:, 1:] * np.array(wts)[:, None]
        B = np.array(rhs) * np.array(wts)[:, None]
        sol, *_ = np.linalg.lstsq(A, B, rcond=None)
        shifts = np.vstack([np.zeros((1, 2)), sol])
        largest = float(np.max(np.hypot(shifts[:, 0], shifts[:, 1])))
        if largest < 0.05:
            self.status("Registration refinement: plate solutions already agree (< 0.05 px)")
            return stats
        if largest > 50:
            self.status(f"Registration refinement skipped: implausible shift {largest:.1f} px (check the tiles' plate solutions)")
            return stats
        for t, d in zip(tiles, shifts):
            t.shift = d.astype(np.float64)
        self.status("Registration refinement: shifts " + ", ".join(f"{t.name} ({d[0]:+.2f}, {d[1]:+.2f})" for t, d in zip(tiles, shifts)))
        for k, t in enumerate(tiles[1:], start=1):
            if np.hypot(*t.shift) >= 0.02:
                self._check()
                self.progress(0.55 + 0.1 * k / n, f"Re-projecting {t.path.name}")
                self.warp_tile(t)
        # residual scatter after the shift
        for st in stats:
            i, j = st.i, st.j
            box = self._intersection(tiles[i], tiles[j])
            if box is None:
                continue
            both = self._crop(tiles[i], box)[1] & self._crop(tiles[j], box)[1]
            disp = self._match(self._stars_in(tiles[i], box, both), self._stars_in(tiles[j], box, both), radius=3.0)
            if len(disp) >= 5:
                med = np.median(disp, axis=0)
                st.rms = float(np.sqrt(np.mean(np.sum((disp - med) ** 2, axis=1))))
                st.stars = len(disp)
                self.status(f"  after refinement {tiles[i].name} / {tiles[j].name}: residual ({med[0]:+.2f}, {med[1]:+.2f}) px, scatter {st.rms:.2f} px")
        poor = [st for st in stats if np.isfinite(st.rms) and st.rms > 1.0 and st.stars >= 20]
        if poor:
            names = sorted({tiles[k].name for st in poor for k in (st.i, st.j)})
            self.status(f"Warning: star scatter above 1 px in {len(poor)} overlap(s) ({', '.join(names)}) - a plate solution disagrees with its neighbours; stars near those seams may be soft")
        return stats

    # ------------------------------------------------------------------ photometry

    def match_photometry(self, tiles: list[Tile], stats: list[OverlapStats], block: int = 32) -> list[OverlapStats]:
        s = self.settings
        n, c = len(tiles), tiles[0].image.shape[0]
        height, width = self.canvas
        pairs = self.overlap_pairs(tiles)
        by_pair = {(st.i, st.j): st for st in stats}
        samples = []  # (i, j, a (C,K), b (C,K), X (K,), Y (K,))
        for i, j, box, both in pairs:
            ia, _ = self._crop(tiles[i], box)
            ib, _ = self._crop(tiles[j], box)
            masked_a = np.where(both[None], ia, np.nan)
            masked_b = np.where(both[None], ib, np.nan)
            ma, frac = block_medians(masked_a, block, 0.5)
            mb, _ = block_medians(masked_b, block, 0.5)
            keep = np.isfinite(ma).all(axis=0) & np.isfinite(mb).all(axis=0)
            if keep.sum() < 4:
                continue
            by, bx = np.nonzero(keep)
            X = ((bx + 0.5) * block + box[0]) / width * 2 - 1
            Y = ((by + 0.5) * block + box[1]) / height * 2 - 1
            samples.append((i, j, ma[:, keep], mb[:, keep], X, Y))
            st = by_pair.get((i, j))
            if st is None:
                st = OverlapStats(i, j, int(both.sum()), 0, (0.0, 0.0), float("nan"), float("nan"), float("nan"))
                stats.append(st)
                by_pair[(i, j)] = st
            st.residual_before = float(np.median(np.abs(ma[:, keep] - mb[:, keep])))
        for t in tiles:
            t.gain, t.offset, t.plane = np.ones(c), np.zeros(c), np.zeros((c, 2))
        if not samples:
            self.status("Photometric matching skipped: no usable overlaps")
            return stats
        connected = {0}
        changed = True
        while changed:
            changed = False
            for i, j, *_ in samples:
                if (i in connected) != (j in connected):
                    connected |= {i, j}
                    changed = True
        for k in range(n):
            if k not in connected:
                self.status(f"Warning: {tiles[k].name} is not connected to the reference tile through overlaps; left unscaled")
        per_tile = 4 if s.match_gradient else 2
        cols = per_tile * (n - 1)
        for ch in range(c):
            level = float(np.median(np.concatenate([a[ch] for _, _, a, _, _, _ in samples])))
            level = level if abs(level) > 1e-12 else 1.0
            rows, rhs = [], []
            counts = np.zeros(n)
            for i, j, a, b, X, Y in samples:
                k = a.shape[1]
                counts[i] += k
                counts[j] += k
                A = np.zeros((k, cols))
                r = np.zeros(k)
                av, bv = a[ch] / level, b[ch] / level
                for t_idx, vals, sign in ((i, av, 1.0), (j, bv, -1.0)):
                    if t_idx == 0:
                        r -= sign * vals  # reference: gain 1, offset 0, no plane
                        continue
                    base = per_tile * (t_idx - 1)
                    A[:, base] = sign * vals
                    A[:, base + 1] = sign
                    if s.match_gradient:
                        A[:, base + 2] = sign * X
                        A[:, base + 3] = sign * Y
                rows.append(A)
                rhs.append(r)
            A = np.concatenate(rows)
            r = np.concatenate(rhs)
            # weak priors: gain -> 1, plane -> 0. In normalised units the data carry ~K*var(signal) of
            # gain information (0.01K for a 10% nebula contrast) and block-median noise is ~1e-7K, so
            # 1e-4K only takes over when an overlap is featureless sky.
            prior_rows, prior_rhs = [], []
            for t_idx in range(1, n):
                base = per_tile * (t_idx - 1)
                wg = math.sqrt(1e-4 * max(counts[t_idx], 1.0))
                pr = np.zeros(cols)
                pr[base] = wg
                prior_rows.append(pr)
                prior_rhs.append(wg)
                if s.match_gradient:
                    wp = math.sqrt(1e-3 * max(counts[t_idx], 1.0))
                    for off in (2, 3):
                        pr = np.zeros(cols)
                        pr[base + off] = wp
                        prior_rows.append(pr)
                        prior_rhs.append(0.0)
            A_full = np.concatenate([A, np.array(prior_rows)])
            r_full = np.concatenate([r, np.array(prior_rhs)])
            weights = np.ones(len(A_full))
            sol = np.zeros(cols)
            for _ in range(5):
                sol, *_ = np.linalg.lstsq(A_full * weights[:, None], r_full * weights, rcond=None)
                resid = A @ sol - r
                sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
                new_w = np.concatenate([(np.abs(resid) < 3 * sigma).astype(float), np.ones(len(prior_rows))])
                if np.array_equal(new_w, weights):
                    break
                weights = new_w
            for t_idx in range(1, n):
                base = per_tile * (t_idx - 1)
                g = float(np.clip(sol[base], 0.2, 5.0))
                tiles[t_idx].gain[ch] = g
                tiles[t_idx].offset[ch] = float(sol[base + 1]) * level
                if s.match_gradient:
                    tiles[t_idx].plane[ch] = sol[base + 2 : base + 4] * level
        for t in tiles:
            self.status(
                f"Photometry {t.name}: gain [{', '.join(f'{v:.4f}' for v in t.gain)}], offset [{', '.join(_fmt(v, True) for v in t.offset)}]"
                + (f", plane [{', '.join(f'{_fmt(p[0], True)}/{_fmt(p[1], True)}' for p in t.plane)}]" if s.match_gradient else "")
            )
        # residual after
        for i, j, a, b, X, Y in samples:
            ca = self._apply_photometry_samples(tiles[i], a, X, Y)
            cb = self._apply_photometry_samples(tiles[j], b, X, Y)
            st = by_pair[(i, j)]
            st.residual_after = float(np.median(np.abs(ca - cb)))
            self.status(f"Overlap {tiles[i].name} / {tiles[j].name}: median |difference| {_fmt(st.residual_before)} -> {_fmt(st.residual_after)}")
        return stats

    @staticmethod
    def _apply_photometry_samples(t: Tile, vals: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return t.gain[:, None] * vals + t.offset[:, None] + t.plane[:, 0:1] * X[None] + t.plane[:, 1:2] * Y[None]

    def _corrected(self, t: Tile) -> tuple[torch.Tensor, torch.Tensor]:
        """Photometrically corrected warped tile on the device: (image with 0 where invalid, mask)."""

        backend = self.backend
        x0, y0, x1, y1 = t.bbox
        h, w = y1 - y0, x1 - x0
        height, width = self.canvas
        img = to_tensor(np.nan_to_num(t.warped, nan=0.0), backend)
        mask = to_tensor(t.valid.astype(np.float32), backend)
        gain = torch.tensor(t.gain, dtype=torch.float32, device=backend.device).view(-1, 1, 1)
        offset = torch.tensor(t.offset, dtype=torch.float32, device=backend.device).view(-1, 1, 1)
        xs = (torch.arange(x0, x1, device=backend.device, dtype=torch.float32) + 0.5) / width * 2 - 1
        ys = (torch.arange(y0, y1, device=backend.device, dtype=torch.float32) + 0.5) / height * 2 - 1
        px = torch.tensor(t.plane[:, 0], dtype=torch.float32, device=backend.device).view(-1, 1, 1)
        py = torch.tensor(t.plane[:, 1], dtype=torch.float32, device=backend.device).view(-1, 1, 1)
        img = (gain * img + offset + px * xs.view(1, 1, w) + py * ys.view(1, h, 1)) * mask
        return img, mask

    # ------------------------------------------------------------------ blending

    def blend(self, tiles: list[Tile]) -> tuple[np.ndarray, np.ndarray]:
        from scipy import ndimage

        s = self.settings
        backend = self.backend
        height, width = self.canvas
        c = tiles[0].image.shape[0]
        den_wide = np.zeros((height, width), dtype=np.float32)
        best = np.zeros((height, width), dtype=np.float32)
        label = np.full((height, width), -1, dtype=np.int16)
        coverage = np.zeros((height, width), dtype=np.uint8)
        for k, t in enumerate(tiles):
            self._check()
            x0, y0, x1, y1 = t.bbox
            dist = ndimage.distance_transform_edt(t.valid).astype(np.float32)
            w = np.clip(dist / max(s.feather, 1.0), 0.0, 1.0)
            w = np.where(t.valid, np.maximum(w, 1e-3), 0.0).astype(np.float32)
            t.weight = w
            den_wide[y0:y1, x0:x1] += w
            coverage[y0:y1, x0:x1] += t.valid
            sub_best = best[y0:y1, x0:x1]
            better = w > sub_best
            sub_best[better] = w[better]
            label[y0:y1, x0:x1][better] = k
        del best
        den_narrow = np.zeros((height, width), dtype=np.float32)
        for k, t in enumerate(tiles):
            if s.blend_mode != "seam":
                break
            self._check()
            x0, y0, x1, y1 = t.bbox
            onehot = to_tensor((label[y0:y1, x0:x1] == k).astype(np.float32), backend)
            narrow = _gaussian_blur(onehot.unsqueeze(0), s.seam_width)[0]
            narrow = to_numpy(narrow) * t.valid
            t.narrow = narrow.astype(np.float32)
            den_narrow[y0:y1, x0:x1] += t.narrow
        out = np.zeros((c, height, width), dtype=np.float32)
        # L = wide-feather composite (smooth background), H = hard-seam composite (one tile per place,
        # so misregistered stars are never doubled). out = H + blur(L - H) takes the low frequencies
        # from L and the detail from H, with the blur over the shared canvas so no tile edge biases it.
        hard = np.zeros_like(out) if s.blend_mode == "seam" else None
        for k, t in enumerate(tiles):
            self._check()
            self.progress(0.75 + 0.15 * k / len(tiles), f"Blending {t.path.name}")
            x0, y0, x1, y1 = t.bbox
            img, mask = self._corrected(t)
            w_wide = to_tensor(t.weight / np.maximum(den_wide[y0:y1, x0:x1], 1e-6), backend)
            out[:, y0:y1, x0:x1] += to_numpy(img * w_wide.unsqueeze(0))
            if hard is not None:
                w_narrow = to_tensor(t.narrow / np.maximum(den_narrow[y0:y1, x0:x1], 1e-6), backend)
                hard[:, y0:y1, x0:x1] += to_numpy(img * w_narrow.unsqueeze(0))
                del w_narrow
            t.warped = None  # free host memory as we go
            del img, mask, w_wide
            backend.empty_cache()
        if hard is not None:
            self.progress(0.92, "Blending bands")
            out -= hard
            smooth = _blur_canvas(out, coverage > 0, max(4.0, s.feather / 2.0), backend)
            out = hard
            out += smooth
            del smooth
        out[:, coverage == 0] = 0.0
        return out, coverage

    # ------------------------------------------------------------------ output

    def save(self, tiles: list[Tile], image: np.ndarray, coverage: np.ndarray, crop_box: tuple[int, int, int, int] | None = None) -> tuple[Path, Path | None]:
        s = self.settings
        first = tiles[0]
        base = {k: v for k, v in first.header.items() if k not in ("DRIZSCL", "CROPX0", "CROPY0", "CROPDPTH", "MAPTYPE", "NFRAMES", "NEFF", "REJFRAC", "BGNOISE")}
        items: list[tuple[str, Any, str]] = [
            ("GPUSTACK", "mosaic", "GPUStacker mosaic"),
            ("MOSTILES", len(tiles), "Number of mosaic tiles"),
            ("MOSSCALE", round(self.out_scale, 5), "Output pixel scale [arcsec/px]"),
            ("MOSGRAD", s.gradient_degree, "Per-tile gradient polynomial degree (0 = off)"),
            ("MOSFEATH", s.feather, "Background feather width [px]"),
            ("MOSSEAM", s.seam_width, "Detail seam width [px]"),
            ("MOSPHOT", bool(s.photometric), "Photometric matching in overlaps"),
            ("MOSCROP", crop_box is not None, "Largest fully covered rectangle"),
        ]
        if crop_box is not None:
            y0, y1, x0, x1 = crop_box
            items.extend([
                ("CROPX0", x0, "Autocrop left edge in mosaic canvas"),
                ("CROPY0", y0, "Autocrop top edge in mosaic canvas"),
                ("CROPX1", x1, "Autocrop right edge exclusive"),
                ("CROPY1", y1, "Autocrop bottom edge exclusive"),
            ])
        for k, t in enumerate(tiles):
            items.append((f"MOSTIL{k:02d}", t.path.name[:68], ""))
            items.append((f"MOSGN{k:03d}", round(float(np.mean(t.gain)), 5), f"Tile {k} mean gain"))
        xpix = first.header.get("XPIXSZ")
        try:
            if xpix is not None:
                items.append(("XPIXSZ", float(_coerce(xpix)) * self.out_scale / first.scale, "Effective pixel size [um]"))
                items.append(("YPIXSZ", float(_coerce(xpix)) * self.out_scale / first.scale, "Effective pixel size [um]"))
        except (TypeError, ValueError):
            pass
        items.append(("RA", float(self.out_wcs.wcs.crval[0]), "Mosaic centre RA [deg]"))
        items.append(("DEC", float(self.out_wcs.wcs.crval[1]), "Mosaic centre Dec [deg]"))
        items.extend(wcs_header_items(self.out_wcs))
        out = save_fits(s.output, image, items, base)
        cov_path = None
        if s.save_coverage:
            cov_path = out.with_name(out.stem + "_coverage" + out.suffix)
            save_fits(cov_path, coverage.astype(np.float32), [("MAPTYPE", "mosaic_coverage", "Tiles per pixel")] + wcs_header_items(self.out_wcs))
        return out, cov_path


def build_mosaic(settings: MosaicSettings, status_cb: StatusCallback = _noop_status, progress_cb: ProgressCallback = _noop_progress) -> MosaicResult:
    return MosaicBuilder(settings, status_cb, progress_cb).run()
