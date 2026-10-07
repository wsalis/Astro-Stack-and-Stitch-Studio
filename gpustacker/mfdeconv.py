"""ImageMM multi-frame PSF-aware deconvolution (majorize-minimize, robust weights).

Implements the method of Sukurdeep (2025, AJ 170, doi:10.3847/1538-3881/adfb72)
following the practical notes of Marek (2026, doi:10.5281/zenodo.19168050):
Huber-weighted least squares with a soft star keep-mask and per-pixel variance
normalisation, multiplicative MM update with step clipping (kappa) and relaxation
(alpha), auto-k PSF estimation, and median(|u-1|) early stopping. See CREDITS.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .backend import Backend, StatusCallback
from .detection import (
    FWHM_PER_SIGMA,
    Stars,
    build_star_psf,
    choose_psf_size,
    crop_kernel,
    detect_stars,
    estimate_background,
    gaussian_kernel,
    luma,
    psf_support_radius,
    soften_kernel,
)

EPS = 1e-6
ProgressCallback = Callable[[float, str], None]


def _noop_status(_: str) -> None:
    pass


def _noop_progress(_: float, __: str) -> None:
    pass


@dataclass
class MFDeconvSettings:
    iterations: int = 30
    kappa: float = 2.0
    relax: float = 0.6
    huber_delta: float = -1.5  # negative => auto: delta = |value| * RMS(residual)
    # lrgb: deconvolve luminance only and re-apply the stack's colour (no colour ringing/halos)
    color_mode: Literal["lrgb", "luma", "perchannel"] = "lrgb"
    psf_size: int = 0  # 0 => auto-k: at least 4 sigma, grown to cover the measured wing (see psf_wing_floor)
    psf_wing_floor: float = 1e-3  # auto-k keeps the kernel out to where the star wing falls below this fraction of the peak
    psf_soften_sigma: float = 0.25
    use_star_masks: bool = True
    mask_thresh_sigma: float = 5.0
    mask_grow_px: int = 2
    mask_ellipse_scale: float = 2.5
    mask_soft_sigma: float = 1.5
    mask_max_radius_px: int = 25
    use_variance_maps: bool = True
    variance_smooth_sigma: float = 2.0
    tile_size: int = 1024
    tile_overlap: int = 0  # 0 => 3 * psf size
    early_stop_tol: float = 0.02  # stop when the median |update| drops below this fraction of the sky noise
    dering_sigma: float = 2.0  # floor x at local background - sigma*noise to stop dark rings; 0 disables
    dering_star_sigma: float = 1.0  # lift a 1px ring around a star whose MEAN sits this many standard errors below background; <0 disables
    dering_box: int = 32  # background block size; must exceed the ring radius but follow nebula structure
    save_psf_dir: Path | None = None


# Strength presets (values untested beyond first-principles reasoning; tune after comparing on real data).
# "frames" is the number of best-weighted frames fed to the solver.
MFDECONV_PRESETS: dict[str, dict[str, float | int]] = {
    "gentle": {"frames": 24, "iterations": 10, "kappa": 1.5, "relax": 0.6, "huber_delta": -1.5, "dering_sigma": 2.0, "early_stop_tol": 0.1},
    "normal": {"frames": 24, "iterations": 15, "kappa": 1.5, "relax": 0.6, "huber_delta": -1.5, "dering_sigma": 2.0, "early_stop_tol": 0.05},
    "strong": {"frames": 12, "iterations": 30, "kappa": 2.0, "relax": 0.6, "huber_delta": -1.5, "dering_sigma": 1.5, "early_stop_tol": 0.02},
}


@dataclass
class FrameAssets:
    psf: np.ndarray
    psf_source: str
    fwhm: float
    keep_mask: np.ndarray | None  # (H, W) in [0, 1]
    variance: np.ndarray | None  # (H, W) DN^2
    stars: int
    channel_psfs: np.ndarray | None = None  # (C, k, k): colours focus differently (e.g. Ha vs OIII through a refractor)


@dataclass
class MFDeconvResult:
    image: np.ndarray  # (C, H, W)
    iterations_run: int
    assets: list[FrameAssets] = field(default_factory=list)
    early_stopped: bool = False


# ----------------------------------------------------------------------------- assets


def estimate_psf(image2d: np.ndarray, settings: MFDeconvSettings, header_fwhm: float | None = None) -> tuple[np.ndarray, str, float, Stars]:
    """Auto-k PSF: star-derived kernel over a shortlist of sizes, Gaussian fallback."""

    stars = detect_stars(image2d, thresh_sigma=settings.mask_thresh_sigma)
    fwhm = header_fwhm if header_fwhm and header_fwhm > 0 else stars.median_fwhm()
    if not np.isfinite(fwhm) or fwhm <= 0:
        fwhm = 3.0
    k = settings.psf_size if settings.psf_size > 0 else choose_psf_size(fwhm)
    k |= 1
    if settings.psf_size <= 0:
        # measure the wing on a generous stamp and trim to where it drops below the floor
        wide = build_star_psf(image2d, stars, choose_psf_size(fwhm, lo=k, sigmas=10.0))
        if wide is not None:
            k_wing = min(wide.shape[-1], 2 * max(k // 2, psf_support_radius(wide, settings.psf_wing_floor) + 1) + 1)
            return soften_kernel(crop_kernel(wide, k_wing), settings.psf_soften_sigma), f"stars(k={k_wing})", float(fwhm), stars
    shortlist = [s for s in dict.fromkeys([k, k - 4, 21, 17, 15, 13, 11]) if s >= 5]
    for size in shortlist:
        psf = build_star_psf(image2d, stars, size)
        if psf is not None:
            return soften_kernel(psf, settings.psf_soften_sigma), f"stars(k={size})", float(fwhm), stars
    psf = gaussian_kernel(k, fwhm)
    return soften_kernel(psf, settings.psf_soften_sigma), f"gaussian(k={k})", float(fwhm), stars


def saturation_level(image2d: np.ndarray, plateau_px: int = 4) -> float | None:
    """Level above which pixels are clipped, or None when nothing is saturated.

    Saturated stars have flat-topped cores: several pixels at (nearly) the image maximum.
    """

    finite = image2d[np.isfinite(image2d)]
    if finite.size == 0:
        return None
    top = float(finite.max())
    if int((finite >= 0.995 * top).sum()) < plateau_px:
        return None
    return 0.9 * top


def build_keep_mask(shape: tuple[int, int], stars: Stars, settings: MFDeconvSettings, image2d: np.ndarray | None = None) -> np.ndarray:
    """Soft keep-mask M = 1 - (dilated, blurred disks around SATURATED stars).

    Unsaturated stars must stay in the data term: masking a core removes the only evidence that it
    should brighten, so the solver dims it from the wing residuals alone (flat disc + dark rim).
    """

    height, width = shape
    mask = np.zeros((height, width), dtype=np.float32)
    sat = saturation_level(image2d) if image2d is not None else None
    if sat is not None and len(stars):
        xi = np.clip(np.round(stars.x).astype(int), 0, width - 1)
        yi = np.clip(np.round(stars.y).astype(int), 0, height - 1)
        clipped = np.asarray(image2d)[yi, xi] >= sat
        radii = np.clip(settings.mask_ellipse_scale * np.maximum(stars.a, stars.b), 1.0, settings.mask_max_radius_px)
        yy, xx = np.mgrid[0:height, 0:width]
        for x, y, r in zip(stars.x[clipped], stars.y[clipped], radii[clipped]):
            r_int = int(math.ceil(r)) + 1
            x0, x1 = max(0, int(x) - r_int), min(width, int(x) + r_int + 1)
            y0, y1 = max(0, int(y) - r_int), min(height, int(y) + r_int + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            disk = (xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2 <= r * r
            mask[y0:y1, x0:x1][disk] = 1.0
        if settings.mask_grow_px > 0:
            mask = ndimage.binary_dilation(mask > 0.5, iterations=settings.mask_grow_px).astype(np.float32)
        if settings.mask_soft_sigma > 0:
            mask = ndimage.gaussian_filter(mask, settings.mask_soft_sigma)
    return np.clip(1.0 - mask, 0.0, 1.0).astype(np.float32)


def build_variance_map(image2d: np.ndarray, gain: float | None, read_noise: float | None, settings: MFDeconvSettings, scale: float = 1.0) -> np.ndarray:
    """Variance in normalized DN^2 from a native-DN image that already includes sky."""

    img = np.clip(np.asarray(image2d, dtype=np.float32), 0.0, None)
    _, rms = estimate_background(img)
    if gain and gain > 0:
        rn = read_noise if read_noise is not None and read_noise >= 0 else rms * gain
        var = (img * gain + rn * rn) / (gain * gain)
    else:
        var = np.full_like(img, max(rms, 1e-6) ** 2)
    var *= float(scale) ** 2
    if settings.variance_smooth_sigma > 0:
        var = ndimage.gaussian_filter(var, settings.variance_smooth_sigma)
    return np.clip(var, 1e-9, None).astype(np.float32)


def build_luma_variance_map(frame: np.ndarray, gain: float | None, read_noise: float | None, settings: MFDeconvSettings, scales: np.ndarray) -> np.ndarray:
    """Combine native-channel variances after the per-channel linear flux scaling."""

    image = np.asarray(frame, dtype=np.float32)
    scale = np.asarray(scales, dtype=np.float32).reshape(-1)
    if image.shape[0] == 1:
        return build_variance_map(image[0], gain, read_noise, settings, float(scale[0]))
    coefficients = (0.2126, 0.7152, 0.0722)
    variance = np.zeros(image.shape[-2:], dtype=np.float32)
    for channel, (coefficient, channel_scale) in enumerate(zip(coefficients, scale)):
        variance += (coefficient * channel_scale) ** 2 * build_variance_map(image[channel], gain, read_noise, settings)
    return variance


def prepare_assets(frames: list[np.ndarray], settings: MFDeconvSettings, gains: list[float | None] | None = None, read_noises: list[float | None] | None = None, header_fwhms: list[float | None] | None = None, status_cb: StatusCallback = _noop_status, variance_frames: list[np.ndarray] | None = None, variance_scales: list[np.ndarray] | None = None, rejection_masks: list[np.ndarray] | None = None) -> list[FrameAssets]:
    if rejection_masks is not None and len(rejection_masks) != len(frames):
        raise ValueError("One rejection mask is required per MFDeconv frame")
    assets: list[FrameAssets] = []
    for t, frame in enumerate(frames):
        lum = luma(frame)
        coverage = np.isfinite(lum)
        lum = np.nan_to_num(lum, nan=0.0)
        psf, source, fwhm, stars = estimate_psf(lum, settings, (header_fwhms or [None] * len(frames))[t])
        keep = None
        if settings.use_star_masks:
            keep = build_keep_mask(lum.shape, stars, settings, lum)
        if not coverage.all():
            keep = (keep if keep is not None else np.ones(lum.shape, dtype=np.float32)) * coverage.astype(np.float32)
        if rejection_masks is not None:
            rejected = np.asarray(rejection_masks[t], dtype=bool)
            if rejected.shape != lum.shape:
                raise ValueError(f"Rejection mask shape {rejected.shape} does not match frame shape {lum.shape}")
            keep = (keep if keep is not None else np.ones(lum.shape, dtype=np.float32)) * (~rejected)
        var = None
        if settings.use_variance_maps:
            gain = (gains or [None] * len(frames))[t]
            read_noise = (read_noises or [None] * len(frames))[t]
            if variance_frames is None:
                var = build_variance_map(lum, gain, read_noise, settings)
            else:
                scales = (variance_scales or [np.ones(frame.shape[0], dtype=np.float32) for frame in frames])[t]
                var = build_luma_variance_map(variance_frames[t], gain, read_noise, settings, scales)
        channel_psfs = None
        if settings.color_mode == "perchannel" and frame.shape[0] > 1 and source.startswith("stars"):
            size = psf.shape[-1]
            sat = float(np.percentile(lum, 99.995))
            per = []
            for c in range(frame.shape[0]):
                p = build_star_psf(np.nan_to_num(frame[c], nan=0.0), stars, size, saturation=sat)
                per.append(soften_kernel(p, settings.psf_soften_sigma) if p is not None else psf)
            channel_psfs = np.stack(per).astype(np.float32)
        assets.append(FrameAssets(psf, source, fwhm, keep, var, len(stars), channel_psfs))
        status_cb(f"MFDeconv assets {t + 1}/{len(frames)}: PSF {source}, FWHM {fwhm:.2f}px, stars {len(stars)}")
        if settings.save_psf_dir is not None:
            from .io import save_fits

            save_fits(Path(settings.save_psf_dir) / f"psf_{t:06d}.fit", psf, [("PSFSRC", source, "PSF origin"), ("FWHM", fwhm, "px")])
    return assets


# ----------------------------------------------------------------------------- core MM


def _psf_fwhm(psf: np.ndarray) -> float:
    """Gaussian-equivalent FWHM of a unit-sum kernel from its second moment."""

    r = np.arange(psf.shape[0]) - psf.shape[0] // 2
    var = float((psf * (r[:, None] ** 2 + r[None, :] ** 2)).sum()) / 2.0
    return FWHM_PER_SIGMA * np.sqrt(max(var, 0.0))


def _fft_corr(x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """'valid' cross-correlation of x (N, H, W) with kernels (N or 1, k, k) -> (N, H-k+1, W-k+1)."""

    k = kernels.shape[-1]
    height, width = x.shape[-2:]
    size = (height + k - 1, width + k - 1)
    spec = torch.fft.rfft2(x, s=size) * torch.fft.rfft2(torch.flip(kernels, (-2, -1)), s=size)
    return torch.fft.irfft2(spec, s=size)[..., k - 1 : height, k - 1 : width]


FFT_MIN_K = 17  # cuDNN direct conv is fast below this; at k=29 it is ~30x slower than the FFT route


def _conv_same(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Depthwise 'same' convolution with reflect padding. x: (C, H, W), kernel: (k, k) or per-channel (C, k, k)."""

    channels = x.shape[0]
    k = kernel.shape[-1]
    pad = k // 2
    if kernel.ndim == 3 and kernel.shape[0] == channels:
        weight = kernel.unsqueeze(1)
    else:
        weight = kernel.reshape(-1, k, k)[:1].unsqueeze(1).expand(channels, 1, k, k)
    xp = F.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    if k > FFT_MIN_K:
        return _fft_corr(xp[0], weight[:, 0])
    return F.conv2d(xp, weight, groups=channels)[0]


def _weight_map(y: torch.Tensor, pred: torch.Tensor, huber_delta: float, variance: torch.Tensor | None, keep: torch.Tensor | None) -> torch.Tensor:
    r = y - pred
    if huber_delta < 0:
        delta = (-huber_delta) * torch.sqrt(torch.mean(r * r)).clamp_min(EPS)
    else:
        delta = torch.tensor(huber_delta, device=y.device, dtype=y.dtype)
    ar = r.abs()
    psi_over_r = torch.where(ar <= delta, torch.ones_like(ar), delta / (ar + EPS))
    if variance is None:
        mad = torch.median((r - torch.median(r)).abs()) * 1.4826
        variance = (mad * mad).clamp_min(EPS)
    w = psi_over_r / (variance + EPS)
    if keep is not None:
        w = w * keep
    return w


def local_sky_noise(x: torch.Tensor, box: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Local background (lower-quartile of ``box`` blocks, smoothly upsampled) and per-channel robust noise.

    The floor must follow nebulosity, not just the sky: a dark ring around a star embedded in
    nebula sits far above the sky, so a sky-only floor never catches it. The lower quartile of a
    small block tracks the surrounding nebula while staying below stars and respecting dark lanes.
    """

    channels, height, width = x.shape
    pad_h = (-height) % box
    pad_w = (-width) % box
    xp = F.pad(x.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect")[0]
    blocks = xp.unfold(1, box, box).unfold(2, box, box)  # (C, bh, bw, box, box)
    flat = blocks.reshape(channels, blocks.shape[1], blocks.shape[2], -1)
    level = flat.kthvalue(max(1, flat.shape[-1] // 4), dim=-1).values
    bg = F.interpolate(level.unsqueeze(0), size=(xp.shape[1], xp.shape[2]), mode="bilinear", align_corners=False)[0][:, :height, :width]
    resid = (x - bg).flatten(1)
    centre = resid.median(dim=1).values.view(channels, 1)
    noise = ((resid - centre).abs().median(dim=1).values * 1.4826).view(channels, 1, 1).clamp_min(EPS)
    # the lower quartile sits ~0.67 sigma under the local median; add that back so "sigma below
    # background" keeps its meaning regardless of the quartile choice
    return bg + centre.view(channels, 1, 1), noise


def sky_floor(x: torch.Tensor, sigma: float, box: int) -> torch.Tensor:
    sky, noise = local_sky_noise(x, box)
    return sky - sigma * noise


def star_seeds(x: torch.Tensor, sky: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """(1, 1, H, W) float map with 1 at compact point-source peaks.

    A star is a local maximum > 5 sigma that stands > 3 sigma above the mean of its 7x7 surroundings;
    smooth nebulosity fails the second test.
    """

    lum = (x - sky).mean(dim=0, keepdim=True).unsqueeze(0)
    n = noise.mean()
    peak = (lum == F.max_pool2d(lum, 3, stride=1, padding=1)) & (lum > 5.0 * n)
    compact = (lum - F.avg_pool2d(F.pad(lum, (3, 3, 3, 3), mode="replicate"), 7, stride=1)) > 3.0 * n
    return (peak & compact).float()


def ring_kernels(r_min: int, r_max: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(R, 1, K, K) 1px-annulus indicators and their pixel counts for radii r_min..r_max."""

    yy, xx = torch.meshgrid(torch.arange(-r_max - 1, r_max + 2, device=device), torch.arange(-r_max - 1, r_max + 2, device=device), indexing="ij")
    rr = torch.sqrt((xx**2 + yy**2).float())
    rings = torch.stack([((rr >= r - 0.5) & (rr < r + 0.5)).float() for r in range(r_min, r_max + 1)]).unsqueeze(1)
    return rings, rings.flatten(1).sum(dim=1)


def lift_star_rings(x: torch.Tensor, sky: torch.Tensor, noise: torch.Tensor, seeds: torch.Tensor, rings: torch.Tensor, counts: torch.Tensor, sigma: float) -> torch.Tensor:
    """Raise any 1px annulus around a star whose mean sits below the background by more than ``sigma``
    standard errors of that mean; the deficit is added uniformly over the annulus (texture unchanged)."""

    pad = rings.shape[-1] // 2
    out = x.clone()
    for c in range(x.shape[0]):
        resid = (x[c] - sky[c]).view(1, *x.shape[1:])
        means = _fft_corr(F.pad(resid.unsqueeze(0), (pad, pad, pad, pad), mode="replicate")[0], rings[:, 0]) / counts.view(-1, 1, 1)
        tol = sigma * noise[c] / counts.sqrt().view(-1, 1, 1)
        deficit = (-means - tol).clamp_min(0.0) * seeds[0]  # (R, H, W), nonzero only at star centres
        # transpose of the ring correlation = correlation with the flipped rings, summed over radii
        out[c] += _fft_corr(F.pad(deficit, (pad, pad, pad, pad)), torch.flip(rings[:, 0], (-2, -1))).sum(dim=0)
    return out


def mm_deconvolve(y: torch.Tensor, psfs: torch.Tensor, keep: torch.Tensor | None, variance: torch.Tensor | None, settings: MFDeconvSettings, progress_cb: ProgressCallback = _noop_progress) -> tuple[torch.Tensor, int, bool]:
    """Run the MM loop on device tensors.

    y: (T, C, H, W); psfs: (T, k, k); keep/variance: (T, 1, H, W) or None.
    Returns (x, iterations_run, early_stopped).
    """

    frames = y.shape[0]
    covered = ~torch.isnan(y).all(dim=0)  # (C, H, W): at least one frame has data
    x = torch.nan_to_num(torch.nanmedian(y, dim=0).values, nan=0.0).clamp_min(0.0)
    fill = x[covered].median() if bool(covered.any()) else torch.tensor(0.0, device=y.device)
    x = torch.where(covered, x, fill)  # NaN here would spread through every convolution
    y = torch.nan_to_num(y, nan=0.0).clamp_min(0.0)
    sky, noise = local_sky_noise(x, settings.dering_box)
    floor = sky - settings.dering_sigma * noise if settings.dering_sigma > 0 else None
    psfs_t = torch.flip(psfs, dims=(-2, -1))
    ring_weight = None
    if settings.dering_sigma > 0 and settings.dering_star_sigma >= 0:
        # the global floor must leave room for noise below the background, so on its own it lets the solver
        # dig a coherent trough around every star down to it; lift only troughs too coherent to be noise
        ring_weight = star_seeds(x, sky, noise)
        ring_k, ring_n = ring_kernels(2, psfs.shape[-1] // 2 + 4, x.device)
    frozen = None
    if keep is not None:
        # fraction of each pixel's PSF footprint that still has data; masked cores fall well below 1
        support = sum(_conv_same(keep[t].expand_as(x), psfs_t[t]) for t in range(frames)) / frames
        frozen = support < 0.5
    ran = 0
    early = False
    for it in range(1, settings.iterations + 1):
        num = torch.zeros_like(x)
        den = torch.zeros_like(x)
        for t in range(frames):
            pred = _conv_same(x, psfs[t])
            w = _weight_map(y[t], pred, settings.huber_delta, None if variance is None else variance[t], None if keep is None else keep[t])
            num += _conv_same(w * y[t], psfs_t[t])
            den += _conv_same(w * pred, psfs_t[t])
        u = torch.clamp(num / (den + EPS), 1.0 / settings.kappa, settings.kappa)
        u = torch.where(den > EPS, u, torch.ones_like(u))  # no data -> leave pixel alone
        u = torch.nan_to_num(u, nan=1.0)
        if frozen is not None:
            u = torch.where(frozen, torch.ones_like(u), u)
        x_next = (x * u).clamp_min(0.0)
        if floor is not None:
            x_next = torch.maximum(x_next, floor)
        ran = it
        # relative to the noise, not to x: a sky pedestal makes |u-1| tiny everywhere
        early = settings.early_stop_tol > 0 and float(torch.median((x_next - x).abs() / noise)) < settings.early_stop_tol
        x = x_next if early else (1.0 - settings.relax) * x + settings.relax * x_next
        if ring_weight is not None:
            # after the relax blend: applied to x_next only, the blend would carry the old trough forward
            resid = (x - sky).flatten(1)
            noise_now = 1.4826 * (resid - resid.median(dim=1, keepdim=True).values).abs().median(dim=1).values
            x = lift_star_rings(x, sky, noise_now, ring_weight, ring_k, ring_n, settings.dering_star_sigma)
        if early:
            progress_cb(1.0, f"MFDeconv iter {it}/{settings.iterations} (early stop)")
            break
        progress_cb(it / settings.iterations, f"MFDeconv iter {it}/{settings.iterations}")
    return x, ran, early


# ----------------------------------------------------------------------------- tiling


def _tiles(length: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    if length <= tile:
        return [(0, length)]
    step = max(1, tile - 2 * overlap)
    starts = list(range(0, max(1, length - tile) + 1, step))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return [(s, s + tile) for s in starts]


def _padded_slice(arr: torch.Tensor, y0: int, y1: int, x0: int, x1: int, pad: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Slice [..., y0-pad:y1+pad, x0-pad:x1+pad] with reflect fill beyond the image."""

    height, width = arr.shape[-2:]
    ys0, ys1 = max(0, y0 - pad), min(height, y1 + pad)
    xs0, xs1 = max(0, x0 - pad), min(width, x1 + pad)
    chunk = arr[..., ys0:ys1, xs0:xs1]
    pads = (xs0 - (x0 - pad), (x1 + pad) - xs1, ys0 - (y0 - pad), (y1 + pad) - ys1)
    if any(pads):
        shape = chunk.shape
        flat = chunk.reshape(-1, 1, *shape[-2:])
        flat = F.pad(flat, pads, mode="reflect")
        chunk = flat.reshape(*shape[:-2], *flat.shape[-2:])
    return chunk, pads


def run_mfdeconv(frames: list[np.ndarray], backend: Backend, settings: MFDeconvSettings, gains: list[float | None] | None = None, read_noises: list[float | None] | None = None, header_fwhms: list[float | None] | None = None, status_cb: StatusCallback = _noop_status, progress_cb: ProgressCallback = _noop_progress, variance_frames: list[np.ndarray] | None = None, variance_scales: list[np.ndarray] | None = None, rejection_masks: list[np.ndarray] | None = None) -> MFDeconvResult:
    """Deconvolve registered (C, H, W) frames into one latent image, tiling to fit VRAM."""

    if len(frames) < 2:
        raise ValueError("MFDeconv needs at least two registered frames")
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"All frames must share one shape, got {shapes}")
    channels, height, width = frames[0].shape
    status_cb(f"MFDeconv: {len(frames)} frames, {channels}x{height}x{width}, mode={settings.color_mode}")

    assets = prepare_assets(frames, settings, gains, read_noises, header_fwhms, status_cb, variance_frames, variance_scales, rejection_masks)
    k = max(a.psf.shape[-1] for a in assets)
    per_channel = settings.color_mode == "perchannel" and all(a.channel_psfs is not None for a in assets)

    def padded(p: np.ndarray) -> np.ndarray:
        m = (k - p.shape[-1]) // 2
        return np.pad(p, ((0, 0),) * (p.ndim - 2) + ((m, m), (m, m)))

    psfs_np = np.stack([padded(a.channel_psfs if per_channel else a.psf) for a in assets]).astype(np.float32)
    psfs = torch.from_numpy(psfs_np).to(backend.device)
    if per_channel:
        widths = np.median([[_psf_fwhm(cp) for cp in a.channel_psfs] for a in assets], axis=0)
        status_cb("MFDeconv: per-channel PSFs, FWHM-equivalent R/G/B " + "/".join(f"{v:.2f}" for v in widths) + " px")

    if settings.color_mode in ("luma", "lrgb") and channels == 3:
        y_all = torch.from_numpy(np.stack([luma(f)[None] for f in frames]))
    else:
        y_all = torch.from_numpy(np.stack(frames))
    work_channels = y_all.shape[1]
    keep_all = torch.from_numpy(np.stack([a.keep_mask[None] for a in assets])) if all(a.keep_mask is not None for a in assets) else None
    var_all = torch.from_numpy(np.stack([a.variance[None] for a in assets])) if all(a.variance is not None for a in assets) else None

    overlap = settings.tile_overlap if settings.tile_overlap > 0 else 3 * k
    per_pixel = len(frames) * work_channels * 4 * 10
    fit_rows = backend.budget_bytes // max(1, per_pixel * width)
    tile = settings.tile_size
    if fit_rows >= height and backend.budget_bytes >= per_pixel * width * height:
        tile = max(height, width)
    row_tiles = _tiles(height, tile, overlap)
    col_tiles = _tiles(width, tile, overlap)
    total_tiles = len(row_tiles) * len(col_tiles)
    if total_tiles == 1:
        overlap = 0  # whole image at once; conv handles borders via reflect padding
    status_cb(f"MFDeconv: PSF k={k}, {total_tiles} tile(s) of {tile}px (overlap {overlap}px)")

    out = torch.zeros((work_channels, height, width), dtype=torch.float32)
    done = 0
    iters_run = 0
    early_any = False
    for y0, y1 in row_tiles:
        for x0, x1 in col_tiles:
            y_tile, _ = _padded_slice(y_all, y0, y1, x0, x1, overlap)
            k_tile = _padded_slice(keep_all, y0, y1, x0, x1, overlap)[0].to(backend.device) if keep_all is not None else None
            v_tile = _padded_slice(var_all, y0, y1, x0, x1, overlap)[0].to(backend.device) if var_all is not None else None
            base = done / total_tiles

            def tile_progress(frac: float, msg: str, base: float = base) -> None:
                progress_cb(base + frac / total_tiles, msg)

            x_tile, ran, early = mm_deconvolve(y_tile.to(backend.device), psfs, k_tile, v_tile, settings, tile_progress)
            iters_run = max(iters_run, ran)
            early_any |= early
            out[:, y0:y1, x0:x1] = x_tile[:, overlap : overlap + (y1 - y0), overlap : overlap + (x1 - x0)].cpu()
            done += 1
            del x_tile, y_tile, k_tile, v_tile
            backend.empty_cache()
    status_cb(f"MFDeconv: finished after {iters_run} iteration(s){' with early stop' if early_any else ''}")
    image = out.numpy()
    if settings.color_mode == "lrgb" and channels == 3:
        image = _apply_luminance(np.nan_to_num(np.nanmean(np.stack(frames), axis=0)).astype(np.float32), image[0])
    return MFDeconvResult(image, iters_run, assets, early_any)


def _apply_luminance(colour: np.ndarray, lum_new: np.ndarray, min_snr: float = 3.0) -> np.ndarray:
    """Replace the luminance of ``colour`` (C, H, W) with ``lum_new`` keeping the colour of the SIGNAL.

    The luminance change is shared out in proportion to each channel's share of the above-sky signal;
    on sky (signal < min_snr sigma) it is shared neutrally, so the background colour is untouched.
    """

    lum_old = luma(colour)
    bg = np.stack([estimate_background(c, 64)[0] for c in colour])
    sig = colour - bg
    sig_l = luma(sig)
    _, noise = estimate_background(lum_old, 64)
    weights = np.where(sig_l[None] > min_snr * noise, sig / np.maximum(sig_l[None], 1e-6), 1.0)
    weights = np.clip(weights, 0.0, 3.0).astype(np.float32)
    return (colour + (lum_new - lum_old)[None] * weights).astype(np.float32)
