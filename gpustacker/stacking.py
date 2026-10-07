"""Chunked (row-band) GPU stacking with outlier rejection.

Registered frames live on disk as float32 ``.npy`` memmaps shaped (C, H, W) with NaN
marking uncovered pixels. Bands of rows are streamed to the device so the stack never
has to fit in VRAM all at once.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .backend import Backend, rows_per_band

RejectionMethod = Literal["none", "median", "sigma", "winsorized", "percentile", "gesd"]
ProgressCallback = Callable[[float, str], None]


def _noop_progress(_: float, __: str) -> None:
    pass


@dataclass
class FrameNorm:
    offset: float | np.ndarray = 0.0  # background level of this frame; scalar or per-channel (C,)
    scale: float | np.ndarray = 1.0  # multiplicative factor to bring noise/level onto the reference; scalar or (C,)
    weight: float = 1.0
    # optional local normalisation: low-res (C, bh, bw) maps, frame' = scale_map * frame + offset_map
    scale_map: np.ndarray | None = None
    offset_map: np.ndarray | None = None
    weight_map: np.ndarray | None = None  # low-resolution spatial quality weights on the reference grid


def norm_vec(value: float | np.ndarray, channels: int) -> np.ndarray:
    """Broadcast a scalar or (C,) normalisation parameter to a float32 (C,) array."""

    return np.broadcast_to(np.asarray(value, dtype=np.float32).reshape(-1), (channels,)).copy()


@dataclass
class FrameStore:
    """Registered frames on disk plus their normalisation parameters."""

    directory: Path
    shape: tuple[int, int, int]  # (C, H, W)
    paths: list[Path] = field(default_factory=list)
    norms: list[FrameNorm] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    mask_paths: list[Path] = field(default_factory=list)  # per-frame rejection masks (optional)

    def __len__(self) -> int:
        return len(self.paths)

    def add(self, name: str, data: np.ndarray, norm: FrameNorm | None = None) -> Path:
        path = self._reserve(name, norm)
        np.save(path, np.ascontiguousarray(data, dtype=np.float32))
        return path

    def add_async(self, name: str, data: np.ndarray, norm: FrameNorm | None, executor) -> "Future[Path]":
        """Reserve the slot now (keeps frame order) and write on the executor."""

        path = self._reserve(name, norm)
        arr = np.ascontiguousarray(data, dtype=np.float32)

        def write() -> Path:
            np.save(path, arr)
            return path

        return executor.submit(write)

    def _reserve(self, name: str, norm: FrameNorm | None) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"reg_{len(self.paths):04d}.npy"
        self.paths.append(path)
        self.norms.append(norm or FrameNorm())
        self.names.append(name)
        return path

    def memmaps(self) -> list[np.memmap]:
        return [np.load(p, mmap_mode="r") for p in self.paths]

    def drop(self, index: int) -> None:
        """Remove a registered frame (e.g. failed a post-registration filter)."""

        path = self.paths.pop(index)
        self.norms.pop(index)
        self.names.pop(index)
        try:
            path.unlink()
        except OSError:
            pass

    def save_index(self) -> Path:
        index = {
            "shape": list(self.shape),
            "frames": [
                {"name": n, "path": str(p), "offset": np.asarray(f.offset, dtype=float).tolist(), "scale": np.asarray(f.scale, dtype=float).tolist(), "weight": f.weight}
                for n, p, f in zip(self.names, self.paths, self.norms)
            ],
        }
        out = self.directory / "index.json"
        out.write_text(json.dumps(index, indent=2), encoding="utf-8")
        return out

    def cleanup(self) -> None:
        for p in self.paths + self.mask_paths:
            try:
                p.unlink()
            except OSError:
                pass


@dataclass
class StackSettings:
    method: RejectionMethod = "winsorized"
    sigma_low: float = 3.0
    sigma_high: float = 3.0
    iterations: int = 3
    percentile_low: float = 0.2
    percentile_high: float = 0.1
    gesd_outliers: float = 0.3  # max fraction of frames that may be rejected per pixel
    gesd_significance: float = 0.05
    gesd_low_relax: float = 1.5  # >1 makes low (dark) outliers harder to reject
    use_weights: bool = True
    normalize: bool = True
    large_scale: bool = False  # grow high-rejections across dense structures (satellite/plane trails)
    large_scale_box: int = 15
    large_scale_density: float = 0.3
    large_scale_grow: int = 3


@dataclass
class StackResult:
    image: np.ndarray  # (C, H, W)
    rejected_low: int
    rejected_high: int
    total_samples: int
    bands: int
    coverage: np.ndarray | None = None  # (H, W) valid frames per pixel (channel 0)
    rejection: np.ndarray | None = None  # (H, W) rejected fraction per pixel, mean over channels
    crop_box: tuple[int, int, int, int] | None = None  # (y0, y1, x0, x1) applied to image/maps

    @property
    def rejection_fraction(self) -> float:
        return (self.rejected_low + self.rejected_high) / max(1, self.total_samples)


def _nanstd(x: torch.Tensor, mean: torch.Tensor, dim: int = 0) -> torch.Tensor:
    valid = ~torch.isnan(x)
    count = valid.sum(dim=dim).clamp_min(1)
    diff = torch.where(valid, x - mean, torch.zeros_like(x))
    return torch.sqrt((diff * diff).sum(dim=dim) / count.clamp_min(2).sub(1))


def _sigma_clip(x: torch.Tensor, lo: float, hi: float, iters: int, winsorize: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (clipped stack with NaN rejections, low-reject mask, high-reject mask)."""

    work = x.clone()
    rej_low = torch.zeros_like(x, dtype=torch.bool)
    rej_high = torch.zeros_like(x, dtype=torch.bool)
    for _ in range(max(1, iters)):
        center = torch.nanmedian(work, dim=0).values
        if winsorize:
            # Huber-style: start from MAD, then iterate winsorised mean/sigma (1.134 bias correction)
            std = 1.4826 * torch.nanmedian((work - center).abs(), dim=0).values
            for _w in range(3):
                lo_b, hi_b = center - 1.5 * std, center + 1.5 * std
                wins = torch.minimum(torch.maximum(work, lo_b), hi_b)
                center = torch.nanmean(wins, dim=0)
                std = 1.134 * _nanstd(wins, center)
        else:
            std = _nanstd(work, center)
        std = std.clamp_min(1e-12)
        dev = work - center
        new_low = dev < -lo * std
        new_high = dev > hi * std
        if not bool((new_low | new_high).any()):
            break
        rej_low |= new_low
        rej_high |= new_high
        work = torch.where(new_low | new_high, torch.full_like(work, float("nan")), work)
    return work, rej_low, rej_high


def _percentile_clip(x: torch.Tensor, lo: float, hi: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center = torch.nanmedian(x, dim=0).values
    denom = center.abs().clamp_min(1e-6)
    rel = (x - center) / denom
    rej_low = rel < -lo
    rej_high = rel > hi
    work = torch.where(rej_low | rej_high, torch.full_like(x, float("nan")), x)
    return work, rej_low, rej_high


_GESD_TABLE_CACHE: dict[tuple[int, int, float], np.ndarray] = {}


def gesd_critical_table(n_max: int, k: int, alpha: float) -> np.ndarray:
    """lambda[i-1, n] for i=1..k outliers among n samples (Rosner 1983); inf where undefined."""

    key = (n_max, k, alpha)
    cached = _GESD_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    from scipy.stats import t as student_t

    table = np.full((k, n_max + 1), np.inf, dtype=np.float32)
    for n in range(3, n_max + 1):
        for i in range(1, min(k, n - 2) + 1):
            dof = n - i - 1
            p = 1.0 - alpha / (2.0 * (n - i + 1))
            tv = student_t.ppf(p, dof)
            table[i - 1, n] = (n - i) * tv / np.sqrt((dof + tv * tv) * (n - i + 1))
    _GESD_TABLE_CACHE[key] = table
    return table


def _gesd_clip(x: torch.Tensor, max_frac: float, alpha: float, low_relax: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generalized Extreme Studentized Deviate test, vectorised over pixels.

    The most extreme sample is always the current min or max, so each pixel is sorted once
    and the test walks inward with running sums: O(N log N) sort + O(k) cheap steps.
    """

    n = x.shape[0]
    k = int(max(1, min(n - 3, np.floor(n * max_frac))))
    if n < 4 or k < 1:
        return x, torch.zeros_like(x, dtype=torch.bool), torch.zeros_like(x, dtype=torch.bool)
    flat = x.reshape(n, -1)
    pixels = flat.shape[1]
    device = x.device
    median = torch.nanmedian(flat, dim=0).values
    shifted = flat - median  # small magnitudes keep the running sums precise
    sorted_vals, sort_idx = torch.sort(shifted, dim=0)  # NaN sorts last
    sort_idx = sort_idx.to(torch.int32)
    valid = ~torch.isnan(shifted)
    count = valid.sum(dim=0).to(torch.float64)
    n_valid = count.to(torch.long)
    s1 = torch.where(valid, shifted, torch.zeros_like(shifted)).sum(dim=0, dtype=torch.float64)
    s2 = torch.where(valid, shifted * shifted, torch.zeros_like(shifted)).sum(dim=0, dtype=torch.float64)
    del shifted, valid
    cols = torch.arange(pixels, device=device)
    lo = torch.zeros(pixels, dtype=torch.long, device=device)
    hi = (n_valid - 1).clamp_min(0)
    stats = torch.full((k, pixels), float("-inf"), dtype=torch.float32, device=device)
    removed = torch.zeros((k, pixels), dtype=torch.int32, device=device)
    for i in range(k):
        active = count >= 3
        c = count.clamp_min(1)
        mean = s1 / c
        var = (s2 - s1 * s1 / c) / (c - 1).clamp_min(1)
        std = var.clamp_min(1e-24).sqrt()
        vlo = sorted_vals[lo, cols].to(torch.float64)
        vhi = sorted_vals[hi, cols].to(torch.float64)
        dlo = (mean - vlo) / std / low_relax
        dhi = (vhi - mean) / std
        pick_hi = dhi >= dlo
        r_i = torch.where(pick_hi, dhi, dlo)
        stats[i] = torch.where(active, r_i, torch.full_like(r_i, float("-inf"))).to(torch.float32)
        pos = torch.where(pick_hi, hi, lo)
        removed[i] = sort_idx[pos, cols]
        v = torch.where(pick_hi, vhi, vlo)
        s1 = torch.where(active, s1 - v, s1)
        s2 = torch.where(active, s2 - v * v, s2)
        count = torch.where(active, count - 1, count)
        hi = torch.where(active & pick_hi, hi - 1, hi).clamp_min(0)
        lo = torch.where(active & ~pick_hi, lo + 1, lo).clamp_max(n - 1)
    del sorted_vals
    table = torch.from_numpy(gesd_critical_table(n, k, alpha)).to(device)
    lam = table[:, n_valid.clamp(0, n)]  # (k, pixels)
    steps = torch.arange(1, k + 1, device=device).view(k, 1)
    n_out = torch.where(stats > lam, steps, torch.zeros_like(steps)).max(dim=0).values  # (pixels,)
    reject = torch.zeros((n, pixels), dtype=torch.bool, device=device)
    for i in range(k):
        sel = n_out > i
        if not bool(sel.any()):
            break
        reject[removed[i, sel].long(), cols[sel]] = True
    rej_low = (reject & (flat < median)).reshape(x.shape)
    rej_high = (reject & ~(flat < median)).reshape(x.shape)
    out = torch.where(reject, torch.full_like(flat, float("nan")), flat).reshape(x.shape)
    return out, rej_low, rej_high


def _large_scale_reject(x: torch.Tensor, rej_high: torch.Tensor, settings: StackSettings) -> torch.Tensor:
    """Where high-rejections are dense (a trail), also reject their dilated neighbourhood in that frame."""

    n, channels = x.shape[0], x.shape[1]
    flat = rej_high.reshape(n * channels, 1, *x.shape[2:]).to(x.dtype)
    box = settings.large_scale_box | 1
    density = F.avg_pool2d(flat, box, stride=1, padding=box // 2, count_include_pad=False)
    dense = (density > settings.large_scale_density).to(x.dtype)
    grow = 2 * settings.large_scale_grow + 1
    region = F.max_pool2d(dense, grow, stride=1, padding=grow // 2) > 0
    extra = region.reshape(x.shape) & ~torch.isnan(x) & ~rej_high
    return extra


@dataclass
class CombineOut:
    image: torch.Tensor  # (...) combined
    rej_low: torch.Tensor | None  # (N, ...) bool
    rej_high: torch.Tensor | None
    valid: torch.Tensor  # (N, ...) bool, input samples that were present


def combine_full(x: torch.Tensor, settings: StackSettings, weights: torch.Tensor | None = None) -> CombineOut:
    """Combine an (N, ...) tensor along axis 0, returning per-sample rejection masks."""

    present = ~torch.isnan(x)
    if settings.method == "median":
        return CombineOut(torch.nan_to_num(torch.nanmedian(x, dim=0).values, nan=0.0), None, None, present)
    if settings.method == "none":
        work, rl, rh = x, None, None
    elif settings.method in ("sigma", "winsorized"):
        work, rl, rh = _sigma_clip(x, settings.sigma_low, settings.sigma_high, settings.iterations, settings.method == "winsorized")
    elif settings.method == "percentile":
        work, rl, rh = _percentile_clip(x, settings.percentile_low, settings.percentile_high)
    elif settings.method == "gesd":
        work, rl, rh = _gesd_clip(x, settings.gesd_outliers, settings.gesd_significance, settings.gesd_low_relax)
    else:
        raise ValueError(f"Unknown rejection method {settings.method!r}")
    if settings.large_scale and rh is not None and x.ndim == 4:
        extra = _large_scale_reject(x, rh, settings)
        rh = rh | extra
        work = torch.where(extra, torch.full_like(work, float("nan")), work)
    valid = ~torch.isnan(work)
    if weights is None:
        out = torch.nanmean(work, dim=0)
    else:
        w = weights.to(x.dtype)
        if w.ndim == 1:
            w = w.view(-1, *([1] * (x.ndim - 1)))
        elif w.ndim == x.ndim - 1:
            w = w.unsqueeze(1)
        w = torch.broadcast_to(w, x.shape)
        num = torch.where(valid, work * w, torch.zeros_like(work)).sum(dim=0)
        den = (valid.to(x.dtype) * w).sum(dim=0)
        out = num / den.clamp_min(1e-12)
    # pixels rejected everywhere fall back to the plain median so nothing is left as NaN
    fallback = torch.nanmedian(x, dim=0).values
    out = torch.where(torch.isnan(out), fallback, out)
    out = torch.nan_to_num(out, nan=0.0)
    return CombineOut(out, rl, rh, present)


def combine_tensor(x: torch.Tensor, settings: StackSettings, weights: torch.Tensor | None = None) -> tuple[torch.Tensor, int, int]:
    """Combine an (N, ...) tensor along axis 0. Returns (image, n_rejected_low, n_rejected_high)."""

    res = combine_full(x, settings, weights)
    return res.image, int(res.rej_low.sum()) if res.rej_low is not None else 0, int(res.rej_high.sum()) if res.rej_high is not None else 0


def _apply_norm_maps(x: torch.Tensor, norms: list[FrameNorm], r0: int, r1: int, height: int) -> torch.Tensor:
    """Apply per-frame local normalisation maps to a band in place. x: (N, C, rows, W)."""

    n, channels, rows, width = x.shape
    for i, norm in enumerate(norms):
        if norm.scale_map is None or norm.offset_map is None:
            continue
        s_full = F.interpolate(torch.from_numpy(norm.scale_map).to(x.device).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
        o_full = F.interpolate(torch.from_numpy(norm.offset_map).to(x.device).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
        x[i] = x[i] * s_full[:, r0:r1, :] + o_full[:, r0:r1, :]
    return x


def stack_store(store: FrameStore, backend: Backend, settings: StackSettings, progress_cb: ProgressCallback = _noop_progress, label: str = "Stacking", reject_masks: bool = False) -> StackResult:
    """Stream the registered frames through the device in row bands and combine them.

    With ``reject_masks`` a (H, W) uint8 mask per frame (1 = rejected in any channel) is written next
    to the registered frames as ``rej_####.npy`` for drizzle to honour.
    """

    n = len(store)
    if n == 0:
        raise ValueError("No frames to stack")
    channels, height, width = store.shape
    maps = store.memmaps()
    mask_maps: list[np.memmap] = []
    if reject_masks:
        store.mask_paths = [p.with_name(p.name.replace("reg_", "rej_")) for p in store.paths]
        mask_maps = [np.lib.format.open_memmap(p, mode="w+", dtype=np.uint8, shape=(height, width)) for p in store.mask_paths]
    local_norm = any(f.scale_map is not None for f in store.norms)
    overhead = 10.0 if settings.method == "gesd" else 6.0
    if settings.large_scale:
        overhead += 2.0
    band = rows_per_band(n, width, channels, backend.budget_bytes, overhead=overhead)
    bands = math.ceil(height / band)
    out = np.empty((channels, height, width), dtype=np.float32)
    coverage = np.empty((height, width), dtype=np.uint16)
    rejection = np.empty((height, width), dtype=np.float32)

    offsets = torch.from_numpy(np.stack([norm_vec(f.offset, channels) for f in store.norms])).to(backend.device).view(n, channels, 1, 1)
    scales = torch.from_numpy(np.stack([norm_vec(f.scale, channels) for f in store.norms])).to(backend.device).view(n, channels, 1, 1)
    ref_offset = offsets[0]
    weights = torch.tensor([f.weight for f in store.norms], device=backend.device) if settings.use_weights else None

    host_buffers = [torch.empty((n, channels, band, width), dtype=torch.float32, pin_memory=backend.is_cuda)]
    if backend.is_cuda and bands > 1:
        host_buffers.append(torch.empty((n, channels, band, width), dtype=torch.float32, pin_memory=True))
    host_np_buffers = [buffer.numpy() for buffer in host_buffers]

    def stage_host_band(buffer_index: int, r0: int, r1: int) -> None:
        host_np = host_np_buffers[buffer_index]
        for i, mm in enumerate(maps):
            host_np[i, :, : r1 - r0, :] = mm[:, r0:r1, :]

    rej_low = rej_high = 0
    prefetch = ThreadPoolExecutor(max_workers=1) if len(host_buffers) > 1 else None
    pending_stage: Future | None = None
    try:
        for bi in range(bands):
            buffer_index = bi % len(host_buffers)
            r0 = bi * band
            r1 = min(height, r0 + band)
            rows = r1 - r0
            if pending_stage is not None:
                pending_stage.result()
                pending_stage = None
            else:
                stage_host_band(buffer_index, r0, r1)
            host = host_buffers[buffer_index]
            x = host[:, :, :rows, :].to(backend.device, non_blocking=True)
            if prefetch is not None and bi + 1 < bands:
                next_r0 = (bi + 1) * band
                next_r1 = min(height, next_r0 + band)
                pending_stage = prefetch.submit(stage_host_band, (bi + 1) % len(host_buffers), next_r0, next_r1)
            if local_norm:
                x = _apply_norm_maps(x, store.norms, r0, r1, height)
            elif settings.normalize:
                x = (x - offsets) * scales + ref_offset
            band_weights = weights
            if settings.use_weights and any(norm.weight_map is not None for norm in store.norms):
                base_weights = weights if weights is not None else torch.ones(n, device=backend.device)
                gy = torch.zeros(rows, device=backend.device) if height == 1 else 2.0 * torch.arange(r0, r1, device=backend.device) / (height - 1) - 1.0
                gx = torch.zeros(width, device=backend.device) if width == 1 else torch.linspace(-1.0, 1.0, width, device=backend.device)
                grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
                sample_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                spatial = []
                for norm in store.norms:
                    if norm.weight_map is None:
                        spatial.append(torch.ones((1, rows, width), device=backend.device))
                    else:
                        field = torch.as_tensor(norm.weight_map, dtype=torch.float32, device=backend.device)[None, None]
                        spatial.append(F.grid_sample(field, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)[0])
                band_weights = base_weights.view(n, 1, 1, 1) * torch.stack(spatial)
            res = combine_full(x, settings, band_weights)
            out[:, r0:r1, :] = res.image.cpu().numpy()
            coverage[r0:r1, :] = res.valid[:, 0].sum(dim=0).to(torch.int32).cpu().numpy().astype(np.uint16)
            rej_count = torch.zeros_like(res.valid[0], dtype=torch.float32)
            if res.rej_low is not None:
                rej_count += res.rej_low.sum(dim=0).to(torch.float32)
                rej_low += int(res.rej_low.sum())
            if res.rej_high is not None:
                rej_count += res.rej_high.sum(dim=0).to(torch.float32)
                rej_high += int(res.rej_high.sum())
            present = res.valid.sum(dim=0).to(torch.float32).clamp_min(1.0)
            rejection[r0:r1, :] = (rej_count / present).mean(dim=0).cpu().numpy()
            if mask_maps:
                rej_any = torch.zeros_like(res.valid, dtype=torch.bool)
                if res.rej_low is not None:
                    rej_any |= res.rej_low
                if res.rej_high is not None:
                    rej_any |= res.rej_high
                rej_frame = rej_any.any(dim=1).to(torch.uint8).cpu().numpy()  # (N, rows, W)
                for i, mm in enumerate(mask_maps):
                    mm[r0:r1, :] = rej_frame[i]
            progress_cb((bi + 1) / bands, f"{label} band {bi + 1}/{bands}")
    finally:
        if prefetch is not None:
            prefetch.shutdown(wait=True, cancel_futures=True)
    for mm in mask_maps:
        mm.flush()
    del mask_maps
    backend.empty_cache()
    return StackResult(out, rej_low, rej_high, n * channels * height * width, bands, coverage, rejection)


def estimate_norm(image: torch.Tensor, sample_stride: int = 8) -> tuple[float, float]:
    """Robust (median, MAD-sigma) of valid pixels, sub-sampled for speed."""

    sample = image[..., ::sample_stride, ::sample_stride].flatten()
    sample = sample[~torch.isnan(sample)]
    if sample.numel() == 0:
        return 0.0, 1.0
    med = torch.median(sample)
    mad = torch.median((sample - med).abs()) * 1.4826
    return float(med), float(mad.clamp_min(1e-9))
