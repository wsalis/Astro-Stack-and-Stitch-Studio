"""End-to-end pipeline: calibrate -> debayer -> analyse -> filter -> register -> weight -> stack
(-> local-normalised second pass) -> autocrop -> diagnostics -> (optional) MFDeconv."""

from __future__ import annotations

import json
import os
import shutil
import time
import copy
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal

import numpy as np
import torch

from . import __version__
from .backend import Backend, StatusCallback, select_backend, to_numpy, to_tensor
from .calibration import Masters, calibrate, load_masters
from .cosmetic import CosmeticSettings, CosmeticStats, cosmetic_correct
from .debayer import debayer
from .detection import Stars, analyse_frame, fwhm_grid, fwhm_grid_with_counts
from .diagnostics import apply_box, crop_box_for, effective_frames, stack_noise, write_frames_csv
from .drizzle import DrizzleAccumulator, DrizzleSettings, drizzle_frame, new_accumulator
from .io import FrameMeta, load_frame, save_fits
from .mfdeconv import MFDeconvResult, MFDeconvSettings, run_mfdeconv
from .normalization import LocalNormSettings, fit_norm_maps, reference_block_stats
from .platesolve import find_astap, solve_fits
from .quality import FilterSettings, StarSet, aperture_flux, compute_weights, field_weight_factors, sample_field_grid, select_photometry_stars, transparency
from .registration import Alignment, Interpolation, estimate_alignment, refine_alignment, source_coords_at, warp_to_reference
from .stacking import FrameNorm, FrameStore, StackResult, StackSettings, estimate_norm, norm_vec, stack_store

ProgressCallback = Callable[[float, str], None]
Weighting = Literal["psfsw", "psfsw+fwhm", "psfsw+field", "noise", "fwhm", "both", "none"]


def weighting_output_path(output: Path, mode: Weighting) -> Path:
    return output.with_name(f"{output.stem}_weight-{mode.replace('+', '-')}{output.suffix}")


def _noop_status(_: str) -> None:
    pass


def _noop_progress(_: float, __: str) -> None:
    pass


def blend_mfdeconv(master: np.ndarray, deconvolved: np.ndarray, strength: float) -> np.ndarray:
    """Blend the MFDeconv correction into the master; 0 is master, 1 is full deconvolution."""

    if not 0.0 <= strength <= 1.0:
        raise ValueError("MFDeconv blend must be between 0 and 1")
    master = np.asarray(master, dtype=np.float32)
    deconvolved = np.asarray(deconvolved, dtype=np.float32)
    if master.shape != deconvolved.shape:
        raise ValueError(f"MFDeconv/master shape mismatch: {deconvolved.shape} vs {master.shape}")
    if strength == 0.0:
        return master.copy()
    if strength == 1.0:
        return deconvolved.copy()
    return master + np.float32(strength) * (deconvolved - master)


@dataclass
class PipelineSettings:
    lights: list[Path]
    output: Path
    work_dir: Path | None = None
    bias: Path | None = None
    dark: Path | None = None
    flat: Path | None = None
    pedestal: float = 0.0
    cosmetic: Literal["auto", "on", "off"] = "auto"  # auto = on when no master dark
    cosmetic_settings: CosmeticSettings = field(default_factory=CosmeticSettings)
    debayer: Literal["auto", "on", "off"] = "auto"
    debayer_method: Literal["vng", "bilinear"] = "vng"
    bayer_pattern: str | None = None
    reference: Path | None = None
    detection_sigma: float = 5.0
    min_stars: int = 8
    weighting: Weighting = "psfsw"
    weightings: list[Weighting] = field(default_factory=list)
    filters: FilterSettings = field(default_factory=FilterSettings)
    interpolation: Interpolation = "lanczos3"
    refine_registration: bool = True
    local_norm: LocalNormSettings = field(default_factory=LocalNormSettings)
    refit_local_norm_per_weighting: bool = False
    stack: StackSettings = field(default_factory=StackSettings)
    autocrop: float = 0.0  # 0 = off, 1.0 = keep only pixels covered by every frame
    save_maps: bool = True  # coverage / rejection maps as FITS
    plate_solve: bool = False  # ASTAP-solve the master, drizzle and MFDeconv outputs
    astap_path: Path | None = None  # None = auto-detect
    drizzle: DrizzleSettings = field(default_factory=DrizzleSettings)
    mfdeconv: MFDeconvSettings | None = None
    mfdeconv_frames: int = 12
    mfdeconv_blend: float = 0.3
    keep_registered: bool = False
    device: Literal["auto", "cpu"] = "auto"
    vram_fraction: float = 0.75
    workers: int = 0  # CPU worker processes for star detection; 0 = all cores but one

    def __post_init__(self) -> None:
        if not np.isfinite(self.mfdeconv_blend) or not 0.0 <= self.mfdeconv_blend <= 1.0:
            raise ValueError("MFDeconv blend must be between 0 and 1")


@dataclass
class FrameInfo:
    index: int
    path: Path
    meta: FrameMeta
    stars: Stars
    fwhm: float
    noise: float
    background: float
    fwhm_grid: np.ndarray | None = None  # (cells, cells) median FWHM per sensor region, pre-registration
    alignment: Alignment | None = None
    weight: float = 1.0
    transparency: float = float("nan")
    phot_stars: int = 0
    flux_scaled: bool = False  # normalisation scale came from star photometry (else noise ratio)
    rejected_reason: str | None = None
    store_index: int | None = None
    fwhm_count_grid: np.ndarray | None = None

    @property
    def flipped(self) -> bool | None:
        return None if self.alignment is None else self.alignment.flipped


def select_mfdeconv_indices(infos: list[FrameInfo], count: int) -> list[int]:
    """Select registered, accepted frames by sharpness; stack weight breaks FWHM ties."""

    eligible = [f for f in infos if f.store_index is not None and f.rejected_reason is None]
    eligible.sort(key=lambda f: (not np.isfinite(f.fwhm), f.fwhm if np.isfinite(f.fwhm) else float("inf"), -f.weight))
    return [f.store_index for f in eligible[: max(2, count)] if f.store_index is not None]


@dataclass
class PipelineResult:
    output: Path
    stack: StackResult
    frames: list[FrameInfo]
    reference_index: int
    mfdeconv_output: Path | None = None
    mfdeconv: MFDeconvResult | None = None
    drizzle_output: Path | None = None
    report_path: Path | None = None
    csv_path: Path | None = None
    seconds: float = 0.0
    weighting_outputs: dict[str, Path] = field(default_factory=dict)


class _InlineExecutor:
    """Executor stand-in that runs tasks immediately (single-worker mode and tests)."""

    def submit(self, fn, *args, **kwargs) -> Future:
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # propagate through .result() like a real executor
            fut.set_exception(exc)
        return fut

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass


def _luma_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.shape[0] == 1:
        return image[0]
    return 0.2126 * image[0] + 0.7152 * image[1] + 0.0722 * image[2]


class StackingPipeline:
    def __init__(self, settings: PipelineSettings, status_cb: StatusCallback = _noop_status, progress_cb: ProgressCallback = _noop_progress) -> None:
        self.settings = settings
        self.status = status_cb
        self.progress = progress_cb
        self.backend: Backend = select_backend(settings.device, settings.vram_fraction, status_cb)
        self.masters: Masters = load_masters(self.backend, settings.bias, settings.dark, settings.flat, settings.pedestal)
        self.cosmetic_enabled = settings.cosmetic == "on" or (settings.cosmetic == "auto" and settings.dark is None)
        self.cosmetic_stats = CosmeticStats()
        self._cosmetic_frames = 0
        self._cancel = False
        cores = os.cpu_count() or 2
        # beyond ~8 workers detection already outruns the GPU main thread, and large pools are flaky on Windows
        self.cpu_workers = settings.workers if settings.workers > 0 else max(1, min(8, cores - 1))
        self.io_workers = int(min(4, max(1, self.cpu_workers)))
        self.phot: StarSet | None = None
        status_cb(f"Workers: {self.cpu_workers} detection process(es), {self.io_workers} loader thread(s)")

    def cancel(self) -> None:
        self._cancel = True

    def _check_cancel(self) -> None:
        if self._cancel:
            raise RuntimeError("Cancelled")

    def _detection_pool(self):
        if self.cpu_workers <= 1:
            return _InlineExecutor()
        return ProcessPoolExecutor(max_workers=self.cpu_workers)

    # ------------------------------------------------------------------ loading

    def _prefetch(self, paths: Iterable[Path], loader: ThreadPoolExecutor) -> Iterator[tuple[Path, np.ndarray, FrameMeta]]:
        """Yield frames in order while the next few decode on loader threads."""

        paths = list(paths)
        pending: list[tuple[Path, Future]] = []
        nxt = 0
        while nxt < len(paths) or pending:
            while nxt < len(paths) and len(pending) <= self.io_workers:
                pending.append((paths[nxt], loader.submit(load_frame, paths[nxt])))
                nxt += 1
            path, fut = pending.pop(0)
            data, meta = fut.result()
            yield path, data, meta

    def _calibrate_data(self, data: np.ndarray, meta: FrameMeta, path: Path, skip_debayer: bool = False) -> torch.Tensor:
        tensor = to_tensor(data, self.backend)
        tensor = calibrate(tensor, self.masters)
        pattern = self.settings.bayer_pattern or meta.bayer_pattern
        want = self.settings.debayer == "on" or (self.settings.debayer == "auto" and meta.is_cfa)
        if self.cosmetic_enabled:
            tensor, stats = cosmetic_correct(tensor, self.settings.cosmetic_settings, cfa=want and tensor.shape[0] == 1)
            self.cosmetic_stats += stats
            self._cosmetic_frames += 1
        if want and tensor.shape[0] == 1 and not skip_debayer:
            if not pattern:
                raise ValueError(f"{path.name}: debayer requested but no Bayer pattern known (use --bayer)")
            tensor = debayer(tensor, pattern, self.settings.debayer_method)
        return tensor

    def _cfa_pattern_for(self, meta: FrameMeta) -> str | None:
        want = self.settings.debayer == "on" or (self.settings.debayer == "auto" and meta.is_cfa)
        return (self.settings.bayer_pattern or meta.bayer_pattern) if want else None

    def _load_calibrated(self, path: Path) -> tuple[torch.Tensor, FrameMeta]:
        data, meta = load_frame(path)
        return self._calibrate_data(data, meta, path), meta

    # ------------------------------------------------------------------ pass 1: analyse

    def analyse(self) -> list[FrameInfo]:
        paths = self.settings.lights
        total = len(paths)
        infos: list[FrameInfo | None] = [None] * total
        max_inflight = max(2, self.cpu_workers * 2)

        inline = _InlineExecutor()
        state = {"broken": False}

        def collect(item: tuple[int, Path, FrameMeta, tuple[int, int], Future]) -> None:
            i, path, meta, shape, fut = item
            try:
                stars, bg, noise = fut.result()
            except BrokenProcessPool:
                if not state["broken"]:
                    state["broken"] = True
                    self.status("Detection worker pool failed (Windows handle error); continuing in-process for the remaining frames")
                tensor = self._calibrate_data(load_frame(path)[0], meta, path)
                stars, bg, noise = analyse_frame(to_numpy(_luma_tensor(tensor)), self.settings.detection_sigma)
                del tensor
            fwhm = stars.median_fwhm()
            grid, grid_counts = fwhm_grid_with_counts(stars, shape)
            info = FrameInfo(i, path, meta, stars, fwhm, noise, bg, fwhm_grid=grid, fwhm_count_grid=grid_counts)
            if stars.total_detected < self.settings.min_stars:
                info.rejected_reason = f"only {stars.total_detected} stars"
            infos[i] = info
            self.status(f"[{i + 1}/{total}] {path.name}: {stars.total_detected} stars, FWHM {fwhm:.2f}px, noise {noise:.3g}, sky {bg:.0f}")
            self.progress(0.2 * (i + 1) / total, f"Analysing {i + 1}/{total}")

        with ThreadPoolExecutor(self.io_workers) as loader, self._detection_pool() as pool:
            pending: list[tuple[int, Path, FrameMeta, tuple[int, int], Future]] = []
            for i, (path, data, meta) in enumerate(self._prefetch(paths, loader)):
                self._check_cancel()
                tensor = self._calibrate_data(data, meta, path)
                lum = to_numpy(_luma_tensor(tensor))
                shape = (int(lum.shape[0]), int(lum.shape[1]))
                del tensor, data
                executor = inline if state["broken"] else pool
                try:
                    fut = executor.submit(analyse_frame, lum, self.settings.detection_sigma)
                except BrokenProcessPool:
                    state["broken"] = True
                    fut = inline.submit(analyse_frame, lum, self.settings.detection_sigma)
                pending.append((i, path, meta, shape, fut))
                if i == 0 and self.cosmetic_enabled:
                    self.status(f"Cosmetic correction: {self.cosmetic_stats.hot} hot / {self.cosmetic_stats.cold} cold pixels fixed in first frame")
                while len(pending) >= max_inflight:
                    collect(pending.pop(0))
            for item in pending:
                self._check_cancel()
                collect(item)
        self.backend.empty_cache()
        return [info for info in infos if info is not None]

    # ------------------------------------------------------------------ filters / reference / weights

    def apply_prefilters(self, infos: list[FrameInfo]) -> None:
        """Star-count, background and FWHM filters (relative to the session median)."""

        f = self.settings.filters
        if not f.enabled:
            return
        usable = [i for i in infos if i.rejected_reason is None]
        if len(usable) < 3:
            return
        med_stars = float(np.median([i.stars.total_detected for i in usable]))
        med_bg = float(np.median([i.background for i in usable]))
        med_fwhm = float(np.nanmedian([i.fwhm for i in usable if np.isfinite(i.fwhm)] or [np.nan]))
        for i in usable:
            reasons = []
            if f.stars_min_ratio > 0 and i.stars.total_detected < f.stars_min_ratio * med_stars:
                reasons.append(f"stars {i.stars.total_detected} < {f.stars_min_ratio:.0%} of median {med_stars:.0f}")
            if f.background_max_ratio > 0 and med_bg > 0 and i.background > f.background_max_ratio * med_bg:
                reasons.append(f"sky {i.background:.0f} > {f.background_max_ratio:.1f}x median {med_bg:.0f}")
            if f.fwhm_max_ratio > 0 and np.isfinite(med_fwhm) and np.isfinite(i.fwhm) and i.fwhm > f.fwhm_max_ratio * med_fwhm:
                reasons.append(f"FWHM {i.fwhm:.2f} > {f.fwhm_max_ratio:.1f}x median {med_fwhm:.2f}")
            if reasons:
                i.rejected_reason = "filter: " + "; ".join(reasons)
                self.status(f"Excluded {i.path.name}: {i.rejected_reason}")

    def tilt_summary(self, infos: list[FrameInfo]) -> dict:
        """Session FWHM map in sensor coordinates: constant pattern (tilt / curvature) vs frame-to-frame scatter."""

        grids = [f.fwhm_grid for f in infos if f.rejected_reason is None and f.fwhm_grid is not None and np.isfinite(f.fwhm_grid).sum() >= 4]
        if not grids:
            return {}
        g = np.stack(grids).astype(np.float64)  # (N, cells, cells)
        per_frame_med = np.nanmedian(g.reshape(len(g), -1), axis=1)
        rel = g / per_frame_med[:, None, None]  # each frame's map relative to its own median FWHM
        median_px = np.nanmedian(g, axis=0)
        pattern = np.nanmedian(rel, axis=0)  # the part shared by every frame
        scatter = float(np.nanmedian(np.nanstd(rel - pattern[None], axis=0))) if len(g) >= 3 else float("nan")
        finite = np.isfinite(pattern)
        soft = np.unravel_index(np.nanargmax(np.where(finite, pattern, -np.inf)), pattern.shape)
        sharp = np.unravel_index(np.nanargmin(np.where(finite, pattern, np.inf)), pattern.shape)
        ratio = float(pattern[soft] / pattern[sharp])

        cells = pattern.shape[0]
        rows_lbl = ["top", "upper-mid", "lower-mid", "bottom"] if cells == 4 else [str(r) for r in range(cells)]
        cols_lbl = ["left", "centre-left", "centre-right", "right"] if cells == 4 else [str(c) for c in range(cells)]

        def name(rc: tuple) -> str:
            return f"{rows_lbl[rc[0]]}/{cols_lbl[rc[1]]}"

        if ratio - 1.0 < 0.08:
            verdict = "field is flat within 8%; no significant tilt"
        elif not np.isfinite(scatter) or scatter < (ratio - 1.0) / 3.0:
            verdict = "pattern is CONSTANT across frames -> mechanical tilt or field curvature (per-region PSF would help; per-frame weighting would not)"
        else:
            verdict = "pattern VARIES between frames -> focus/seeing drift or shifting tilt (per-region frame weighting would help)"

        self.status(f"FWHM map (sensor coords, session median px, {cells}x{cells}, top row first; {len(g)} frames):")
        for r in range(cells):
            self.status("    " + "  ".join("  n/a" if not np.isfinite(v) else f"{v:5.2f}" for v in median_px[r]))
        self.status(
            f"Tilt check: softest {name(soft)} {median_px[soft]:.2f} px vs sharpest {name(sharp)} {median_px[sharp]:.2f} px "
            f"(ratio {ratio:.2f}); frame-to-frame scatter " + ("n/a" if not np.isfinite(scatter) else f"+/-{100 * scatter:.0f}%") + f" -> {verdict}"
        )
        return {
            "cells": cells,
            "frames": len(g),
            "median_fwhm_px": [[None if not np.isfinite(v) else round(float(v), 3) for v in row] for row in median_px],
            "constant_pattern": [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in pattern],
            "soft_to_sharp_ratio": round(ratio, 4),
            "softest_cell": name(soft),
            "sharpest_cell": name(sharp),
            "frame_to_frame_scatter": None if not np.isfinite(scatter) else round(scatter, 4),
            "verdict": verdict,
        }

    def choose_reference(self, infos: list[FrameInfo]) -> int:
        if self.settings.reference is not None:
            for info in infos:
                if info.path.resolve() == Path(self.settings.reference).resolve():
                    return info.index
            raise ValueError(f"Reference {self.settings.reference} is not among the lights")
        usable = [f for f in infos if f.rejected_reason is None and np.isfinite(f.fwhm)]
        if not usable:
            raise ValueError("No usable frames (too few stars detected everywhere)")
        median_stars = float(np.median([f.stars.total_detected for f in usable]))
        pool = [f for f in usable if f.stars.total_detected >= 0.8 * median_stars] or usable
        # haze lowers noise as well as star flux, so a noise term alone would favour hazy frames;
        # the summed flux of the brightest stars is a transparency proxy available before photometry
        bright = {f.index: float(np.sum(np.sort(f.stars.flux)[::-1][:30])) for f in pool}
        top_bright = float(np.percentile(list(bright.values()), 90))
        clear = [f for f in pool if bright[f.index] >= 0.9 * top_bright] or pool
        med_noise = float(np.median([u.noise for u in usable]))
        best = min(clear, key=lambda f: f.fwhm * (1.0 + 0.1 * f.noise / max(1e-9, med_noise)))
        return best.index

    def _set_field_weight_maps(self, infos: list[FrameInfo], store: FrameStore) -> None:
        usable = [f for f in infos if f.rejected_reason is None and f.store_index is not None]
        if not usable:
            return
        grids = [f.fwhm_grid if f.fwhm_grid is not None else np.full((4, 4), np.nan) for f in usable]
        counts = [f.fwhm_count_grid if f.fwhm_count_grid is not None else np.zeros((4, 4), dtype=np.int32) for f in usable]
        factors = field_weight_factors(grids, confidence_counts=counts)
        _, height, width = store.shape
        map_h, map_w = min(64, height), min(64, width)
        ys, xs = np.meshgrid(np.linspace(0, height - 1, map_h), np.linspace(0, width - 1, map_w), indexing="ij")
        target_x = torch.from_numpy(xs.astype(np.float32))
        target_y = torch.from_numpy(ys.astype(np.float32))
        factor_ranges: list[np.ndarray] = []
        for info, factor in zip(usable, factors):
            if info.alignment is None or info.store_index is None:
                continue
            sx, sy = source_coords_at(info.alignment, target_x, target_y)
            weight_map = sample_field_grid(factor, sx.cpu().numpy(), sy.cpu().numpy(), (height, width))
            store.norms[info.store_index].weight_map = weight_map
            factor_ranges.append(factor[np.isfinite(factor)])
        if factor_ranges:
            values = np.concatenate(factor_ranges)
            self.status(f"Field weighting factors: {values.min():.2f}-{values.max():.2f} (persistent sensor pattern removed)")

    def compute_weights(self, infos: list[FrameInfo], store: FrameStore, mode: str | None = None) -> None:
        usable = [f for f in infos if f.rejected_reason is None and f.store_index is not None]
        if not usable:
            return
        mode = mode or self.settings.weighting
        w = compute_weights(
            mode,
            np.array([f.noise for f in usable]),
            np.array([f.fwhm for f in usable]),
            np.array([f.transparency for f in usable]),
        )
        for f, wi in zip(usable, w):
            f.weight = float(wi)
            store.norms[f.store_index].weight = float(wi)
            store.norms[f.store_index].weight_map = None
        if mode == "psfsw+field":
            self._set_field_weight_maps(infos, store)
        unknown = sum(1 for f in usable if not np.isfinite(f.transparency))
        note = f" ({unknown} frame(s) without transparency fell back to noise-only)" if mode.startswith("psfsw") and unknown else ""
        self.status(f"Weights ({mode}): min {w.min():.2f}, median {np.median(w):.2f}, max {w.max():.2f}{note}")

    def apply_transparency_filter(self, infos: list[FrameInfo], store: FrameStore) -> None:
        # transparency was measured against the reference; re-express it against the session median so a
        # hazy reference frame cannot make every other frame look "too bright"
        vals = np.array([f.transparency for f in infos if f.rejected_reason is None and np.isfinite(f.transparency)])
        if vals.size >= 3:
            med = float(np.median(vals))
            if med > 0:
                for f in infos:
                    if np.isfinite(f.transparency):
                        f.transparency = f.transparency / med
        if not self.settings.filters.enabled:
            return
        thr = self.settings.filters.transparency_min
        if thr <= 0:
            return
        dropped: list[FrameInfo] = []
        for f in infos:
            if f.rejected_reason is None and f.store_index is not None and np.isfinite(f.transparency) and f.transparency < thr:
                f.rejected_reason = f"filter: transparency {f.transparency:.2f} < {thr:.2f}"
                self.status(f"Excluded {f.path.name}: {f.rejected_reason}")
                dropped.append(f)
        for f in sorted(dropped, key=lambda d: -(d.store_index or 0)):
            store.drop(f.store_index)  # type: ignore[arg-type]
            for g in infos:
                if g.store_index is not None and g.store_index > f.store_index:  # type: ignore[operator]
                    g.store_index -= 1
            f.store_index = None

    # ------------------------------------------------------------------ pass 2: register

    def register(self, infos: list[FrameInfo], reference_index: int, store: FrameStore) -> None:
        ref = infos[reference_index]
        usable = [f for f in infos if f.rejected_reason is None]
        total = len(usable)
        order = [ref] + [f for f in usable if f.index != reference_index]
        by_path = {f.path: f for f in order}
        self.phot = select_photometry_stars(ref.stars, (store.shape[1], store.shape[2]), ref.fwhm)
        self.status(f"Photometry on {len(self.phot)} reference stars (aperture r={self.phot.radius}px)")
        ref_norm: tuple[np.ndarray, np.ndarray] | None = None
        ref_flux_ch: list[np.ndarray] | None = None
        writes: list[Future] = []
        with ThreadPoolExecutor(self.io_workers) as loader, ThreadPoolExecutor(2) as writer:
            for n, (path, data, meta) in enumerate(self._prefetch([f.path for f in order], loader)):
                self._check_cancel()
                info = by_path[path]
                if info.index == reference_index:
                    info.alignment = Alignment.identity()
                else:
                    try:
                        align = estimate_alignment(info.stars, ref.stars)
                        if self.settings.refine_registration:
                            align = refine_alignment(align, info.stars, ref.stars)
                        info.alignment = align
                    except Exception as exc:  # astroalign raises several types on failure
                        info.rejected_reason = f"alignment failed: {exc}"
                        self.status(f"Skipping {info.path.name}: {info.rejected_reason}")
                        continue
                tensor = self._calibrate_data(data, meta, path)
                warped = warp_to_reference(tensor, info.alignment, mode=self.settings.interpolation)
                lum = _luma_tensor(warped)
                channels = warped.shape[0]
                norms = [estimate_norm(warped[c]) for c in range(channels)]
                med = np.array([m for m, _ in norms], dtype=np.float32)
                mad = np.array([s for _, s in norms], dtype=np.float32)
                if ref_norm is None:
                    ref_norm = (med, mad)
                flux = aperture_flux(lum, self.phot)
                if self.phot.ref_flux is None:
                    self.phot.ref_flux = flux
                info.transparency, info.phot_stars = transparency(flux, self.phot.ref_flux)
                # flux scale from star photometry per channel (PI LocalNormalization >= 1.8.9 style): puts stars
                # AND nebula of every frame on the reference's flux scale; the noise ratio is only a fallback
                flux_ch = [aperture_flux(warped[c], self.phot) for c in range(channels)]
                if ref_flux_ch is None:
                    ref_flux_ch = flux_ch
                t_ch = np.array([transparency(f, r)[0] for f, r in zip(flux_ch, ref_flux_ch)], dtype=np.float32)
                noise_scale = ref_norm[1] / np.maximum(mad, 1e-12)
                scale = np.where(np.isfinite(t_ch) & (t_ch > 0), 1.0 / np.clip(t_ch, 1 / 3, 3), noise_scale).astype(np.float32)
                info.flux_scaled = bool(np.isfinite(t_ch).all())
                info.store_index = len(store)
                writes.append(store.add_async(info.path.name, to_numpy(warped), FrameNorm(offset=med, scale=scale, weight=1.0), writer))
                while len(writes) > 4:  # bound host memory held by queued writes
                    writes.pop(0).result()
                a = info.alignment
                refined = f" poly rms {a.residual_rms:.2f}px/{a.refined_matches}" if a.inverse_poly is not None else ""
                flip = " FLIP" if a.flipped else ""
                tr = f" T={info.transparency:.2f}" if np.isfinite(info.transparency) else ""
                self.status(f"Registered {info.path.name}: shift ({a.shift[0]:+.1f}, {a.shift[1]:+.1f}) rot {a.rotation_deg:+.2f}°{flip} matched {a.matched}{refined}{tr}")
                self.progress(0.2 + 0.35 * (n + 1) / total, f"Registering {n + 1}/{total}")
                del tensor, warped, lum, data
            for fut in writes:
                fut.result()
        self.backend.empty_cache()

    # ------------------------------------------------------------------ local normalisation

    def _fit_local_norm(self, store: FrameStore, reference: np.ndarray, progress_start: float = 0.68, progress_span: float = 0.07) -> None:
        ln = self.settings.local_norm
        ref_med, ref_mad, ref_frac = reference_block_stats(reference, self.backend, ln.block)
        n = len(store)
        for i, path in enumerate(store.paths):
            self._check_cancel()
            frame = torch.from_numpy(np.load(path)).to(self.backend.device)
            scale_map, offset_map = fit_norm_maps(frame, ref_med, ref_mad, ref_frac, ln, scale=store.norms[i].scale)
            store.norms[i].scale_map = scale_map
            store.norms[i].offset_map = offset_map
            del frame
            self.progress(progress_start + progress_span * (i + 1) / n, f"Local normalisation {i + 1}/{n}")
        self.backend.empty_cache()

    # ------------------------------------------------------------------ run

    def _flip_summary(self, infos: list[FrameInfo]) -> dict:
        reg = [f for f in infos if f.alignment is not None and f.rejected_reason is None]
        flipped = [f for f in reg if f.alignment.flipped]  # type: ignore[union-attr]
        normal = [f for f in reg if not f.alignment.flipped]  # type: ignore[union-attr]

        def side(frames: list[FrameInfo]) -> dict:
            return {"frames": len(frames), "median_fwhm": float(np.nanmedian([f.fwhm for f in frames])) if frames else None, "median_noise": float(np.median([f.noise for f in frames])) if frames else None}

        summary = {"flipped_frames": len(flipped), "normal_frames": len(normal), "flipped": side(flipped), "normal": side(normal)}
        if flipped:
            self.status(f"Meridian flip detected: {len(flipped)} of {len(reg)} frames are rotated ~180° (FWHM {summary['flipped']['median_fwhm']:.2f} vs {summary['normal']['median_fwhm']:.2f} px)")
        return summary

    def run(self) -> PipelineResult:
        t0 = time.perf_counter()
        s = self.settings
        if not s.lights:
            raise ValueError("No light frames given")
        output_base = s.output
        selected_weightings = list(dict.fromkeys(s.weightings or [s.weighting]))
        if s.weighting in selected_weightings:
            selected_weightings.remove(s.weighting)
        selected_weightings.insert(0, s.weighting)
        s.weightings = selected_weightings
        if len(selected_weightings) > 1:
            s.output = weighting_output_path(output_base, s.weighting)
        work_dir = s.work_dir or (s.output.parent / "_gpustacker_work")
        infos = self.analyse()
        self.apply_prefilters(infos)
        tilt = self.tilt_summary(infos)
        ref_index = self.choose_reference(infos)
        self.status(f"Reference frame: {infos[ref_index].path.name}")

        probe, _ = self._load_calibrated(infos[ref_index].path)
        shape = tuple(int(v) for v in probe.shape)
        del probe
        store = FrameStore(work_dir, shape)  # type: ignore[arg-type]
        try:
            self.register(infos, ref_index, store)
            if len(store) == 0:
                raise ValueError("No frames could be registered")
            self.apply_transparency_filter(infos, store)
            if len(store) == 0:
                raise ValueError("All frames were excluded by the transparency filter")
            self.compute_weights(infos, store)
            flips = self._flip_summary(infos)
            store.save_index()

            two_pass = s.local_norm.enabled and len(store) >= 3
            want_masks = s.drizzle.enabled or s.mfdeconv is not None
            self.status(f"Stacking {len(store)} frames with {s.stack.method} rejection" + (" (pass 1 of 2, global normalisation)" if two_pass else ""))
            span = 0.13 if two_pass else 0.3
            result = stack_store(store, self.backend, s.stack, lambda f, m: self.progress(0.55 + span * f, m), label="Stacking", reject_masks=want_masks and not two_pass)
            if two_pass:
                self.status("Fitting local normalisation maps against the first-pass stack")
                self._fit_local_norm(store, result.image)
                self.status(f"Stacking pass 2 of 2 with local normalisation ({s.local_norm.block}px blocks)")
                result = stack_store(store, self.backend, s.stack, lambda f, m: self.progress(0.75 + 0.1 * f, m), label="Stacking (local norm)", reject_masks=want_masks)

            crop_items: list[tuple] = []
            if s.autocrop > 0 and result.coverage is not None:
                box = crop_box_for(result.coverage, len(store), s.autocrop)
                if box is None:
                    self.status("Autocrop: no region reaches the requested depth; output left uncropped")
                else:
                    y0, y1, x0, x1 = box
                    h, w = result.coverage.shape
                    result.image = apply_box(result.image, box)
                    result.coverage = apply_box(result.coverage, box)
                    result.rejection = apply_box(result.rejection, box) if result.rejection is not None else None
                    result.crop_box = box
                    crop_items = [("CROPX0", x0, "Autocrop left edge in registered frame"), ("CROPY0", y0, "Autocrop top edge"), ("CROPDPTH", float(s.autocrop), "Autocrop depth fraction")]
                    self.status(f"Autocrop {s.autocrop:.0%} depth: {x1 - x0}x{y1 - y0} px, trimmed left {x0} right {w - x1} top {y0} bottom {h - y1}")

            exposures = [f.meta.exposure or 0.0 for f in infos if f.rejected_reason is None]
            weights = [f.weight for f in infos if f.rejected_reason is None and f.store_index is not None]
            noise = stack_noise(result.image)
            med_frame_noise = float(np.median([f.noise for f in infos if f.rejected_reason is None])) if exposures else float("nan")
            n_eff = effective_frames(weights)
            header_items = [
                ("NFRAMES", len(store), "Frames stacked"),
                ("NEFF", round(n_eff, 2), "Effective frames (sum w)^2/sum w^2"),
                ("EXPTOTAL", float(sum(exposures)), "Total exposure [s]"),
                ("STACKMTH", s.stack.method, "Rejection method"),
                ("REJFRAC", round(result.rejection_fraction, 6), "Rejected sample fraction"),
                ("WEIGHTMD", s.weighting, "Frame weighting"),
                ("LOCNORM", bool(two_pass), "Local normalisation applied"),
                ("INTERP", s.interpolation, "Registration interpolation"),
                ("BGNOISE", round(noise, 4), "Background noise of stack [ADU]"),
                ("GPUSTACK", __version__, "GPUStacker version"),
            ] + crop_items
            if exposures and exposures[0]:
                header_items.append(("EXPTIME", float(sum(exposures)), "Total exposure [s]"))
            save_fits(s.output, result.image, header_items, infos[ref_index].meta.header)
            gain = med_frame_noise / noise if noise > 0 and np.isfinite(med_frame_noise) else float("nan")
            self.status(f"Saved stack -> {s.output} (rejected {result.rejection_fraction * 100:.2f}%, noise {noise:.3g} ADU = {gain:.1f}x better than a single sub, N_eff {n_eff:.1f})")
            self._plate_solve(s.output)
            if s.save_maps and result.coverage is not None:
                cov_path = s.output.with_name(s.output.stem + "_coverage" + s.output.suffix)
                save_fits(cov_path, result.coverage.astype(np.float32), [("MAPTYPE", "coverage", "Valid frames per pixel")] + crop_items)
                rej_path = s.output.with_name(s.output.stem + "_rejection" + s.output.suffix)
                save_fits(rej_path, result.rejection, [("MAPTYPE", "rejection", "Rejected sample fraction per pixel")] + crop_items)
                self.status(f"Saved maps -> {cov_path.name}, {rej_path.name}")

            weighting_outputs = {s.weighting: s.output}
            if len(selected_weightings) > 1:
                modes = selected_weightings[1:]
                base_norms = copy.deepcopy(store.norms)
                base_frame_weights = [f.weight for f in infos]
                for variant_index, mode in enumerate(modes):
                    self.compute_weights(infos, store, mode)
                    interval_start = 0.90 + 0.09 * variant_index / max(1, len(modes))
                    interval_span = 0.09 / max(1, len(modes))
                    if two_pass:
                        if s.refit_local_norm_per_weighting:
                            for norm in store.norms:
                                norm.scale_map = None
                                norm.offset_map = None
                            first = stack_store(
                                store,
                                self.backend,
                                s.stack,
                                lambda f, m, start=interval_start, span=interval_span: self.progress(start + span * 0.4 * f, f"{mode}: {m}"),
                                label=f"Stacking {mode} (pass 1)",
                            )
                            self.status(f"Fitting local normalisation for {mode}")
                            self._fit_local_norm(store, first.image, interval_start + interval_span * 0.4, interval_span * 0.2)
                        else:
                            store.norms = copy.deepcopy(base_norms)
                            self.status(f"Reusing primary local normalisation maps for {mode}")
                        variant = stack_store(
                            store,
                            self.backend,
                            s.stack,
                            lambda f, m, start=interval_start, span=interval_span: self.progress(start + span * f, f"{mode}: {m}"),
                            label=f"Stacking {mode} ({'pass 2' if s.refit_local_norm_per_weighting else 'local norm'})",
                        )
                    else:
                        variant = stack_store(
                            store,
                            self.backend,
                            s.stack,
                            lambda f, m, start=interval_start, span=interval_span: self.progress(start + span * f, f"{mode}: {m}"),
                            label=f"Stacking {mode}",
                        )
                    if result.crop_box is not None:
                        variant.image = apply_box(variant.image, result.crop_box)
                    variant_noise = stack_noise(variant.image)
                    variant_weights = [f.weight for f in infos if f.rejected_reason is None and f.store_index is not None]
                    variant_header = [item for item in header_items if item[0] not in ("NEFF", "REJFRAC", "WEIGHTMD", "BGNOISE")]
                    variant_header.extend([
                        ("NEFF", round(effective_frames(variant_weights), 2), "Effective frames (sum w)^2/sum w^2"),
                        ("REJFRAC", round(variant.rejection_fraction, 6), "Rejected sample fraction"),
                        ("WEIGHTMD", mode, "Frame weighting"),
                        ("BGNOISE", round(variant_noise, 4), "Background noise of stack [ADU]"),
                    ])
                    variant_path = weighting_output_path(output_base, mode)
                    save_fits(variant_path, variant.image, variant_header, infos[ref_index].meta.header)
                    weighting_outputs[mode] = variant_path
                    self.status(f"Saved {mode} master -> {variant_path}")
                    self._plate_solve(variant_path)
                store.norms = base_norms
                for info, weight in zip(infos, base_frame_weights):
                    info.weight = weight

            mf_out = None
            mf_result = None
            drz_out = None
            if s.drizzle.enabled:
                drz_out = self._run_drizzle(store, infos, header_items, result.crop_box)
            if s.mfdeconv is not None:
                mf_out, mf_result = self._run_mfdeconv(store, infos, s.mfdeconv, header_items, result.crop_box, result.image)
            csv_path = self._write_csv(infos)
            report = self._write_report(infos, ref_index, result, mf_result, flips, noise, n_eff, drz_out, tilt, weighting_outputs)
            self.progress(1.0, "Done")
            return PipelineResult(s.output, result, infos, ref_index, mf_out, mf_result, drz_out, report, csv_path, time.perf_counter() - t0, weighting_outputs)
        finally:
            if not s.keep_registered:
                store.cleanup()
                shutil.rmtree(work_dir, ignore_errors=True)

    def _run_mfdeconv(self, store: FrameStore, infos: list[FrameInfo], settings: MFDeconvSettings, header_items: list, crop_box, master_image: np.ndarray) -> tuple[Path, MFDeconvResult]:
        by_name = {f.path.name: f for f in infos}
        order = select_mfdeconv_indices(infos, self.settings.mfdeconv_frames)
        frames: list[np.ndarray] = []
        variance_frames: list[np.ndarray] = []
        variance_scales: list[np.ndarray] = []
        rejection_masks: list[np.ndarray] = []
        gains: list[float | None] = []
        rns: list[float | None] = []
        fwhms: list[float | None] = []
        ref = store.norms[0]
        channels = store.shape[0]
        ref_off = norm_vec(ref.offset, channels)[:, None, None]
        for i in order:
            arr = np.load(store.paths[i]).astype(np.float32)
            norm = store.norms[i]
            variance_frame = arr
            rejection = np.load(store.mask_paths[i]).astype(bool) if store.mask_paths else None
            arr = (arr - norm_vec(norm.offset, channels)[:, None, None]) * norm_vec(norm.scale, channels)[:, None, None] + ref_off
            if crop_box is not None:
                arr = apply_box(arr, crop_box)
                variance_frame = apply_box(variance_frame, crop_box)
                rejection = apply_box(rejection, crop_box) if rejection is not None else None
            frames.append(arr)
            variance_frames.append(variance_frame)
            variance_scales.append(norm_vec(norm.scale, channels))
            if rejection is not None:
                rejection_masks.append(rejection)
            meta = by_name[store.names[i]].meta
            gains.append(meta.gain)
            rns.append(meta.read_noise)
            fwhms.append(by_name[store.names[i]].fwhm)
        if fwhms:
            self.status(f"MFDeconv on {len(frames)} sharpest eligible frames (FWHM {min(fwhms):.2f}-{max(fwhms):.2f}px; stack weights unchanged)")
        else:
            raise ValueError("MFDeconv has no eligible registered frames")
        result = run_mfdeconv(frames, self.backend, settings, gains, rns, fwhms, self.status, lambda f, m: self.progress(0.88 + 0.12 * f, m), variance_frames, variance_scales, rejection_masks or None)
        result.image = blend_mfdeconv(master_image, result.image, self.settings.mfdeconv_blend)
        self.status(f"MFDeconv output blend: {self.settings.mfdeconv_blend:.0%} deconvolved / {1.0 - self.settings.mfdeconv_blend:.0%} master")
        out = self.settings.output.with_name(self.settings.output.stem + "_mfdeconv" + self.settings.output.suffix)
        items = [h for h in header_items if h[0] != "NFRAMES"] + [
            ("NFRAMES", len(frames), "Frames deconvolved"),
            ("MFDECONV", result.iterations_run, "ImageMM MM iterations"),
            ("MFBLEND", self.settings.mfdeconv_blend, "MFDeconv blend (0=master, 1=full)"),
            ("MFKAPPA", settings.kappa, "Update clip"),
            ("MFRELAX", settings.relax, "Relaxation alpha"),
            ("MFCITE", "Sukurdeep 2025 AJ; Marek 2026 zenodo.19168050", "Method credit"),
        ]
        save_fits(out, result.image, items, infos[0].meta.header)
        self.status(f"Saved MFDeconv -> {out}")
        self._plate_solve(out)
        return out, result

    def _run_drizzle(self, store: FrameStore, infos: list[FrameInfo], header_items: list, crop_box) -> Path:
        s = self.settings
        d = s.drizzle
        channels, height, width = store.shape
        by_store = {f.store_index: f for f in infos if f.store_index is not None and f.rejected_reason is None}
        order = [by_store[i] for i in range(len(store)) if i in by_store]
        use_cfa = d.cfa and any(self._cfa_pattern_for(f.meta) for f in order)
        acc = new_accumulator(3 if use_cfa else channels, height, width, d.scale, self.backend.device)
        ref_offset = store.norms[0].offset
        mode = "Bayer-drizzle (raw CFA)" if use_cfa else "drizzle"
        self.status(f"{mode}: scale {d.scale}x, pixfrac {d.pixfrac}, {len(order)} frames")
        with ThreadPoolExecutor(self.io_workers) as loader:
            for n, (path, data, meta) in enumerate(self._prefetch([f.path for f in order], loader)):
                self._check_cancel()
                info = order[n]
                pattern = self._cfa_pattern_for(meta) if use_cfa else None
                tensor = self._calibrate_data(data, meta, path, skip_debayer=pattern is not None)
                if pattern is not None and tensor.shape[0] != 1:
                    pattern = None  # frame was already colour
                mask = None
                if store.mask_paths:
                    mask = torch.from_numpy(np.load(store.mask_paths[info.store_index]).astype(bool)).to(self.backend.device)
                drizzle_frame(acc, tensor, info.alignment, store.norms[info.store_index], ref_offset, d, mask, pattern, (height, width))  # type: ignore[arg-type]
                del tensor, mask, data
                self.progress(0.84 + 0.06 * (n + 1) / len(order), f"Drizzling {n + 1}/{len(order)}")
        image = acc.image(d.min_weight)
        weight = acc.weight_map()
        del acc
        self.backend.empty_cache()
        items = [h for h in header_items if h[0] not in ("CROPX0", "CROPY0", "BGNOISE")]
        if crop_box is not None:
            y0, y1, x0, x1 = crop_box
            box = (y0 * d.scale, y1 * d.scale, x0 * d.scale, x1 * d.scale)
            image = apply_box(image, box)
            weight = apply_box(weight, box)
            items += [("CROPX0", x0 * d.scale, "Autocrop left edge (drizzle px)"), ("CROPY0", y0 * d.scale, "Autocrop top edge (drizzle px)")]
        items += [
            ("DRIZSCL", d.scale, "Drizzle output scale"),
            ("DRIZKERN", d.kernel, "Drizzle drop kernel"),
            ("DRIZPIX", d.pixfrac, "Drizzle pixfrac"),
            ("DRIZMINW", d.min_weight, "Minimum relative drizzle weight"),
            ("DRIZCFA", bool(use_cfa), "Bayer drizzle from raw CFA"),
            ("BGNOISE", round(stack_noise(image), 4), "Background noise of drizzle [ADU]"),
        ]
        out = s.output.with_name(s.output.stem + "_drizzle" + s.output.suffix)
        save_fits(out, image, items, infos[0].meta.header)
        wpath = s.output.with_name(s.output.stem + "_drizzle_weight" + s.output.suffix)
        save_fits(wpath, weight, [("MAPTYPE", "drizzle_weight", "Accumulated drizzle weight"), ("DRIZSCL", d.scale, "")])
        self.status(f"Saved drizzle -> {out} ({image.shape[2]}x{image.shape[1]} px); weight map -> {wpath.name}")
        self._plate_solve(out)
        return out

    def _plate_solve(self, path: Path) -> None:
        s = self.settings
        if not s.plate_solve:
            return
        exe = find_astap(s.astap_path)
        if exe is None:
            self.status("Plate solve skipped: ASTAP not found (install from hnsky.org with a D50/D80 database)")
            return
        self.status(f"Plate solving {path.name} with ASTAP...")
        try:
            res = solve_fits(path, exe)
        except Exception as exc:  # a failed solve must never fail the run
            self.status(f"Plate solve failed for {path.name}: {exc}")
            return
        if res is None:
            self.status(f"Plate solve: no solution for {path.name} (header has no WCS)")
            return
        self.status(
            f"Plate solved {path.name}: RA {res.ra:.4f} Dec {res.dec:+.4f}, {res.scale:.3f}\"/px, "
            f"rotation {res.rotation:.1f} deg, {res.seconds:.1f}s"
        )

    def _frame_rows(self, infos: list[FrameInfo]) -> list[dict]:
        rows = []
        for f in infos:
            a = f.alignment
            rows.append(
                {
                    "index": f.index,
                    "file": f.path.name,
                    "stars": f.stars.total_detected,
                    "fwhm_px": None if not np.isfinite(f.fwhm) else round(f.fwhm, 3),
                    "fwhm_grid": None if f.fwhm_grid is None else " / ".join(" ".join("nan" if not np.isfinite(v) else f"{v:.2f}" for v in row) for row in f.fwhm_grid),
                    "fwhm_grid_counts": None if f.fwhm_count_grid is None else " / ".join(" ".join(str(int(v)) for v in row) for row in f.fwhm_count_grid),
                    "noise": round(f.noise, 4),
                    "background": round(f.background, 2),
                    "transparency": None if not np.isfinite(f.transparency) else round(f.transparency, 4),
                    "phot_stars": f.phot_stars,
                    "flux_scaled": f.flux_scaled,
                    "weight": round(f.weight, 4) if f.rejected_reason is None else None,
                    "shift_x": None if a is None else round(a.shift[0], 3),
                    "shift_y": None if a is None else round(a.shift[1], 3),
                    "rotation_deg": None if a is None else round(a.rotation_deg, 4),
                    "scale": None if a is None else round(a.scale, 6),
                    "matched": None if a is None else a.matched,
                    "refined_matches": None if a is None else a.refined_matches,
                    "residual_rms_px": None if a is None or not np.isfinite(a.residual_rms) else round(a.residual_rms, 3),
                    "flipped": None if a is None else a.flipped,
                    "excluded": f.rejected_reason,
                }
            )
        return rows

    def _write_csv(self, infos: list[FrameInfo]) -> Path:
        return write_frames_csv(self.settings.output.with_suffix(".frames.csv"), self._frame_rows(infos))

    def _write_report(self, infos: list[FrameInfo], ref_index: int, stack: StackResult, mf: MFDeconvResult | None, flips: dict, noise: float, n_eff: float, drizzle_out: Path | None = None, tilt: dict | None = None, weighting_outputs: dict[str, Path] | None = None) -> Path:
        s = self.settings
        excluded = [f for f in infos if f.rejected_reason]
        report = {
            "gpustacker": __version__,
            "backend": self.backend.describe(),
            "workers": {"detection_processes": self.cpu_workers, "loader_threads": self.io_workers},
            "output": str(s.output),
            "reference": str(infos[ref_index].path),
            "settings": {
                "weighting": s.weighting,
                "weightings": list(s.weightings),
                "interpolation": s.interpolation,
                "refine_registration": s.refine_registration,
                "local_norm": asdict(s.local_norm),
                "filters": asdict(s.filters),
                "autocrop": s.autocrop,
                "stack": asdict(s.stack),
                "drizzle": asdict(s.drizzle),
            },
            "drizzle_output": None if drizzle_out is None else str(drizzle_out),
            "weighting_outputs": {mode: str(path) for mode, path in (weighting_outputs or {}).items()},
            "stack": {
                "method": s.stack.method,
                "frames": len([f for f in infos if f.rejected_reason is None]),
                "effective_frames": n_eff,
                "rejected_low": stack.rejected_low,
                "rejected_high": stack.rejected_high,
                "rejection_fraction": stack.rejection_fraction,
                "bands": stack.bands,
                "background_noise": noise,
                "crop_box": None if stack.crop_box is None else {"y0": stack.crop_box[0], "y1": stack.crop_box[1], "x0": stack.crop_box[2], "x1": stack.crop_box[3]},
                "coverage": None if stack.coverage is None else {"min": int(stack.coverage.min()), "max": int(stack.coverage.max()), "full_depth_fraction": float((stack.coverage == stack.coverage.max()).mean())},
            },
            "excluded": {"count": len(excluded), "frames": [{"file": f.path.name, "reason": f.rejected_reason} for f in excluded]},
            "meridian_flip": flips,
            "fwhm_map": tilt or None,
            "cosmetic": None
            if not self.cosmetic_enabled
            else {"hot_sigma": s.cosmetic_settings.hot_sigma, "cold_sigma": s.cosmetic_settings.cold_sigma, "frames_processed": self._cosmetic_frames, "hot_fixed": self.cosmetic_stats.hot, "cold_fixed": self.cosmetic_stats.cold},
            "mfdeconv": None
            if mf is None
            else {
                "iterations": mf.iterations_run,
                "early_stopped": mf.early_stopped,
                "frames": [{"psf": a.psf_source, "fwhm": a.fwhm, "stars": a.stars} for a in mf.assets],
                "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(s.mfdeconv).items()},
            },
            "frames": self._frame_rows(infos),
        }
        path = s.output.with_suffix(".report.json")
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return path
