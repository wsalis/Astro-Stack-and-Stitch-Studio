"""Dark tkinter front-end for the GPUStacker pipeline."""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter

import numpy as np
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import APP_NAME, __version__
from .astro_dark_theme import DARK_BG, DARK_BORDER, DARK_FIELD, DARK_TEXT, configure_dark_theme
from .batch import DEFAULT_GROUP_KEYS, FrameGroup, describe_groups, group_frames, run_batch
from .cosmetic import CosmeticSettings
from .drizzle import DrizzleSettings
from .distributed_gui import _protect_token, _unprotect_token
from .help_text import HELP_SECTIONS
from .io import discover_frames, is_gpustacker_output
from .mfdeconv import MFDECONV_PRESETS, MFDeconvSettings
from .normalization import LocalNormSettings
from .pipeline import FrameInfo, PipelineResult, PipelineSettings, StackingPipeline, preview_frame_filters, weighting_output_path
from .quality import FILTER_PRESETS, WEIGHTING_CHOICES, FilterSettings, filter_preset
from .runlog import RunLog
from .storage import DiskSpaceEstimate, estimate_stack_disk_space, format_bytes
from .stacking import StackSettings, cleanup_registered_store, registered_store_cleanup_plan

IMAGE_TYPES = [("Astro images", "*.fit *.fits *.fts *.xisf"), ("All files", "*.*")]
SETTINGS_PATH = Path.home() / ".gpustacker" / "gui_settings.json"
_AUTO_REFERENCE = "Automatic (sharpest eligible frame)"
_TRANSIENT_VARS = {"count_var", "status_var", "reference_display_var", "reference_stats_var", "reference_choice_status_var", "distributed_token_var", "filter_summary_var", "filter_pending_var", "filter_reason_var"}


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{seconds:.1f}s"

def planned_output_files(output: Path, settings: PipelineSettings) -> list[Path]:
    """Return the files a single stack is expected to create for these settings."""

    modes = list(dict.fromkeys(settings.weightings or [settings.weighting]))
    if settings.weighting in modes:
        modes.remove(settings.weighting)
    modes.insert(0, settings.weighting)
    primary = weighting_output_path(output, settings.weighting) if len(modes) > 1 else output
    paths = [primary]
    if settings.save_maps:
        paths.extend((
            primary.with_name(primary.stem + "_coverage" + primary.suffix),
            primary.with_name(primary.stem + "_rejection" + primary.suffix),
        ))
    for mode in modes[1:]:
        paths.append(weighting_output_path(output, mode))
    if settings.drizzle.enabled:
        paths.extend((
            primary.with_name(primary.stem + "_drizzle" + primary.suffix),
            primary.with_name(primary.stem + "_drizzle_weight" + primary.suffix),
        ))
    if settings.mfdeconv is not None:
        paths.append(primary.with_name(primary.stem + "_mfdeconv" + primary.suffix))
    paths.extend((primary.with_suffix(".frames.csv"), primary.with_suffix(".report.json")))
    return paths

def integration_summary(result: PipelineResult, possible: int) -> str:
    stacked = [f for f in result.frames if f.rejected_reason is None and f.store_index is not None]
    exposure = sum(f.meta.exposure or 0.0 for f in stacked)
    return f"{len(stacked)}/{possible} frames integrated, {format_duration(exposure) if exposure else 'unknown exposure'} integration"


def rank_reference_frames(lights: list[Path], metrics_path: Path) -> dict[str, dict[str, float | int | None]] | None:
    """Rank lights from a matching prior frame report without rereading image data."""

    if not metrics_path.is_file() or len({path.name for path in lights}) != len(lights):
        return None
    try:
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return None
    return _rank_reference_rows(lights, rows)


def rank_reference_infos(infos: list[FrameInfo]) -> dict[str, dict[str, float | int | None]] | None:
    rows = []
    for info in infos:
        exposure = info.meta.exposure
        bright_flux = float(sum(sorted(info.stars.flux)[-30:]))
        focal_length = None
        for key in ("FOCALLEN", "FOCAL_LEN", "FOCAL_LENGTH"):
            try:
                focal_length = float(info.meta.header[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(focal_length) and focal_length > 0:
                break
            focal_length = None
        rows.append({
            "file": info.path.name,
            "stars": info.stars.total_detected,
            "fwhm_px": info.fwhm,
            "noise": info.noise,
            "exposure_s": exposure,
            "focal_length_mm": focal_length,
            "bright_star_rate": bright_flux / exposure if exposure and exposure > 0 else None,
            "excluded": info.rejected_reason or "",
        })
    return _rank_reference_rows([info.path for info in infos], rows)


def _rank_reference_rows(lights: list[Path], rows: list[dict]) -> dict[str, dict[str, float | int | None]] | None:
    if len({path.name for path in lights}) != len(lights) or Counter(row.get("file", "") for row in rows) != Counter(path.name for path in lights):
        return None

    def number(row: dict, key: str) -> float | None:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    metrics = {row["file"]: {
        "stars": number(row, "stars"),
        "fwhm": number(row, "fwhm_px"),
        "noise": number(row, "noise"),
        "exposure": number(row, "exposure_s"),
        "focal_length": number(row, "focal_length_mm"),
        "bright_star_rate": number(row, "bright_star_rate"),
        "excluded": 1 if row.get("excluded", "").strip() else 0,
    } for row in rows}
    usable = [m for m in metrics.values() if not m["excluded"] and m["stars"] is not None and m["fwhm"] is not None and m["noise"] is not None]
    if not usable:
        return None

    median_stars = statistics.median(m["stars"] for m in usable if m["stars"] is not None)
    focal_known = all(m["focal_length"] is not None and m["focal_length"] > 0 for m in usable)
    exposure_known = all(m["exposure"] is not None and m["exposure"] > 0 for m in usable)
    for metric in usable:
        noise = metric["noise"]
        exposure = metric["exposure"]
        metric["adjusted_noise"] = noise / math.sqrt(exposure) if exposure_known else noise
        metric["sharpness"] = metric["fwhm"] / metric["focal_length"] if focal_known else metric["fwhm"]
    median_noise = statistics.median(m["adjusted_noise"] for m in usable)

    def sort_key(item: tuple[str, dict[str, float | int | None]]) -> tuple:
        name, metric = item
        if metric["excluded"] or metric["sharpness"] is None or metric["adjusted_noise"] is None:
            return (2, math.inf, name)
        below_star_cut = metric["stars"] < 0.8 * median_stars
        score = metric["sharpness"] * (1.0 + 0.1 * metric["adjusted_noise"] / max(median_noise, 1e-9))
        return (1 if below_star_cut else 0, score, name)

    ranked = sorted(metrics.items(), key=sort_key)
    rank = 0
    for _, metric in ranked:
        if not metric["excluded"] and metric["sharpness"] is not None and metric["adjusted_noise"] is not None:
            rank += 1
            metric["rank"] = rank
        else:
            metric["rank"] = None
    return metrics


def reference_preview_path(display_label: str, choices: dict[str, Path], metrics: dict[str, dict[str, float | int | None]] | None) -> Path | None:
    """Resolve a preview candidate without changing the run's selected reference."""

    if display_label != _AUTO_REFERENCE:
        return choices.get(display_label)
    if metrics is None:
        return None
    ranked = [(metric["rank"], name) for name, metric in metrics.items() if metric.get("rank") is not None]
    if not ranked:
        return None
    best_name = min(ranked)[1]
    return next((path for path in choices.values() if path.name == best_name), None)


def reference_preview_rgb(image: np.ndarray, max_dimension: int = 760) -> np.ndarray:
    """Create a small, independently stretched RGB image for the Tk preview window."""

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[0] not in (1, 3):
        raise ValueError(f"Preview expects one or three channels, got {arr.shape}")
    channels, height, width = arr.shape
    stride = max(1, math.ceil(max(height, width) / max_dimension))
    sample = arr[:, ::stride, ::stride]
    rgb = np.repeat(sample, 3, axis=0) if channels == 1 else sample
    stretched = np.empty_like(rgb)
    for channel in range(3):
        values = rgb[channel]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            stretched[channel].fill(0)
            continue
        low, high = np.percentile(finite, (1.0, 99.8))
        if high <= low:
            stretched[channel].fill(0)
            continue
        linear = np.clip((values - low) / (high - low), 0.0, 1.0)
        stretched[channel] = np.arcsinh(10.0 * linear) / np.arcsinh(10.0)
    pixels = np.nan_to_num(np.transpose(stretched, (1, 2, 0)), nan=0.0, posinf=1.0, neginf=0.0)
    pixels = (np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
    return pixels


def open_in_file_manager(folder: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(folder))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])


class GPUStackerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(f"{APP_NAME} {__version__}")
        root.geometry("1100x800")
        root.minsize(900, 640)
        configure_dark_theme(root)
        self.lights: list[Path] = []
        self._reference_choices: dict[str, Path] = {}
        self._analysis_cache: dict[str, FrameInfo] | None = None
        self._analysis_cache_signature: str | None = None
        self._reference_metrics: dict[str, dict[str, float | int | None]] | None = None
        self._reference_choices_dirty = True
        self._ranking = False
        self._previewing = False
        self._preview_candidate_path: Path | None = None
        self._preview_window: tk.Toplevel | None = None
        self._preview_photo: tk.PhotoImage | None = None
        self._filter_preview_window: tk.Toplevel | None = None
        self._applying_filter_preset = False
        self._analysis_for_filters = False
        self._filter_measurements: dict[str, FrameInfo] | None = None
        self._filter_measurements_signature: str | None = None
        self.events: queue.Queue = queue.Queue()
        self.pipeline: StackingPipeline | None = None
        self.worker: threading.Thread | None = None
        self._cancel_requested = False
        self._run_active = False
        self._run_started = 0.0
        self._build()
        for name in (
            "bias_var", "dark_var", "flat_var", "cosmetic_var", "hot_sigma", "cold_sigma",
            "debayer_var", "debayer_method_var", "bayer_var", "det_sigma",
        ):
            variable = getattr(self, name, None)
            if isinstance(variable, tk.Variable):
                variable.trace_add("write", self._reference_analysis_settings_changed)
        self.output_var.trace_add("write", self._reference_output_changed)
        self._load_settings()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(100, self._poll)

    # ------------------------------------------------------------------ persistence

    def _setting_vars(self) -> dict[str, tk.Variable]:
        return {k: v for k, v in vars(self).items() if isinstance(v, tk.Variable) and k not in _TRANSIENT_VARS}

    def _save_settings(self) -> None:
        try:
            data = {k: v.get() for k, v in self._setting_vars().items()}
            token = self.distributed_token_var.get()
            if self.distributed_remember_token_var.get() and token and os.name == "nt":
                data["protected_distributed_token"] = _protect_token(token)
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except (OSError, tk.TclError, TypeError) as exc:
            self._append_log(f"Could not save settings: {exc}")

    def _load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self._append_log(f"Could not load settings: {exc}")
            return
        for name, var in self._setting_vars().items():
            if name in data:
                try:
                    var.set(data[name])
                except tk.TclError:
                    pass  # type mismatch from an older settings file: keep the default
        if os.name == "nt" and self.distributed_remember_token_var.get() and data.get("protected_distributed_token"):
            try:
                self.distributed_token_var.set(_unprotect_token(data["protected_distributed_token"]))
            except (OSError, ValueError, UnicodeDecodeError, AttributeError) as exc:
                self._append_log(f"Saved shared token could not be restored; enter it again. ({exc})")

    def _on_close(self) -> None:
        self._save_settings()
        self.root.destroy()

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=9)
        outer.columnconfigure(1, weight=11)
        outer.rowconfigure(1, weight=1)

        self._build_lights(outer)
        self._build_options(outer)
        self._build_run(outer)

    def _build_lights(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Light frames", padding=8)
        box.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 8))
        box.rowconfigure(2, weight=1)
        box.columnconfigure(0, weight=1)
        bar = ttk.Frame(box)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 2))
        ttk.Button(bar, text="Add files…", command=self._add_files).pack(side="left")
        ttk.Button(bar, text="Add folder…", command=self._add_folder).pack(side="left", padx=4)
        ttk.Button(bar, text="+ subfolders…", command=lambda: self._add_folder(recursive=True)).pack(side="left")
        self.count_var = tk.StringVar(value="0 frames")
        ttk.Label(bar, textvariable=self.count_var).pack(side="right")
        actions = ttk.Frame(box)
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Button(actions, text="Remove selected", command=self._remove_selected).pack(side="left")
        ttk.Button(actions, text="Clear", command=self._clear).pack(side="left", padx=4)
        self.listbox = tk.Listbox(box, bg=DARK_FIELD, fg=DARK_TEXT, selectmode="extended", highlightthickness=1, highlightbackground=DARK_BORDER, relief="flat", activestyle="none")
        self.listbox.grid(row=2, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.listbox.yview)
        scroll.grid(row=2, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scroll.set)
        reference = ttk.Frame(box)
        reference.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        reference.columnconfigure(1, weight=1)
        ttk.Label(reference, text="Reference frame").grid(row=0, column=0, sticky="w")
        self.reference_path_var = tk.StringVar(value="")
        self.reference_display_var = tk.StringVar(value=_AUTO_REFERENCE)
        self.reference_combo = ttk.Combobox(reference, textvariable=self.reference_display_var, state="readonly", postcommand=self._refresh_reference_choices)
        self.reference_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.reference_combo.bind("<<ComboboxSelected>>", self._select_reference)
        self.rank_reference_button = ttk.Button(reference, text="Analyze & rank", command=self._rank_references)
        self.rank_reference_button.grid(row=0, column=2, padx=(6, 0))
        actions = ttk.Frame(reference)
        actions.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.preview_reference_button = ttk.Button(actions, text="Preview", command=self._preview_reference)
        self.preview_reference_button.pack(side="left")
        ttk.Button(actions, text="Use as reference", command=self._use_reference_candidate).pack(side="left", padx=6)
        self.reference_stats_var = tk.StringVar(value="No ranking stats yet. Click Analyze & rank.")
        ttk.Label(reference, textvariable=self.reference_stats_var, foreground="#9aa3b2").grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))
        registered = ttk.Frame(box)
        registered.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        registered.columnconfigure(1, weight=1)
        ttk.Label(registered, text="Registered store (overrides listed lights)").grid(row=0, column=0, columnspan=3, sticky="w")
        self.registered_dir_var = tk.StringVar(value="")
        ttk.Entry(registered, textvariable=self.registered_dir_var).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0))
        ttk.Button(registered, text="Browse…", command=self._choose_registered_store).grid(row=1, column=2, padx=(6, 0), pady=(3, 0))
        self.cleanup_registered_button = ttk.Button(registered, text="Clean up…", command=self._cleanup_registered_store)
        self.cleanup_registered_button.grid(row=2, column=2, sticky="e", pady=(3, 0))
        self.frame_storage_dirs_var = tk.StringVar(value="")
        ttk.Button(registered, text="Add frame drive…", command=self._add_frame_storage_drive).grid(row=2, column=0, sticky="w", pady=(3, 0))
        self.frame_storage_list = tk.Listbox(registered, height=2, bg=DARK_FIELD, fg=DARK_TEXT, highlightthickness=1, highlightbackground=DARK_BORDER, relief="flat", activestyle="none")
        self.frame_storage_list.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(3, 0))
        ttk.Button(registered, text="Remove", command=self._remove_frame_storage_drive).grid(row=3, column=2, sticky="e", padx=(6, 0), pady=(3, 0))
        self.frame_storage_dirs_var.trace_add("write", lambda *_: self._refresh_frame_storage_list())
        self.reference_choice_status_var = tk.StringVar(value="Run reference: automatic")
        ttk.Label(reference, textvariable=self.reference_choice_status_var, foreground="#9aa3b2").grid(row=3, column=0, columnspan=3, sticky="w")

        grp = ttk.Frame(box)
        grp.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        grp.columnconfigure(1, weight=1)
        self.batch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(grp, text="Batch: one master per group", variable=self.batch_var).grid(row=0, column=0, sticky="w")
        self.group_by_var = tk.StringVar(value=",".join(DEFAULT_GROUP_KEYS))
        ttk.Entry(grp, textvariable=self.group_by_var, width=18).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(grp, text="Preview", command=self._preview_groups).grid(row=0, column=2)
        ttk.Label(grp, text="Keywords; also NIGHT, FOLDER, SIZE", foreground="#9aa3b2").grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))

    def _path_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, folder: bool = False) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=4)

        def browse() -> None:
            if folder:
                chosen = filedialog.askdirectory()
            else:
                chosen = filedialog.askopenfilename(filetypes=IMAGE_TYPES)
            if chosen:
                var.set(chosen)

        ttk.Button(parent, text="…", width=3, command=browse).grid(row=row, column=2)

    def _build_options(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Settings", padding=8)
        box.grid(row=0, column=1, sticky="nsew")
        box.configure(height=560)
        box.grid_propagate(False)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        canvas = tk.Canvas(box, highlightthickness=0, bg=DARK_FIELD)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(box, orient="vertical", command=canvas.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scroll.set)
        content = ttk.Frame(canvas)
        content.columnconfigure(0, weight=1)
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(content_window, width=event.width))

        pre = ttk.LabelFrame(content, text="Calibration & preprocessing", padding=5)
        pre.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        pre.columnconfigure(1, weight=1)
        stack = ttk.LabelFrame(content, text="Stacking & quality", padding=5)
        stack.grid(row=1, column=0, sticky="ew", pady=4)
        stack.columnconfigure(1, weight=1)
        products = ttk.LabelFrame(content, text="Output products", padding=5)
        products.grid(row=2, column=0, sticky="ew", pady=4)
        products.columnconfigure(1, weight=1)

        self.bias_var, self.dark_var, self.flat_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self._path_row(pre, 0, "Master bias", self.bias_var)
        self._path_row(pre, 1, "Master dark", self.dark_var)
        self._path_row(pre, 2, "Master flat", self.flat_var)

        ttk.Label(pre, text="Cosmetic").grid(row=3, column=0, sticky="w", pady=2)
        row = ttk.Frame(pre)
        row.grid(row=3, column=1, columnspan=2, sticky="ew")
        self.cosmetic_var = tk.StringVar(value="auto")
        ttk.Combobox(row, textvariable=self.cosmetic_var, values=["auto", "on", "off"], width=6, state="readonly").pack(side="left")
        self.hot_sigma = tk.DoubleVar(value=3.0)
        self.cold_sigma = tk.DoubleVar(value=0.0)
        ttk.Label(row, text=" hot σ").pack(side="left")
        ttk.Spinbox(row, textvariable=self.hot_sigma, from_=0, to=20, increment=0.5, width=5).pack(side="left")
        ttk.Label(row, text=" cold σ").pack(side="left")
        ttk.Spinbox(row, textvariable=self.cold_sigma, from_=0, to=20, increment=0.5, width=5).pack(side="left")
        ttk.Label(row, text=" (0 = off; auto = no dark)", foreground="#9aa3b2").pack(side="left")

        ttk.Label(pre, text="Debayer").grid(row=4, column=0, sticky="w", pady=2)
        row = ttk.Frame(pre)
        row.grid(row=4, column=1, columnspan=2, sticky="ew")
        self.debayer_var = tk.StringVar(value="auto")
        ttk.Combobox(row, textvariable=self.debayer_var, values=["auto", "on", "off"], width=6, state="readonly").pack(side="left")
        self.debayer_method_var = tk.StringVar(value="vng")
        ttk.Combobox(row, textvariable=self.debayer_method_var, values=["vng", "bilinear"], width=8, state="readonly").pack(side="left", padx=4)
        ttk.Label(row, text=" pattern").pack(side="left")
        self.bayer_var = tk.StringVar(value="")
        ttk.Combobox(row, textvariable=self.bayer_var, values=["", "RGGB", "BGGR", "GRBG", "GBRG"], width=7, state="readonly").pack(side="left", padx=4)

        ttk.Label(stack, text="Rejection").grid(row=0, column=0, sticky="w", pady=2)
        row = ttk.Frame(stack)
        row.grid(row=0, column=1, columnspan=2, sticky="ew")
        self.method_var = tk.StringVar(value="winsorized")
        ttk.Combobox(row, textvariable=self.method_var, values=["winsorized", "gesd", "sigma", "percentile", "median", "none"], width=11, state="readonly").pack(side="left")
        self.sigma_low = tk.DoubleVar(value=3.0)
        self.sigma_high = tk.DoubleVar(value=3.0)
        self.iters_var = tk.IntVar(value=3)
        ttk.Label(row, text=" low").pack(side="left")
        ttk.Spinbox(row, textvariable=self.sigma_low, from_=0.5, to=10, increment=0.5, width=5).pack(side="left")
        ttk.Label(row, text=" high").pack(side="left")
        ttk.Spinbox(row, textvariable=self.sigma_high, from_=0.5, to=10, increment=0.5, width=5).pack(side="left")
        ttk.Label(row, text=" iters").pack(side="left")
        ttk.Spinbox(row, textvariable=self.iters_var, from_=1, to=10, width=4).pack(side="left")

        ttk.Label(stack, text="GESD").grid(row=1, column=0, sticky="w", pady=2)
        row = ttk.Frame(stack)
        row.grid(row=1, column=1, columnspan=2, sticky="ew")
        self.gesd_outliers = tk.DoubleVar(value=0.3)
        self.gesd_sig = tk.DoubleVar(value=0.05)
        self.gesd_relax = tk.DoubleVar(value=1.5)
        ttk.Label(row, text="outliers").pack(side="left")
        ttk.Spinbox(row, textvariable=self.gesd_outliers, from_=0.05, to=0.5, increment=0.05, width=5).pack(side="left")
        ttk.Label(row, text=" significance").pack(side="left")
        ttk.Spinbox(row, textvariable=self.gesd_sig, from_=0.001, to=0.5, increment=0.01, width=6).pack(side="left")
        ttk.Label(row, text=" low relax").pack(side="left")
        ttk.Spinbox(row, textvariable=self.gesd_relax, from_=1.0, to=5.0, increment=0.1, width=5).pack(side="left")
        self.large_scale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Large-scale (trails)", variable=self.large_scale_var).pack(side="left", padx=8)

        ttk.Label(stack, text="Weighting").grid(row=2, column=0, sticky="w", pady=2)
        row = ttk.Frame(stack)
        row.grid(row=2, column=1, columnspan=2, sticky="ew")
        self.weightings_var = tk.StringVar(value="psfsw")
        self.weighting_button = ttk.Button(row, text="Weighting: psfsw", command=self._choose_weightings)
        self.weighting_button.pack(side="left")
        self.weightings_var.trace_add("write", lambda *_: self._update_weighting_label())
        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Normalise", variable=self.normalize_var).pack(side="left", padx=(8, 0))
        self.local_norm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Local norm (2-pass)", variable=self.local_norm_var).pack(side="left", padx=(8, 0))
        self.variant_norm_var = tk.StringVar(value="Reuse primary maps")
        ttk.Label(stack, text="Variant norm").grid(row=3, column=0, sticky="w", pady=2)
        row = ttk.Frame(stack)
        row.grid(row=3, column=1, columnspan=2, sticky="ew")
        self.variant_norm_choice = ttk.Combobox(row, textvariable=self.variant_norm_var, values=("Reuse primary maps", "Refit per method"), state="disabled", width=24)
        self.variant_norm_choice.pack(side="left")
        self.local_norm_var.trace_add("write", lambda *_: self._update_weighting_label())
        self.cpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Force CPU", variable=self.cpu_var).pack(side="left", padx=(8, 0))
        ttk.Label(row, text=" workers").pack(side="left")
        self.workers_var = tk.IntVar(value=0)
        ttk.Spinbox(row, textvariable=self.workers_var, from_=0, to=8, width=4).pack(side="left")

        self.filters_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(stack, text="Filters", variable=self.filters_enabled_var).grid(row=4, column=0, sticky="w", pady=2)
        row = ttk.Frame(stack)
        row.grid(row=4, column=1, columnspan=2, sticky="ew")
        row.columnconfigure(1, weight=1)
        row.columnconfigure(3, weight=1)
        self.filter_preset_var = tk.StringVar(value="Balanced")
        self.filter_preset_choice = ttk.Combobox(row, textvariable=self.filter_preset_var, values=(*FILTER_PRESETS, "Custom"), state="readonly", width=21)
        self.filter_preset_choice.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self.filter_preset_choice.bind("<<ComboboxSelected>>", lambda _event: self._apply_filter_preset())
        self.filter_preview_button = ttk.Button(row, text="Preview filters...", command=self._preview_filters)
        self.filter_preview_button.grid(row=0, column=2, columnspan=2, sticky="e", pady=(0, 4))
        self.f_transp = tk.DoubleVar(value=0.6)
        self.f_stars = tk.DoubleVar(value=0.6)
        self.f_bg = tk.DoubleVar(value=2.0)
        self.f_fwhm = tk.DoubleVar(value=1.4)
        self.f_psfsw = tk.DoubleVar(value=0.0)
        self._filter_preset_vars = {
            "transparency_min": self.f_transp, "stars_min_ratio": self.f_stars,
            "background_max_ratio": self.f_bg, "fwhm_max_ratio": self.f_fwhm,
        }
        self.filter_controls = []
        for index, (label, var, hi, step) in enumerate((("Transparency min", self.f_transp, 1, 0.05), ("Stars min", self.f_stars, 1, 0.05), ("Sky max", self.f_bg, 10, 0.1), ("FWHM max", self.f_fwhm, 5, 0.1), ("PSFSW min (advanced)", self.f_psfsw, 5, 0.05))):
            field_row, field_column = 1 + index // 2, 2 * (index % 2)
            text = ttk.Label(row, text=label)
            spin = ttk.Spinbox(row, textvariable=var, from_=0, to=hi, increment=step, width=6)
            text.grid(row=field_row, column=field_column, sticky="w", padx=(0, 4), pady=2)
            spin.grid(row=field_row, column=field_column + 1, sticky="w", padx=(0, 10), pady=2)
            self.filter_controls.extend((text, spin))
            var.trace_add("write", self._filter_fields_edited)
        note = ttk.Label(row, text="x median; 0 = off", foreground="#9aa3b2")
        note.grid(row=3, column=2, columnspan=2, sticky="w")
        self.filter_controls.append(note)
        self.camera_aware_filters_var = tk.BooleanVar(value=True)
        camera_aware = ttk.Checkbutton(row, text="Camera/optics-aware", variable=self.camera_aware_filters_var)
        camera_aware.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.filter_controls.append(camera_aware)
        self.camera_aware_filters_var.trace_add("write", self._filter_fields_edited)
        self.filter_summary_var = tk.StringVar(value="Analysis pending")
        ttk.Label(row, textvariable=self.filter_summary_var, wraplength=440).grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.filters_enabled_var.trace_add("write", lambda *_: self._update_filter_controls())
        self._update_filter_controls()

        ttk.Label(stack, text="Registration").grid(row=5, column=0, sticky="w", pady=2)
        row = ttk.Frame(stack)
        row.grid(row=5, column=1, columnspan=2, sticky="ew")
        self.interp_var = tk.StringVar(value="lanczos3")
        ttk.Combobox(row, textvariable=self.interp_var, values=["lanczos3", "bicubic", "bilinear"], width=9, state="readonly").pack(side="left")
        self.refine_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Polynomial refine", variable=self.refine_var).pack(side="left", padx=(8, 0))
        ttk.Label(row, text=" star σ").pack(side="left")
        self.det_sigma = tk.DoubleVar(value=5.0)
        ttk.Spinbox(row, textvariable=self.det_sigma, from_=2, to=30, increment=0.5, width=5).pack(side="left")

        ttk.Label(products, text="Crop & maps").grid(row=0, column=0, sticky="w", pady=2)
        row = ttk.Frame(products)
        row.grid(row=0, column=1, columnspan=2, sticky="ew")
        ttk.Label(row, text="autocrop").pack(side="left")
        self.autocrop_var = tk.StringVar(value="off")
        ttk.Combobox(row, textvariable=self.autocrop_var, values=["off", "100%", "95%", "90%", "80%"], width=6, state="readonly").pack(side="left", padx=(2, 8))
        self.full_field_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Full field", variable=self.full_field_var).pack(side="left", padx=(0, 8))
        self.maps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Coverage/rejection maps", variable=self.maps_var).pack(side="left")
        self.keep_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Keep registered frames", variable=self.keep_var).pack(side="left", padx=8)

        ttk.Label(products, text="Run options").grid(row=1, column=0, sticky="w", pady=2)
        row = ttk.Frame(products)
        row.grid(row=1, column=1, columnspan=3, sticky="ew")
        self.resume_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Resume after failure", variable=self.resume_var).pack(side="left")
        self.solve_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Plate solve (ASTAP)", variable=self.solve_var).pack(side="left")

        ttk.Label(products, text="Drizzle").grid(row=2, column=0, sticky="w", pady=2)
        drizzle_controls = ttk.Frame(products)
        drizzle_controls.grid(row=2, column=1, columnspan=2, sticky="ew")
        row = ttk.Frame(drizzle_controls)
        row.pack(anchor="w")
        self.drizzle_var = tk.StringVar(value="off")
        ttk.Combobox(row, textvariable=self.drizzle_var, values=["off", "2x", "3x", "4x", "5x"], width=5, state="readonly").pack(side="left")
        ttk.Label(row, text=" kernel").pack(side="left")
        self.drizzle_kernel_var = tk.StringVar(value="square")
        ttk.Combobox(row, textvariable=self.drizzle_kernel_var, values=["square", "point", "gaussian"], width=9, state="readonly").pack(side="left")
        ttk.Label(row, text=" pixfrac").pack(side="left")
        self.pixfrac_var = tk.DoubleVar(value=0.8)
        ttk.Spinbox(row, textvariable=self.pixfrac_var, from_=0.3, to=1.0, increment=0.05, width=5).pack(side="left")
        row = ttk.Frame(drizzle_controls)
        row.pack(anchor="w")
        ttk.Label(row, text=" min weight").pack(side="left")
        self.drizzle_min_weight_var = tk.DoubleVar(value=0.05)
        ttk.Spinbox(row, textvariable=self.drizzle_min_weight_var, from_=0.0, to=1.0, increment=0.05, width=5).pack(side="left")
        self.cfa_drizzle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Bayer drizzle (raw CFA for OSC)", variable=self.cfa_drizzle_var).pack(side="left", padx=8)

        mf = ttk.LabelFrame(content, text="ImageMM multi-frame deconvolution", padding=6)
        mf.grid(row=3, column=0, sticky="ew", pady=(4, 2))
        self.mf_enable = tk.BooleanVar(value=True)
        ttk.Checkbutton(mf, text="Produce deconvolved image", variable=self.mf_enable).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(mf, text="Strength").grid(row=0, column=3, sticky="e", padx=(8, 2))
        self.mf_preset = tk.StringVar(value="normal")
        ttk.Combobox(mf, textvariable=self.mf_preset, values=list(MFDECONV_PRESETS) + ["custom"], width=8, state="readonly").grid(row=0, column=4, columnspan=2, sticky="w")
        self.mf_frames = tk.IntVar(value=12)
        self.mf_iters = tk.IntVar(value=30)
        self.mf_kappa = tk.DoubleVar(value=2.0)
        self.mf_relax = tk.DoubleVar(value=0.6)
        self.mf_huber = tk.DoubleVar(value=-1.5)
        self.mf_dering = tk.DoubleVar(value=2.0)
        self.mf_tol = tk.DoubleVar(value=0.02)
        self.mf_color = tk.StringVar(value="lrgb")
        self.mf_masks = tk.BooleanVar(value=True)
        self.mf_var = tk.BooleanVar(value=True)
        self.mf_blend = tk.DoubleVar(value=0.3)
        self._mf_preset_vars = {"frames": self.mf_frames, "iterations": self.mf_iters, "kappa": self.mf_kappa, "relax": self.mf_relax, "huber_delta": self.mf_huber, "dering_sigma": self.mf_dering, "early_stop_tol": self.mf_tol}
        fields = [("Frames", self.mf_frames, 2, 200, 1), ("Iters", self.mf_iters, 1, 500, 1), ("Kappa", self.mf_kappa, 1.05, 10, 0.1), ("Relax", self.mf_relax, 0.05, 1.0, 0.05), ("Huber", self.mf_huber, -10, 1e6, 0.1), ("Dering σ", self.mf_dering, 0.0, 5.0, 0.25), ("Stop tol", self.mf_tol, 0.0, 1.0, 0.01)]
        for i, (label, var, lo, hi, step) in enumerate(fields):
            r, c = 1 + i // 3, (i % 3) * 2
            ttk.Label(mf, text=label).grid(row=r, column=c, sticky="w", padx=(0, 2))
            ttk.Spinbox(mf, textvariable=var, from_=lo, to=hi, increment=step, width=7).grid(row=r, column=c + 1, sticky="w", padx=(0, 8))
        ttk.Label(mf, text="Color").grid(row=3, column=2, sticky="w")
        ttk.Combobox(mf, textvariable=self.mf_color, values=["lrgb", "perchannel", "luma"], width=10, state="readonly").grid(row=3, column=3, sticky="w")
        ttk.Checkbutton(mf, text="Star keep-mask", variable=self.mf_masks).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(mf, text="Variance map", variable=self.mf_var).grid(row=4, column=2, columnspan=2, sticky="w")
        ttk.Label(mf, text="Blend").grid(row=4, column=4, sticky="e", padx=(4, 2))
        ttk.Spinbox(mf, textvariable=self.mf_blend, from_=0.0, to=1.0, increment=0.05, width=5).grid(row=4, column=5, sticky="w")
        ttk.Label(mf, text="Method: Sukurdeep (2025, AJ 170); notes: Marek (2026). See CREDITS.md.", foreground="#9aa3b2").grid(row=5, column=0, columnspan=6, sticky="w", pady=(4, 0))
        self._applying_preset = False
        self.mf_preset.trace_add("write", lambda *_: self._apply_mf_preset())
        for var in self._mf_preset_vars.values():
            var.trace_add("write", lambda *_: self._mf_field_edited())
        self._apply_mf_preset()

        distributed = ttk.LabelFrame(content, text="Experimental multi-system stacking", padding=5)
        distributed.grid(row=4, column=0, sticky="ew", pady=4)
        self.distributed_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(distributed, text="Use remote worker for stack bands", variable=self.distributed_enabled_var).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(distributed, text="Worker host").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.distributed_host_var = tk.StringVar()
        ttk.Entry(distributed, textvariable=self.distributed_host_var, width=28).grid(row=1, column=1, sticky="ew", padx=4, pady=(4, 0))
        ttk.Label(distributed, text="Port").grid(row=1, column=2, sticky="e", pady=(4, 0))
        self.distributed_port_var = tk.StringVar(value="47861")
        ttk.Entry(distributed, textvariable=self.distributed_port_var, width=8).grid(row=1, column=3, sticky="w", padx=(4, 0), pady=(4, 0))
        ttk.Label(distributed, text="Shared token").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.distributed_token_var = tk.StringVar()
        ttk.Entry(distributed, textvariable=self.distributed_token_var, show="*").grid(row=2, column=1, columnspan=3, sticky="ew", padx=4, pady=(4, 0))
        self.distributed_remember_token_var = tk.BooleanVar(value=os.name == "nt")
        ttk.Checkbutton(
            distributed,
            text="Remember token securely",
            variable=self.distributed_remember_token_var,
            state="normal" if os.name == "nt" else "disabled",
        ).grid(row=3, column=1, sticky="w", padx=4, pady=(3, 0))
        ttk.Label(distributed, text="Remote rows per task").grid(row=4, column=0, sticky="w", pady=(4, 0))
        self.distributed_rows_var = tk.IntVar(value=64)
        ttk.Spinbox(distributed, textvariable=self.distributed_rows_var, from_=1, to=4096, width=8).grid(row=4, column=1, sticky="w", padx=4, pady=(4, 0))
        ttk.Label(distributed, text="The worker must be enabled in Multi-system setup. Both computers need the same registered-store folder.", foreground="#9aa3b2", wraplength=510).grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 0))
        distributed.columnconfigure(1, weight=1)

        self.output_var = tk.StringVar(value="")
        ttk.Label(products, text="Output FITS").grid(row=3, column=0, sticky="w", pady=(4, 2))
        ttk.Entry(products, textvariable=self.output_var).grid(row=3, column=1, sticky="ew", padx=4, pady=(4, 2))
        ttk.Button(products, text="…", width=3, command=self._pick_output).grid(row=3, column=2, pady=(4, 2))
        ttk.Button(products, text="Folder…", command=self._pick_output_folder).grid(row=3, column=3, padx=(4, 0), pady=(4, 2))
        ttk.Label(products, text="Batch mode: this is the output folder", foreground="#9aa3b2").grid(row=4, column=1, sticky="w")

    def _apply_mf_preset(self) -> None:
        preset = MFDECONV_PRESETS.get(self.mf_preset.get())
        if preset is None:
            return
        self._applying_preset = True
        try:
            for key, var in self._mf_preset_vars.items():
                var.set(preset[key])
        finally:
            self._applying_preset = False

    def _mf_field_edited(self) -> None:
        if self._applying_preset or self.mf_preset.get() == "custom":
            return
        preset = MFDECONV_PRESETS.get(self.mf_preset.get(), {})
        try:
            matches = all(abs(float(var.get()) - float(preset[key])) < 1e-9 for key, var in self._mf_preset_vars.items())
        except (tk.TclError, ValueError):
            matches = False  # field mid-edit (empty / partial number)
        if not matches:
            self._applying_preset = True
            self.mf_preset.set("custom")
            self._applying_preset = False

    def _build_run(self, parent: ttk.Frame) -> None:
        box = ttk.Frame(parent)
        box.grid(row=1, column=1, sticky="nsew", pady=(8, 0))
        box.rowconfigure(2, weight=1)
        box.columnconfigure(0, weight=1)
        bar = ttk.Frame(box)
        bar.grid(row=0, column=0, sticky="ew")
        self.run_btn = ttk.Button(bar, text="Stack", style="Primary.TButton", command=self._start)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(bar, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)
        ttk.Button(bar, text="Compare…", command=self._open_compare).pack(side="left")
        ttk.Button(bar, text="Tilt…", command=self._open_tilt).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Mosaic…", command=self._open_mosaic).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Multi-system…", command=self._open_distributed).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Help", command=self._open_help).pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(bar, textvariable=self.status_var).pack(side="right")
        self.progress = ttk.Progressbar(box, maximum=1000)
        self.progress.grid(row=1, column=0, sticky="ew", pady=6)
        self.log = tk.Text(box, bg=DARK_FIELD, fg=DARK_TEXT, insertbackground=DARK_TEXT, relief="flat", highlightthickness=1, highlightbackground=DARK_BORDER, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.grid(row=2, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.log.yview)
        scroll.grid(row=2, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

    # ------------------------------------------------------------------ light list

    def _refresh_reference_choices(self) -> None:
        if not self._reference_choices_dirty:
            return
        selected_path = self._preview_candidate_path
        metrics = None
        if self._analysis_cache is not None:
            metrics = self._reference_metrics
            if metrics is None:
                try:
                    filters = self._filter_settings()
                except (ValueError, tk.TclError):
                    return
                evaluated, _ = preview_frame_filters(list(self._analysis_cache.values()), filters)
                metrics = rank_reference_infos(evaluated)
        output_text = self.output_var.get() if hasattr(self, "output_var") else ""
        if metrics is None and output_text:
            metrics = rank_reference_frames(self.lights, Path(output_text).with_suffix(".frames.csv"))
        self._reference_metrics = metrics
        ordered_lights = self.lights
        if metrics is not None:
            ordered_lights = sorted(self.lights, key=lambda path: metrics[path.name]["rank"] or math.inf)
        name_counts: dict[str, int] = {}
        for path in self.lights:
            name_counts[path.name] = name_counts.get(path.name, 0) + 1

        choices: dict[str, Path] = {}
        for path in ordered_lights:
            label = path.name
            if name_counts[label] > 1:
                label = f"{path.name} ({path.parent.name})"
            if label in choices:
                label = str(path)
            metric = metrics.get(path.name) if metrics is not None else None
            if metric is not None:
                if metric["rank"] is not None:
                    prefix = f"#{int(metric['rank']):02d} | "
                elif metric["excluded"]:
                    prefix = "Excluded | "
                else:
                    prefix = ""
                label = prefix + label
            choices[label] = path

        if self._preview_candidate_path is not None and not any(path.resolve() == self._preview_candidate_path.resolve() for path in choices.values()):
            self._preview_candidate_path = None
        self._reference_choices = choices
        self.reference_combo.configure(values=(_AUTO_REFERENCE, *choices))
        selected_label = next(
            (
                label
                for label, path in choices.items()
                if selected_path and path.resolve() == selected_path.resolve()
            ),
            _AUTO_REFERENCE,
        )
        self.reference_display_var.set(selected_label)
        self._update_reference_stats()
        self._reference_choices_dirty = False

    def _select_reference(self, _event=None) -> None:
        self._preview_candidate_path = self._reference_choices.get(self.reference_display_var.get())
        self._update_reference_stats()

    def _update_reference_stats(self) -> None:
        selected_reference = self.reference_path_var.get()
        self.reference_choice_status_var.set(f"Run reference: {Path(selected_reference).name if selected_reference else 'automatic'}")
        if not self._reference_metrics:
            self.reference_stats_var.set("No ranking stats yet. Click Analyze & rank.")
            return
        path = self._preview_candidate_path
        if path is None:
            path = reference_preview_path(_AUTO_REFERENCE, self._reference_choices, self._reference_metrics)
        if not path:
            self.reference_stats_var.set("No eligible ranked frame to preview.")
            return
        metric = self._reference_metrics.get(path.name)
        if metric is None:
            self.reference_stats_var.set("No statistics available for this frame.")
            return
        parts = []
        if metric["rank"] is not None:
            parts.append(f"Rank #{int(metric['rank'])}")
        if metric["fwhm"] is not None:
            parts.append(f"FWHM {metric['fwhm']:.2f}px")
        if metric["stars"] is not None:
            parts.append(f"{int(metric['stars'])} stars")
        if metric["noise"] is not None:
            parts.append(f"noise {metric['noise']:.3g}")
        if metric["exposure"] is not None:
            parts.append(f"{metric['exposure']:g}s")
        if metric["focal_length"] is not None:
            parts.append(f"{metric['focal_length']:g}mm")
        if metric["bright_star_rate"] is not None:
            parts.append(f"signal/s {metric['bright_star_rate']:.3g}")
        self.reference_stats_var.set("  |  ".join(parts) if parts else "No statistics available for this frame.")

    def _use_reference_candidate(self) -> None:
        path = reference_preview_path(self.reference_display_var.get(), self._reference_choices, self._reference_metrics)
        if path is None:
            messagebox.showinfo(APP_NAME, "Analyze & rank before choosing the automatic reference.")
            return
        self.reference_path_var.set(str(path))
        self._update_reference_stats()

    def _preview_reference(self) -> None:
        if self._ranking or self._previewing:
            return
        path = reference_preview_path(self.reference_display_var.get(), self._reference_choices, self._reference_metrics)
        if path is None:
            messagebox.showinfo(APP_NAME, "Analyze & rank before previewing the automatic reference.")
            return
        try:
            settings = self._settings()
            pipeline = StackingPipeline(settings, lambda _: None)
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror(APP_NAME, str(exc))
            return
        self._previewing = True
        self._cancel_requested = False
        self.run_btn.configure(state="disabled")
        self.rank_reference_button.configure(state="disabled")
        self.preview_reference_button.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.pipeline = pipeline
        self.status_var.set(f"Preparing preview: {path.name}")
        self.worker = threading.Thread(target=self._preview_worker, args=(path, pipeline), daemon=True)
        self.worker.start()

    def _preview_worker(self, path: Path, pipeline: StackingPipeline) -> None:
        try:
            tensor, meta = pipeline._load_calibrated(path)
            if self._cancel_requested:
                self.events.put(("reference_preview_cancelled", None))
                return
            pixels = reference_preview_rgb(tensor.detach().cpu().numpy())
            self.events.put(("reference_preview", (path, pixels, meta)))
        except Exception as exc:
            self.events.put(("reference_preview_error", f"{type(exc).__name__}: {exc}"))

    def _show_reference_preview(self, path: Path, pixels: np.ndarray, meta) -> None:
        if self._preview_window is not None and self._preview_window.winfo_exists():
            self._preview_window.destroy()
        window = tk.Toplevel(self.root)
        window.title(f"{APP_NAME} — Reference preview: {path.name}")
        window.transient(self.root)
        configure_dark_theme(window)
        height, width = pixels.shape[:2]
        self._preview_photo = tk.PhotoImage(master=window, width=width, height=height)
        for y0 in range(0, height, 16):
            rows = []
            for row in pixels[y0 : y0 + 16]:
                colors = " ".join(f"#{red:02x}{green:02x}{blue:02x}" for red, green, blue in row)
                rows.append("{" + colors + "}")
            self._preview_photo.put(" ".join(rows), to=(0, y0))
        ttk.Label(window, image=self._preview_photo).pack(padx=8, pady=8)
        exposure = f"{meta.exposure:g}s" if meta.exposure is not None else "exposure unknown"
        ttk.Label(window, text=f"{path.name}  |  {meta.shape[1]}x{meta.shape[0]}  |  {meta.channels} source channel(s)  |  {exposure}").pack(padx=8, pady=(0, 8))
        window.geometry(f"{width + 32}x{height + 72}")
        self._preview_window = window

    def _refresh_list(self) -> None:
        self.listbox.delete(0, "end")
        for p in self.lights:
            self.listbox.insert("end", p.name)
        self.count_var.set(f"{len(self.lights)} frames")
        if self.lights and not self.output_var.get():
            self.output_var.set(str(self.lights[0].parent / "gpustack.fit"))
        self._refresh_reference_choices()

    def _add_files(self) -> None:
        chosen = filedialog.askopenfilenames(filetypes=IMAGE_TYPES)
        self._extend([Path(c) for c in chosen])

    def _add_folder(self, recursive: bool = False) -> None:
        chosen = filedialog.askdirectory()
        if chosen:
            self._extend(discover_frames(chosen, recursive=recursive))

    def _group_keys(self) -> tuple[str, ...]:
        return tuple(k.strip().upper() for k in self.group_by_var.get().split(",") if k.strip())

    def _preview_groups(self) -> None:
        if not self.lights:
            messagebox.showinfo(APP_NAME, "Add some light frames first")
            return
        groups = group_frames(list(self.lights), self._group_keys(), self._append_log)
        self._append_log(f"Groups by {', '.join(self._group_keys())}:")
        for line in describe_groups(groups):
            self._append_log("  " + line)

    def _extend(self, paths: list[Path]) -> None:
        known = {p.resolve() for p in self.lights}
        fresh = [p for p in paths if p.resolve() not in known]
        outputs = [p for p in fresh if is_gpustacker_output(p)]
        if outputs:
            self._append_log(f"Skipped {len(outputs)} GPUStacker output file(s)")
        self.lights.extend(p for p in fresh if p not in outputs)
        self._invalidate_analysis_cache()
        self._refresh_list()

    def _remove_selected(self) -> None:
        for idx in sorted(self.listbox.curselection(), reverse=True):
            del self.lights[idx]
        self._invalidate_analysis_cache()
        self._refresh_list()

    def _clear(self) -> None:
        self.lights.clear()
        self._invalidate_analysis_cache()
        self._refresh_list()

    def _invalidate_analysis_cache(self) -> None:
        self._analysis_cache = None
        self._analysis_cache_signature = None
        self._filter_measurements = None
        self._filter_measurements_signature = None
        self._reference_metrics = None
        self._reference_choices_dirty = True
        if hasattr(self, "filter_summary_var"):
            self._refresh_filter_preview()

    def _reference_analysis_settings_changed(self, *_args) -> None:
        self._invalidate_analysis_cache()

    def _reference_output_changed(self, *_args) -> None:
        self._reference_choices_dirty = True

    def _analysis_signature(self, settings: PipelineSettings) -> str:
        def fingerprint(path: Path | None) -> tuple | None:
            if path is None:
                return None
            try:
                stat = path.stat()
                return str(path.resolve()), stat.st_size, stat.st_mtime_ns
            except OSError:
                return str(path)

        values = {
            "lights": [fingerprint(path) for path in settings.lights],
            "masters": [fingerprint(settings.bias), fingerprint(settings.dark), fingerprint(settings.flat)],
            "cosmetic": settings.cosmetic,
            "cosmetic_settings": vars(settings.cosmetic_settings),
            "debayer": settings.debayer,
            "debayer_method": settings.debayer_method,
            "bayer_pattern": settings.bayer_pattern,
            "detection_sigma": settings.detection_sigma,
            "min_stars": settings.min_stars,
        }
        return json.dumps(values, sort_keys=True)

    def _rank_references(self, for_filters: bool = False) -> None:
        try:
            settings = self._settings()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror(APP_NAME, str(exc))
            return
        self.rank_reference_button.configure(state="disabled")
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self._ranking = True
        self._analysis_for_filters = for_filters
        self.filter_preview_button.configure(state="disabled")
        self._cancel_requested = False
        self.progress["value"] = 0
        self.status_var.set("Analyzing frames for filter preview..." if for_filters else "Analyzing frames for reference ranking...")
        signature = self._analysis_signature(settings)
        self.worker = threading.Thread(target=self._rank_worker, args=(settings, signature), daemon=True)
        self.worker.start()

    def _rank_worker(self, settings: PipelineSettings, signature: str) -> None:
        status = lambda message: self.events.put(("log", message))
        progress = lambda fraction, message: self.events.put(("progress", (fraction, message)))
        try:
            pipeline = StackingPipeline(settings, status, progress)
            self.pipeline = pipeline
            infos = pipeline.analyse()
            self.events.put(("ranked", (infos, signature)))
        except Exception as exc:
            self.events.put(("rank_error", f"{type(exc).__name__}: {exc}"))

    def _pick_output(self) -> None:
        chosen = filedialog.asksaveasfilename(defaultextension=".fit", filetypes=[("FITS", "*.fit *.fits")])
        if chosen:
            self.output_var.set(chosen)

    def _pick_output_folder(self) -> None:
        chosen = filedialog.askdirectory()
        if not chosen:
            return
        current = Path(self.output_var.get()) if self.output_var.get() else None
        filename = current.name if current is not None and current.suffix.lower() in (".fit", ".fits", ".fts") else "gpustack.fit"
        self.output_var.set(str(Path(chosen) / filename))

    # ------------------------------------------------------------------ run

    def _settings(self) -> PipelineSettings:
        registered_dir = Path(self.registered_dir_var.get()) if self.registered_dir_var.get().strip() else None
        registered_storage_dirs = [Path(value) for value in self.frame_storage_dirs_var.get().splitlines() if value.strip()]
        distributed_enabled = bool(self.distributed_enabled_var.get())
        if not self.lights and registered_dir is None:
            raise ValueError("Add at least two light frames")
        if not self.output_var.get():
            raise ValueError("Choose an output path")
        if distributed_enabled and self.batch_var.get():
            raise ValueError("Experimental multi-system stacking is not available in batch mode")
        if distributed_enabled and len(set(self.weightings_var.get().split(","))) > 1:
            raise ValueError("Experimental multi-system stacking supports one weighting method per run")
        if registered_dir is not None and (self.resume_var.get() or self.drizzle_var.get() != "off"):
            raise ValueError("Registered-store stacking does not support resume or drizzle")
        if registered_dir is not None and registered_storage_dirs:
            raise ValueError("Drive selection applies only when registering original light frames")
        if distributed_enabled and registered_storage_dirs:
            raise ValueError("Multi-drive registered storage is not supported with experimental network workers")
        opt = lambda v: Path(v) if v.strip() else None  # noqa: E731
        mf = None
        if self.mf_enable.get():
            mf = MFDeconvSettings(
                iterations=int(self.mf_iters.get()),
                kappa=float(self.mf_kappa.get()),
                relax=float(self.mf_relax.get()),
                huber_delta=float(self.mf_huber.get()),
                dering_sigma=float(self.mf_dering.get()),
                early_stop_tol=float(self.mf_tol.get()),
                color_mode=self.mf_color.get(),  # type: ignore[arg-type]
                use_star_masks=bool(self.mf_masks.get()),
                use_variance_maps=bool(self.mf_var.get()),
            )
        weightings = [mode for mode in self.weightings_var.get().split(",") if mode in WEIGHTING_CHOICES] or ["psfsw"]
        settings = PipelineSettings(
            lights=[] if registered_dir is not None else list(self.lights),
            output=Path(self.output_var.get()),
            registered_dir=registered_dir,
            reference=opt(self.reference_path_var.get()),
            bias=opt(self.bias_var.get()),
            dark=opt(self.dark_var.get()),
            flat=opt(self.flat_var.get()),
            cosmetic=self.cosmetic_var.get(),  # type: ignore[arg-type]
            cosmetic_settings=CosmeticSettings(hot_sigma=float(self.hot_sigma.get()), cold_sigma=float(self.cold_sigma.get()), fix_hot=float(self.hot_sigma.get()) > 0, fix_cold=float(self.cold_sigma.get()) > 0),
            debayer=self.debayer_var.get(),  # type: ignore[arg-type]
            debayer_method=self.debayer_method_var.get(),  # type: ignore[arg-type]
            bayer_pattern=self.bayer_var.get() or None,
            detection_sigma=float(self.det_sigma.get()),
            weighting=weightings[0],  # type: ignore[arg-type]
            weightings=weightings,  # type: ignore[arg-type]
            filters=self._filter_settings(),
            interpolation=self.interp_var.get(),  # type: ignore[arg-type]
            refine_registration=bool(self.refine_var.get()),
            local_norm=LocalNormSettings(enabled=bool(self.local_norm_var.get())),
            refit_local_norm_per_weighting=self.variant_norm_var.get() == "Refit per method",
            autocrop=0.0 if self.autocrop_var.get() == "off" else float(self.autocrop_var.get().rstrip("%")) / 100.0,
            full_field=bool(self.full_field_var.get()),
            save_maps=bool(self.maps_var.get()),
            plate_solve=bool(self.solve_var.get()),
            drizzle=DrizzleSettings(enabled=self.drizzle_var.get() != "off", scale=int(self.drizzle_var.get().rstrip("x")) if self.drizzle_var.get() != "off" else 2, kernel=self.drizzle_kernel_var.get(), pixfrac=float(self.pixfrac_var.get()), min_weight=float(self.drizzle_min_weight_var.get()), cfa=bool(self.cfa_drizzle_var.get())),
            stack=StackSettings(
                method=self.method_var.get(),  # type: ignore[arg-type]
                sigma_low=float(self.sigma_low.get()),
                sigma_high=float(self.sigma_high.get()),
                iterations=int(self.iters_var.get()),
                gesd_outliers=float(self.gesd_outliers.get()),
                gesd_significance=float(self.gesd_sig.get()),
                gesd_low_relax=float(self.gesd_relax.get()),
                normalize=bool(self.normalize_var.get()),
                large_scale=bool(self.large_scale_var.get()),
            ),
            mfdeconv=mf,
            mfdeconv_frames=int(self.mf_frames.get()),
            mfdeconv_blend=float(self.mf_blend.get()),
            keep_registered=bool(self.keep_var.get()),
            registered_storage_dirs=registered_storage_dirs,
            distributed_enabled=distributed_enabled,
            distributed_host=self.distributed_host_var.get().strip(),
            distributed_port=int(self.distributed_port_var.get()),
            distributed_token=self.distributed_token_var.get(),
            distributed_rows_per_task=int(self.distributed_rows_var.get()),
            resume=bool(self.resume_var.get()),
            device="cpu" if self.cpu_var.get() else "auto",
            workers=int(self.workers_var.get()),
        )
        if self._analysis_cache is not None and self._analysis_signature(settings) == self._analysis_cache_signature:
            settings.analysis_cache = self._analysis_cache
        elif self._analysis_cache is not None:
            self._invalidate_analysis_cache()
        return settings

    def _update_weighting_label(self) -> None:
        selected = [mode for mode in self.weightings_var.get().split(",") if mode in WEIGHTING_CHOICES]
        if not selected:
            selected = ["psfsw"]
        label = selected[0] if len(selected) == 1 else f"{selected[0]} (+{len(selected) - 1})"
        self.weighting_button.configure(text=f"Weighting: {label}")
        enabled = len(selected) > 1 and self.local_norm_var.get()
        self.variant_norm_choice.configure(state="readonly" if enabled else "disabled")

    def _update_filter_controls(self) -> None:
        state = "normal" if self.filters_enabled_var.get() else "disabled"
        for control in self.filter_controls:
            control.configure(state=state)
        self.filter_preset_choice.configure(state="readonly" if self.filters_enabled_var.get() else "disabled")
        self._filter_fields_edited()

    def _filter_settings(self) -> FilterSettings:
        return FilterSettings(
            enabled=bool(self.filters_enabled_var.get()), transparency_min=float(self.f_transp.get()),
            stars_min_ratio=float(self.f_stars.get()), background_max_ratio=float(self.f_bg.get()),
            fwhm_max_ratio=float(self.f_fwhm.get()), psfsw_min_ratio=float(self.f_psfsw.get()),
            camera_aware=bool(self.camera_aware_filters_var.get()),
        )

    def _apply_filter_preset(self) -> None:
        name = self.filter_preset_var.get()
        if name not in FILTER_PRESETS:
            return
        settings = filter_preset(name)
        self._applying_filter_preset = True
        try:
            for key, variable in self._filter_preset_vars.items():
                variable.set(getattr(settings, key))
        finally:
            self._applying_filter_preset = False
        self._filter_fields_edited()

    def _filter_fields_edited(self, *_args) -> None:
        if self._applying_filter_preset:
            return
        name = self.filter_preset_var.get().removesuffix(" (modified)")
        if name in FILTER_PRESETS:
            preset = filter_preset(name)
            try:
                matches = all(abs(float(variable.get()) - getattr(preset, key)) < 1e-9 for key, variable in self._filter_preset_vars.items())
            except (ValueError, tk.TclError):
                matches = False
            self.filter_preset_var.set(name if matches else f"{name} (modified)")
        if hasattr(self, "filter_summary_var") and hasattr(self, "_refresh_filter_preview"):
            self._reference_metrics = None
            self._reference_choices_dirty = True
            self._refresh_filter_preview()

    def _preview_filters(self) -> None:
        if self.batch_var.get():
            messagebox.showinfo(APP_NAME, "Filter preview requires a single stacking group. Batch presets are applied separately to each group.", parent=self.root)
            return
        if self.registered_dir_var.get().strip():
            messagebox.showinfo(APP_NAME, "Filter analysis requires the original light frames, not a retained registered store.", parent=self.root)
            return
        try:
            settings = self._settings()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror(APP_NAME, str(exc), parent=self.root)
            return
        if self._filter_measurements_signature != self._analysis_signature(settings):
            self._filter_measurements = None
        if self._analysis_cache is not None and self._analysis_signature(settings) == self._analysis_cache_signature:
            self._show_filter_preview()
        else:
            self._rank_references(for_filters=True)

    def _show_filter_preview(self) -> None:
        if self._filter_preview_window is not None and self._filter_preview_window.winfo_exists():
            self._filter_preview_window.lift()
            self._refresh_filter_preview()
            return
        window = tk.Toplevel(self.root)
        self._filter_preview_window = window
        window.title(f"{APP_NAME} - Frame filter preview")
        window.geometry("1140x620")
        window.minsize(850, 440)
        configure_dark_theme(window)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(3, weight=1)
        controls = ttk.Frame(window, padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        choice = ttk.Combobox(controls, textvariable=self.filter_preset_var, values=(*FILTER_PRESETS, "Custom"), state="readonly", width=21)
        choice.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        choice.bind("<<ComboboxSelected>>", lambda _event: self._apply_filter_preset())
        ttk.Checkbutton(controls, text="Filters enabled", variable=self.filters_enabled_var).grid(row=0, column=2, columnspan=2, sticky="w")
        ttk.Checkbutton(controls, text="Camera/optics-aware", variable=self.camera_aware_filters_var).grid(row=0, column=4, columnspan=4, sticky="w")
        for column, (label, variable) in enumerate((("Stars min", self.f_stars), ("Sky max", self.f_bg), ("FWHM max", self.f_fwhm), ("Transparency min", self.f_transp), ("PSFSW min", self.f_psfsw))):
            ttk.Label(controls, text=label).grid(row=1, column=2 * column, sticky="w", padx=(0, 4))
            ttk.Spinbox(controls, textvariable=variable, from_=0, to=10, increment=0.05, width=6).grid(row=1, column=2 * column + 1, sticky="w", padx=(0, 12))
        summary_label = ttk.Label(window, textvariable=self.filter_summary_var, wraplength=800)
        summary_label.grid(row=1, column=0, sticky="w", padx=10)
        self.filter_pending_var = tk.StringVar()
        pending_label = ttk.Label(window, textvariable=self.filter_pending_var, wraplength=800)
        pending_label.grid(row=2, column=0, sticky="w", padx=10, pady=(3, 8))
        body = ttk.Frame(window, padding=(10, 0, 10, 0))
        body.grid(row=3, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        columns = ("file", "decision", "exposure", "stars", "sky", "fwhm", "transparency", "psfsw", "reason")
        self._filter_tree = ttk.Treeview(body, columns=columns, show="headings", selectmode="browse")
        for column, title, width in zip(columns, ("Frame", "Decision", "Exposure (s)", "Stars", "Sky", "FWHM (px)", "Transparency", "Relative PSFSW", "Reasons"), (250, 85, 85, 65, 85, 85, 100, 110, 420)):
            self._filter_tree.heading(column, text=title, command=lambda key=column: self._sort_filter_preview(key))
            self._filter_tree.column(column, width=width, minwidth=55, stretch=column in ("file", "reason"))
        self._filter_tree.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(body, orient="vertical", command=self._filter_tree.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(body, orient="horizontal", command=self._filter_tree.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self._filter_tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self._filter_sort = "file"
        self._filter_reverse = False
        self.filter_reason_var = tk.StringVar()
        reason_label = ttk.Label(window, textvariable=self.filter_reason_var, wraplength=800)
        reason_label.grid(row=4, column=0, sticky="w", padx=10, pady=8)
        def resize_labels(event):
            if event.widget == window:
                for label in (summary_label, pending_label, reason_label):
                    label.configure(wraplength=max(200, event.width - 24))
        window.bind("<Configure>", resize_labels)
        self._filter_tree.bind("<<TreeviewSelect>>", self._filter_selection_changed)
        self._refresh_filter_preview()

    def _sort_filter_preview(self, column: str) -> None:
        self._filter_reverse = not self._filter_reverse if column == self._filter_sort else False
        self._filter_sort = column
        self._refresh_filter_preview()

    def _filter_selection_changed(self, *_args) -> None:
        selected = self._filter_tree.selection()
        if selected:
            info = self._filter_preview_infos[int(selected[0])]
            self.filter_reason_var.set(f"{info.path.name}: {info.rejected_reason or 'Keep'}")

    def _refresh_filter_preview(self) -> None:
        cache = getattr(self, "_analysis_cache", None)
        window = getattr(self, "_filter_preview_window", None)
        tree = self._filter_tree if window is not None and window.winfo_exists() else None
        if not cache:
            self.filter_summary_var.set("Analysis pending")
            if tree is not None:
                tree.delete(*tree.get_children())
            return
        try:
            settings = self._filter_settings()
        except (ValueError, tk.TclError):
            self.filter_summary_var.set("Enter valid non-negative filter thresholds")
            if tree is not None:
                tree.delete(*tree.get_children())
            return
        evaluated, baseline = preview_frame_filters(list(cache.values()), settings)
        measurements = (getattr(self, "_filter_measurements", None) or {}) if getattr(self, "_filter_measurements_signature", None) == self._analysis_cache_signature else {}
        for key, info in zip(cache, evaluated):
            measured = measurements.get(key)
            if measured is None:
                continue
            info.transparency = measured.transparency
            info.psfsw = measured.psfsw
            if measured.rejected_reason and not measured.rejected_reason.startswith("filter:"):
                info.rejected_reason = measured.rejected_reason
            if settings.enabled and (info.rejected_reason is None or info.rejected_reason.startswith("filter:")):
                reasons = [info.rejected_reason.removeprefix("filter: ")] if info.rejected_reason else []
                if settings.transparency_min > 0 and np.isfinite(info.transparency) and info.transparency < settings.transparency_min:
                    reasons.append(f"transparency {info.transparency:.3g} < {settings.transparency_min:g}")
                if settings.psfsw_min_ratio > 0 and np.isfinite(info.psfsw) and info.psfsw < settings.psfsw_min_ratio:
                    reasons.append(f"PSFSW {info.psfsw:.4g} < {settings.psfsw_min_ratio:g} of exposure-group median")
                if reasons:
                    info.rejected_reason = "filter: " + "; ".join(reasons)
        self._filter_preview_infos = evaluated
        kept = [info for info in evaluated if info.rejected_reason is None]
        total_exposure = sum(info.meta.exposure for info in evaluated if info.meta.exposure is not None and math.isfinite(info.meta.exposure) and info.meta.exposure > 0)
        kept_exposure = sum(info.meta.exposure for info in kept if info.meta.exposure is not None and math.isfinite(info.meta.exposure) and info.meta.exposure > 0)
        exposure_text = f"{format_duration(kept_exposure)} / {format_duration(total_exposure)} known exposure ({kept_exposure / total_exposure:.0%})" if total_exposure > 0 else "Exposure unknown"
        setup_text = ""
        if settings.enabled and settings.camera_aware:
            setup_text = " Camera/optics-aware: active." if baseline["mixed_setups"]["active"] else " Camera/optics-aware: legacy comparisons."
        self.filter_summary_var.set(f"Keep {len(kept)}/{len(evaluated)} frames; reject {len(evaluated) - len(kept)}. {exposure_text}{setup_text}")
        if tree is None:
            return
        pending = ["Last-run measurements"] if measurements else []
        if settings.enabled and settings.transparency_min > 0:
            unknown = sum(not np.isfinite(info.transparency) for info in evaluated)
            pending.append(f"Transparency: {unknown} pending/unknown" if measurements else "Transparency: pending registration")
        if settings.enabled and settings.psfsw_min_ratio > 0:
            unknown = sum(not np.isfinite(info.psfsw) for info in evaluated)
            pending.append(f"PSFSW: {unknown} pending/unknown" if measurements else "PSFSW: pending registration")
        self.filter_pending_var.set("; ".join(pending) or "Late quality filters off")
        selected = tree.selection()
        tree.delete(*tree.get_children())

        def sort_key(item):
            info = item[1]
            values = {
                "file": str(info.path).casefold(), "decision": bool(info.rejected_reason),
                "exposure": info.meta.exposure or 0, "stars": info.stars.total_detected,
                "sky": info.background, "fwhm": info.fwhm,
                "transparency": info.transparency, "psfsw": getattr(info, "psfsw", float("nan")),
                "reason": info.rejected_reason or "",
            }
            value = values[self._filter_sort]
            return math.inf if isinstance(value, (int, float)) and not math.isfinite(value) else value

        def number(value, missing="Unknown"):
            return f"{value:.3g}" if value is not None and math.isfinite(value) else missing

        for index, info in sorted(enumerate(evaluated), key=sort_key, reverse=self._filter_reverse):
            tree.insert("", "end", iid=str(index), values=(
                info.path.name, "Reject" if info.rejected_reason else "Keep", number(info.meta.exposure),
                info.stars.total_detected, number(info.background), number(info.fwhm),
                number(info.transparency, "Unknown" if measurements else "Pending"), number(info.psfsw, "Unknown" if measurements else "Pending"), info.rejected_reason or "",
            ))
        if selected and tree.exists(selected[0]):
            tree.selection_set(selected)

    def _choose_weightings(self) -> None:
        popup = tk.Toplevel(self.root)
        popup.title(f"{APP_NAME} — Select weighting methods")
        popup.transient(self.root)
        configure_dark_theme(popup)
        ttk.Label(popup, text="Hold Ctrl and click to add or remove methods").pack(anchor="w", padx=10, pady=(10, 4))
        body = ttk.Frame(popup, padding=(10, 0, 10, 8))
        body.pack(fill="both", expand=True)
        choices = tk.Listbox(body, selectmode=tk.EXTENDED, height=min(10, len(WEIGHTING_CHOICES)), exportselection=False, bg=DARK_FIELD, fg=DARK_TEXT, relief="flat", highlightthickness=1, highlightbackground=DARK_BORDER)
        choices.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(body, orient="vertical", command=choices.yview)
        scroll.pack(side="right", fill="y")
        choices.configure(yscrollcommand=scroll.set)
        selected = set(self.weightings_var.get().split(","))
        for index, mode in enumerate(WEIGHTING_CHOICES):
            choices.insert("end", mode)
            if mode in selected:
                choices.selection_set(index)
        buttons = ttk.Frame(popup, padding=(10, 0, 10, 10))
        buttons.pack(fill="x")

        def apply() -> None:
            modes = [choices.get(index) for index in choices.curselection()] or ["psfsw"]
            self.weightings_var.set(",".join(modes))
            popup.destroy()

        ttk.Button(buttons, text="Cancel", command=popup.destroy).pack(side="right")
        ttk.Button(buttons, text="Apply", style="Primary.TButton", command=apply).pack(side="right", padx=6)

    def _confirm_stack(self, outputs: list[Path], notes: list[str] | None = None, disk_estimate: DiskSpaceEstimate | None = None) -> bool:
        dialog = tk.Toplevel(self.root)
        dialog.title(f"{APP_NAME} — Confirm stack outputs")
        dialog.transient(self.root)
        dialog.geometry("760x500")
        dialog.minsize(540, 320)
        configure_dark_theme(dialog)
        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=f"This run is expected to generate {len(outputs)} file(s):").pack(anchor="w", pady=(0, 8))
        if disk_estimate is None:
            ttk.Label(frame, text="Disk-space estimate unavailable; verify free space on the output drive.", foreground="#d6ad62", wraplength=720, justify="left").pack(anchor="w", pady=(0, 8))
        else:
            volume_lines = [
                f"{volume}: {format_bytes(required)} required; {format_bytes(free)} free"
                for volume, required, free in disk_estimate.volume_usage
            ] or [f"{disk_estimate.volume}: {format_bytes(disk_estimate.expected_bytes)} required; {format_bytes(disk_estimate.free_bytes)} free"]
            short = any(required > free for _, required, free in disk_estimate.volume_usage)
            low_headroom = any(int(required * 1.1) > free for _, required, free in disk_estimate.volume_usage)
            space_color = "#d6ad62" if low_headroom else DARK_TEXT
            space_line = (
                f"Estimated peak disk use: {format_bytes(disk_estimate.expected_bytes)} "
                f"(working files {format_bytes(disk_estimate.working_bytes)} + outputs {format_bytes(disk_estimate.output_bytes)})."
            )
            if short:
                space_line += " WARNING: at least one volume is short."
            elif low_headroom:
                space_line += " WARNING: less than 10% headroom remains on at least one volume."
            ttk.Label(frame, text=space_line, foreground=space_color, wraplength=720, justify="left").pack(anchor="w", pady=(0, 2))
            ttk.Label(frame, text="\n".join(volume_lines), foreground=space_color, wraplength=720, justify="left").pack(anchor="w", pady=(0, 4))
            ttk.Label(
                frame,
                text="Estimate uses the largest frame grid and channel layout in each group; filters may reduce scratch use. Registered files are removed after stacking unless Keep registered is enabled.",
                foreground="#9aa3b2",
                wraplength=720,
                justify="left",
            ).pack(anchor="w", pady=(0, 8))
        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True)
        files = tk.Listbox(body, exportselection=False, bg=DARK_FIELD, fg=DARK_TEXT, relief="flat", highlightthickness=1, highlightbackground=DARK_BORDER, font=("Consolas", 9))
        files.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(body, orient="vertical", command=files.yview)
        scroll.pack(side="right", fill="y")
        files.configure(yscrollcommand=scroll.set)
        for output in outputs:
            files.insert("end", str(output))
        if notes:
            ttk.Label(frame, text="\n".join(notes), foreground="#d6ad62", wraplength=720, justify="left").pack(anchor="w", pady=(8, 0))
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(12, 0))
        confirmed = False

        def close(value: bool) -> None:
            nonlocal confirmed
            confirmed = value
            dialog.destroy()

        ttk.Button(buttons, text="No", command=lambda: close(False)).pack(side="right")
        ttk.Button(buttons, text="Yes, start Stack", style="Primary.TButton", command=lambda: close(True)).pack(side="right", padx=(0, 6))
        dialog.protocol("WM_DELETE_WINDOW", lambda: close(False))
        dialog.bind("<Escape>", lambda _event: close(False))
        dialog.grab_set()
        dialog.lift()
        dialog.focus_force()
        self.root.wait_window(dialog)
        return confirmed

    def _start(self) -> None:
        try:
            settings = self._settings()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror(APP_NAME, str(exc))
            return
        batch_groups: list[FrameGroup] | None = None
        batch_output_dir: Path | None = None
        group_inputs: list[tuple[str, list[Path], list[Path]]] | None = None
        notes: list[str] = []
        if settings.registered_dir is not None and self.batch_var.get():
            messagebox.showerror(APP_NAME, "Batch grouping is not available when stacking a registered store.")
            return
        if self.batch_var.get():
            group_messages: list[str] = []
            batch_groups = group_frames(list(settings.lights), self._group_keys(), group_messages.append)
            out = Path(self.output_var.get())
            batch_output_dir = out if out.suffix.lower() not in (".fit", ".fits", ".fts") else out.parent
            notes.extend(group_messages)
            skipped = [group for group in batch_groups if len(group.lights) < 2]
            notes.extend(f"Skipped {group.key.slug}: fewer than 2 frames" for group in skipped)
            group_inputs = []
            planned = []
            for group in batch_groups:
                if len(group.lights) < 2:
                    continue
                group_outputs = planned_output_files(batch_output_dir / f"{group.key.slug}.fit", settings)
                planned.extend(group_outputs)
                group_inputs.append((group.key.label, group.lights, group_outputs))
        else:
            planned = planned_output_files(settings.output, settings)
        if not planned:
            messagebox.showerror(APP_NAME, "No stack outputs are planned. Check the selected frames and batch groups.")
            return
        storage_dir = batch_output_dir if batch_output_dir is not None else settings.output.parent
        try:
            if settings.registered_dir is not None:
                raise ValueError("Registered-store scratch usage is not calibrated")
            disk_estimate = estimate_stack_disk_space(settings, planned, storage_dir, group_inputs=group_inputs)
            if group_inputs is not None:
                notes.extend(
                    f"{label}: peak working {format_bytes(working)}, outputs {format_bytes(output)}"
                    for label, working, output in disk_estimate.group_details
                )
        except Exception:
            disk_estimate = None
        if settings.full_field:
            notes.append("Full-field scratch use may exceed this estimate because the aligned canvas size is known only during registration.")
        if not self._confirm_stack(planned, notes, disk_estimate):
            return
        self._save_settings()
        self._set_running(True)
        self._clear_log()
        self._cancel_requested = False
        self._run_started = time.perf_counter()
        self.worker = threading.Thread(target=self._work, args=(settings, batch_groups, batch_output_dir), daemon=True)
        self.worker.start()

    def _work(self, settings: PipelineSettings, batch_groups: list[FrameGroup] | None = None, batch_output_dir: Path | None = None) -> None:
        log_path = settings.output.with_suffix(".log")
        if batch_groups is not None and batch_output_dir is not None:
            log_path = batch_output_dir / "gpustacker_batch.log"
        try:
            run_log = RunLog(log_path)
        except OSError as exc:
            run_log = None
            self.events.put(("log", f"Could not create run log {log_path}: {exc}"))

        def status(message: str) -> None:
            if run_log is not None:
                run_log.write(message)
            self.events.put(("log", message))

        def progress(fraction: float, message: str) -> None:
            if run_log is not None:
                run_log.write(f"[{fraction:.0%}] {message}")
            self.events.put(("progress", (fraction, message)))

        if run_log is not None:
            run_log.write(f"GPUStacker {__version__}")
            run_log.write(f"Output: {settings.output}")
            self.events.put(("log", f"Run log: {log_path}"))
        try:
            if batch_groups is not None:
                if batch_output_dir is None:
                    raise ValueError("Batch output directory was not captured before confirmation")
                groups = batch_groups
                out_dir = batch_output_dir
                for line in describe_groups(groups):
                    status(line)

                def track(p) -> None:
                    self.pipeline = p

                batch = run_batch(settings, groups, out_dir, None, status, progress, on_pipeline=track, cancelled=lambda: self._cancel_requested)
                done = sum(1 for _, p, _ in batch.outputs if p)
                lines = [f"{done} of {len(batch.outputs)} group(s) stacked"]
                for (group, _, err), res in zip(batch.outputs, batch.results):
                    label = group.key.label or "(all)"
                    if res is None:
                        lines.append(f"  {label}: 0/{len(group.lights)} frames — FAILED: {err}")
                    else:
                        lines.append(f"  {label}: {integration_summary(res, len(group.lights))}")
                if run_log is not None:
                    for line in lines:
                        run_log.write(line)
                self.events.put(("done", (f"Batch finished: {done}/{len(batch.outputs)} group(s) -> {out_dir}", out_dir, lines)))
            else:
                self.pipeline = StackingPipeline(settings, status, progress)
                result = self.pipeline.run()
                if settings.registered_dir is None:
                    self.events.put(("filter_measurements", (result.frames, self._analysis_signature(settings))))
                lines = [integration_summary(result, len(result.frames) if settings.registered_dir is not None else len(settings.lights)), f"Master: {result.output.name}"]
                for label, path in (("Drizzle", result.drizzle_output), ("MFDeconv", result.mfdeconv_output)):
                    if path is not None:
                        lines.append(f"{label}: {path.name}")
                if run_log is not None:
                    for line in lines:
                        run_log.write(line)
                self.events.put(("done", (f"Finished in {result.seconds:.1f}s -> {result.output}", result.output.parent, lines)))
        except RuntimeError as exc:
            if str(exc) == "Cancelled":
                if run_log is not None:
                    run_log.write("Run cancelled")
                self.events.put(("cancelled", None))
            else:
                if run_log is not None:
                    run_log.write_exception("Run failed", exc)
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))
        except Exception as exc:  # surface everything in the GUI log
            if run_log is not None:
                run_log.write_exception("Run failed", exc)
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            if run_log is not None:
                run_log.close()

    def _choose_registered_store(self) -> None:
        chosen = filedialog.askdirectory(title="Choose retained registered store")
        if chosen:
            self.registered_dir_var.set(chosen)

    def _add_frame_storage_drive(self) -> None:
        chosen = filedialog.askdirectory(title="Choose a folder on a drive for registered frames")
        if not chosen:
            return
        selected = [Path(value) for value in self.frame_storage_dirs_var.get().splitlines() if value.strip()]
        path = Path(chosen).resolve()
        if path not in selected:
            selected.append(path)
            self.frame_storage_dirs_var.set("\n".join(str(value) for value in selected))

    def _remove_frame_storage_drive(self) -> None:
        selected = list(self.frame_storage_list.curselection())
        if not selected:
            return
        paths = self.frame_storage_dirs_var.get().splitlines()
        del paths[selected[0]]
        self.frame_storage_dirs_var.set("\n".join(paths))

    def _refresh_frame_storage_list(self) -> None:
        if not hasattr(self, "frame_storage_list"):
            return
        self.frame_storage_list.delete(0, tk.END)
        for path in self.frame_storage_dirs_var.get().splitlines():
            self.frame_storage_list.insert(tk.END, path)

    def _cleanup_registered_store(self) -> None:
        if self._run_active:
            messagebox.showerror(APP_NAME, "Wait for the active task to finish before cleaning up a registered store.")
            return
        value = self.registered_dir_var.get().strip()
        if not value:
            messagebox.showerror(APP_NAME, "Choose a registered store first.")
            return
        directory = Path(value)
        try:
            artifacts = registered_store_cleanup_plan(directory)
            total_bytes = sum(path.stat().st_size for path in artifacts)
        except (OSError, ValueError) as exc:
            messagebox.showerror(APP_NAME, f"Cannot validate registered store: {exc}")
            return
        if not messagebox.askyesno(
            "Clean up registered store",
            f"Permanently delete {len(artifacts)} validated store file(s) ({format_bytes(total_bytes)}) from:\n{directory}\n\nFiles not listed by the store index will be preserved.",
        ):
            return
        try:
            count, removed_bytes = cleanup_registered_store(directory)
        except (OSError, ValueError) as exc:
            messagebox.showerror(APP_NAME, f"Store cleanup was incomplete: {exc}")
            return
        self.registered_dir_var.set("")
        messagebox.showinfo(APP_NAME, f"Removed {count} registered-store file(s), {format_bytes(removed_bytes)}.")

    def _open_compare(self) -> None:
        from .compare_gui import CompareWindow

        out = Path(self.output_var.get()) if self.output_var.get() else None
        modes = [mode for mode in self.weightings_var.get().split(",") if mode in WEIGHTING_CHOICES]
        if out is not None and len(set(modes)) > 1:
            out = weighting_output_path(out, modes[0])
        CompareWindow(tk.Toplevel(self.root), gpu=out if out and out.is_file() else None)

    def _open_tilt(self) -> None:
        from .tilt_gui import TiltWindow

        TiltWindow(tk.Toplevel(self.root), list(self.lights))

    def _open_mosaic(self) -> None:
        from .mosaic_gui import MosaicWindow

        # Preload the masters of the last batch run (one per panel) when they exist on disk.
        tiles: list[Path] = []
        out = Path(self.output_var.get()) if self.output_var.get() else None
        if out is not None and self.batch_var.get():
            folder = out if out.is_dir() or out.suffix.lower() not in (".fit", ".fits", ".fts") else out.parent
            if folder.is_dir():
                tiles = [p for p in sorted(folder.glob("*.fit")) if is_gpustacker_output(p) and not any(p.stem.endswith(s) for s in ("_coverage", "_rejection", "_drizzle", "_drizzle_weight", "_mfdeconv"))]
        MosaicWindow(tk.Toplevel(self.root), tiles or None)

    def _open_distributed(self) -> None:
        from .distributed_gui import DistributedWindow

        store = Path(self.registered_dir_var.get()) if self.registered_dir_var.get() else None
        if store is None and self.distributed_enabled_var.get() and self.output_var.get():
            store = Path(self.output_var.get()).parent / "_gpustacker_work"
            try:
                store.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                messagebox.showerror(APP_NAME, f"Could not prepare the shared worker store: {exc}", parent=self.root)
                return
        DistributedWindow(tk.Toplevel(self.root), store)

    def _open_help(self) -> None:
        HelpWindow(tk.Toplevel(self.root))

    def _show_finished(self, folder: Path, lines: list[str], elapsed: float) -> None:
        win = tk.Toplevel(self.root)
        win.title(f"{APP_NAME} — Finished")
        win.configure(bg=DARK_BG)
        win.transient(self.root)
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=16)
        frame.pack(fill="both", expand=True)
        title = "Stacking cancelled" if self._cancel_requested else "Stacking complete"
        ttk.Label(frame, text=title, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(frame, text=f"Total time: {format_duration(elapsed)}").pack(anchor="w", pady=(6, 0))
        for line in lines:
            ttk.Label(frame, text=line).pack(anchor="w")
        ttk.Label(frame, text=str(folder), foreground="#9aa3b2").pack(anchor="w", pady=(6, 0))
        buttons = ttk.Frame(frame)
        buttons.pack(anchor="e", pady=(12, 0))

        def open_folder() -> None:
            try:
                open_in_file_manager(folder)
            except OSError as exc:
                messagebox.showerror(APP_NAME, f"Could not open {folder}: {exc}", parent=win)

        ttk.Button(buttons, text="Open output folder", style="Primary.TButton", command=open_folder).pack(side="left")
        ttk.Button(buttons, text="Close", command=win.destroy).pack(side="left", padx=(6, 0))
        win.bind("<Escape>", lambda _e: win.destroy())
        win.bell()
        win.lift()
        win.focus_force()

    def _cancel(self) -> None:
        self._cancel_requested = True
        if self.pipeline:
            self.pipeline.cancel()
            self.status_var.set("Cancelling…")

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "progress":
                    frac, msg = payload
                    self.progress["value"] = int(frac * 1000)
                    self.status_var.set(msg)
                elif kind == "done":
                    message, folder, lines = payload
                    elapsed = time.perf_counter() - self._run_started
                    self._append_log(message)
                    self._append_log(f"Total wall time: {format_duration(elapsed)}")
                    self.status_var.set("Done")
                    self._set_running(False)
                    self._reference_choices_dirty = True
                    self._show_finished(Path(folder), lines, elapsed)
                elif kind == "ranked":
                    infos, signature = payload
                    self._analysis_cache = {str(info.path.resolve()): info for info in infos}
                    self._analysis_cache_signature = signature
                    self._reference_metrics = None
                    self._reference_choices_dirty = True
                    self._ranking = False
                    self._set_running(False)
                    self._refresh_reference_choices()
                    self._refresh_filter_preview()
                    if self._analysis_for_filters:
                        self._show_filter_preview()
                    self.status_var.set("Filter analysis ready" if self._analysis_for_filters else "Reference ranking ready")
                    self._analysis_for_filters = False
                elif kind == "filter_measurements":
                    infos, signature = payload
                    self._filter_measurements = {str(info.path.resolve()): info for info in infos}
                    self._filter_measurements_signature = signature
                    self._refresh_filter_preview()
                elif kind == "rank_error":
                    self._ranking = False
                    self._set_running(False)
                    if self._cancel_requested:
                        self.status_var.set("Ranking cancelled")
                    else:
                        self._append_log(payload)
                        self.status_var.set("Ranking failed")
                elif kind == "reference_preview":
                    path, pixels, meta = payload
                    self._previewing = False
                    self._set_running(False)
                    self.status_var.set("Reference preview ready")
                    self._show_reference_preview(path, pixels, meta)
                elif kind == "reference_preview_error":
                    self._previewing = False
                    self._set_running(False)
                    self._append_log(payload)
                    self.status_var.set("Reference preview failed")
                elif kind == "reference_preview_cancelled":
                    self._previewing = False
                    self._set_running(False)
                    self.status_var.set("Preview cancelled")
                elif kind == "error":
                    self._append_log(payload)
                    self.status_var.set("Failed")
                    self._set_running(False)
                elif kind == "cancelled":
                    self._append_log("Stacking cancelled")
                    self.status_var.set("Cancelled")
                    self._set_running(False)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _set_running(self, running: bool) -> None:
        self._run_active = running
        active = running or self._ranking or self._previewing
        self.run_btn.configure(state="disabled" if active else "normal")
        self.rank_reference_button.configure(state="disabled" if active else "normal")
        self.preview_reference_button.configure(state="disabled" if active else "normal")
        self.filter_preview_button.configure(state="disabled" if active else "normal")
        self.cleanup_registered_button.configure(state="disabled" if active else "normal")
        self.cancel_btn.configure(state="normal" if active else "disabled")
        if running:
            self.progress["value"] = 0
            self.status_var.set("Starting…")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


class HelpWindow:
    """Section list on the left, scrollable documentation on the right."""

    def __init__(self, win: tk.Toplevel) -> None:
        win.title(f"{APP_NAME} — Help")
        win.geometry("980x700")
        win.minsize(700, 480)
        win.configure(bg=DARK_BG)
        outer = ttk.Frame(win, padding=8)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)
        self.sections = tk.Listbox(outer, bg=DARK_FIELD, fg=DARK_TEXT, width=30, exportselection=False, highlightthickness=1, highlightbackground=DARK_BORDER, relief="flat", activestyle="none")
        self.sections.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self.text = tk.Text(outer, bg=DARK_FIELD, fg=DARK_TEXT, relief="flat", highlightthickness=1, highlightbackground=DARK_BORDER, wrap="word", padx=10, pady=8, font=("Consolas", 10))
        self.text.grid(row=0, column=1, sticky="nsew")
        scroll = ttk.Scrollbar(outer, orient="vertical", command=self.text.yview)
        scroll.grid(row=0, column=2, sticky="ns")
        self.text.configure(yscrollcommand=scroll.set)
        self.text.tag_configure("h1", font=("Segoe UI", 14, "bold"), spacing1=14, spacing3=6)
        self._marks: list[str] = []
        for i, (title, body) in enumerate(HELP_SECTIONS):
            self.sections.insert("end", title)
            mark = f"sec{i}"
            self.text.mark_set(mark, "end-1c")
            self.text.mark_gravity(mark, "left")
            self._marks.append(mark)
            self.text.insert("end", title + "\n", "h1")
            self.text.insert("end", body.strip() + "\n\n")
        self.text.configure(state="disabled")
        self.sections.bind("<<ListboxSelect>>", self._jump)
        self.sections.selection_set(0)
        win.bind("<Escape>", lambda _e: win.destroy())

    def _jump(self, _event: object = None) -> None:
        sel = self.sections.curselection()
        if sel:
            self.text.yview(self._marks[sel[0]])


def main() -> int:
    root = tk.Tk()
    GPUStackerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
