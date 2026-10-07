"""Dark tkinter front-end for the GPUStacker pipeline."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import __version__
from .astro_dark_theme import DARK_BG, DARK_BORDER, DARK_FIELD, DARK_TEXT, configure_dark_theme
from .batch import DEFAULT_GROUP_KEYS, FrameGroup, describe_groups, group_frames, run_batch
from .cosmetic import CosmeticSettings
from .drizzle import DrizzleSettings
from .help_text import HELP_SECTIONS
from .io import discover_frames, is_gpustacker_output
from .mfdeconv import MFDECONV_PRESETS, MFDeconvSettings
from .normalization import LocalNormSettings
from .pipeline import PipelineResult, PipelineSettings, StackingPipeline, weighting_output_path
from .quality import WEIGHTING_CHOICES, FilterSettings
from .stacking import StackSettings

IMAGE_TYPES = [("Astro images", "*.fit *.fits *.fts *.xisf"), ("All files", "*.*")]
SETTINGS_PATH = Path.home() / ".gpustacker" / "gui_settings.json"
_TRANSIENT_VARS = {"count_var", "status_var"}


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
        root.title(f"GPUStacker {__version__}")
        root.geometry("1100x800")
        root.minsize(900, 640)
        configure_dark_theme(root)
        self.lights: list[Path] = []
        self.events: queue.Queue = queue.Queue()
        self.pipeline: StackingPipeline | None = None
        self.worker: threading.Thread | None = None
        self._cancel_requested = False
        self._run_started = 0.0
        self._build()
        self._load_settings()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(100, self._poll)

    # ------------------------------------------------------------------ persistence

    def _setting_vars(self) -> dict[str, tk.Variable]:
        return {k: v for k, v in vars(self).items() if isinstance(v, tk.Variable) and k not in _TRANSIENT_VARS}

    def _save_settings(self) -> None:
        try:
            data = {k: v.get() for k, v in self._setting_vars().items()}
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
        grp = ttk.Frame(box)
        grp.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
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
        ttk.Spinbox(row, textvariable=self.workers_var, from_=0, to=64, width=4).pack(side="left")

        self.filters_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(stack, text="Filters", variable=self.filters_enabled_var).grid(row=4, column=0, sticky="w", pady=2)
        row = ttk.Frame(stack)
        row.grid(row=4, column=1, columnspan=2, sticky="ew")
        self.f_transp = tk.DoubleVar(value=0.6)
        self.f_stars = tk.DoubleVar(value=0.6)
        self.f_bg = tk.DoubleVar(value=2.0)
        self.f_fwhm = tk.DoubleVar(value=1.4)
        self.filter_controls = []
        for label, var, lo, hi, step in (("transp <", self.f_transp, 0, 1, 0.05), (" stars <", self.f_stars, 0, 1, 0.05), (" sky >", self.f_bg, 0, 10, 0.1), (" FWHM >", self.f_fwhm, 0, 5, 0.1)):
            text = ttk.Label(row, text=label)
            spin = ttk.Spinbox(row, textvariable=var, from_=lo, to=hi, increment=step, width=5)
            text.pack(side="left")
            spin.pack(side="left")
            self.filter_controls.extend((text, spin))
        note = ttk.Label(row, text=" (x median; 0 = off)", foreground="#9aa3b2")
        note.pack(side="left")
        self.filter_controls.append(note)
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
        self.maps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Coverage/rejection maps", variable=self.maps_var).pack(side="left")
        self.keep_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Keep registered frames", variable=self.keep_var).pack(side="left", padx=8)
        self.solve_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Plate solve (ASTAP)", variable=self.solve_var).pack(side="left")

        ttk.Label(products, text="Drizzle").grid(row=1, column=0, sticky="w", pady=2)
        drizzle_controls = ttk.Frame(products)
        drizzle_controls.grid(row=1, column=1, columnspan=2, sticky="ew")
        row = ttk.Frame(drizzle_controls)
        row.pack(anchor="w")
        self.drizzle_var = tk.StringVar(value="off")
        ttk.Combobox(row, textvariable=self.drizzle_var, values=["off", "2x", "3x"], width=5, state="readonly").pack(side="left")
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

        self.output_var = tk.StringVar(value="")
        ttk.Label(products, text="Output FITS").grid(row=2, column=0, sticky="w", pady=(4, 2))
        ttk.Entry(products, textvariable=self.output_var).grid(row=2, column=1, sticky="ew", padx=4, pady=(4, 2))
        ttk.Button(products, text="…", width=3, command=self._pick_output).grid(row=2, column=2, pady=(4, 2))
        ttk.Button(products, text="Folder…", command=self._pick_output_folder).grid(row=2, column=3, padx=(4, 0), pady=(4, 2))
        ttk.Label(products, text="Batch mode: this is the output folder", foreground="#9aa3b2").grid(row=3, column=1, sticky="w")

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

    def _refresh_list(self) -> None:
        self.listbox.delete(0, "end")
        for p in self.lights:
            self.listbox.insert("end", p.name)
        self.count_var.set(f"{len(self.lights)} frames")
        if self.lights and not self.output_var.get():
            self.output_var.set(str(self.lights[0].parent / "gpustack.fit"))

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
            messagebox.showinfo("GPUStacker", "Add some light frames first")
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
        self._refresh_list()

    def _remove_selected(self) -> None:
        for idx in sorted(self.listbox.curselection(), reverse=True):
            del self.lights[idx]
        self._refresh_list()

    def _clear(self) -> None:
        self.lights.clear()
        self._refresh_list()

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
        if not self.lights:
            raise ValueError("Add at least two light frames")
        if not self.output_var.get():
            raise ValueError("Choose an output path")
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
        return PipelineSettings(
            lights=list(self.lights),
            output=Path(self.output_var.get()),
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
            filters=FilterSettings(enabled=bool(self.filters_enabled_var.get()), transparency_min=float(self.f_transp.get()), stars_min_ratio=float(self.f_stars.get()), background_max_ratio=float(self.f_bg.get()), fwhm_max_ratio=float(self.f_fwhm.get())),
            interpolation=self.interp_var.get(),  # type: ignore[arg-type]
            refine_registration=bool(self.refine_var.get()),
            local_norm=LocalNormSettings(enabled=bool(self.local_norm_var.get())),
            refit_local_norm_per_weighting=self.variant_norm_var.get() == "Refit per method",
            autocrop=0.0 if self.autocrop_var.get() == "off" else float(self.autocrop_var.get().rstrip("%")) / 100.0,
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
            device="cpu" if self.cpu_var.get() else "auto",
            workers=int(self.workers_var.get()),
        )

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

    def _choose_weightings(self) -> None:
        popup = tk.Toplevel(self.root)
        popup.title("Select weighting methods")
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

    def _confirm_stack(self, outputs: list[Path], notes: list[str] | None = None) -> bool:
        dialog = tk.Toplevel(self.root)
        dialog.title("Confirm Stack outputs")
        dialog.transient(self.root)
        dialog.geometry("760x500")
        dialog.minsize(540, 320)
        configure_dark_theme(dialog)
        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=f"This run is expected to generate {len(outputs)} file(s):").pack(anchor="w", pady=(0, 8))
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
            messagebox.showerror("GPUStacker", str(exc))
            return
        batch_groups: list[FrameGroup] | None = None
        batch_output_dir: Path | None = None
        notes: list[str] = []
        if self.batch_var.get():
            group_messages: list[str] = []
            batch_groups = group_frames(list(settings.lights), self._group_keys(), group_messages.append)
            out = Path(self.output_var.get())
            batch_output_dir = out if out.suffix.lower() not in (".fit", ".fits", ".fts") else out.parent
            notes.extend(group_messages)
            skipped = [group for group in batch_groups if len(group.lights) < 2]
            notes.extend(f"Skipped {group.key.slug}: fewer than 2 frames" for group in skipped)
            planned = [
                path
                for group in batch_groups
                if len(group.lights) >= 2
                for path in planned_output_files(batch_output_dir / f"{group.key.slug}.fit", settings)
            ]
        else:
            planned = planned_output_files(settings.output, settings)
        if not planned:
            messagebox.showerror("GPUStacker", "No stack outputs are planned. Check the selected frames and batch groups.")
            return
        if not self._confirm_stack(planned, notes):
            return
        self._save_settings()
        self._set_running(True)
        self._clear_log()
        self._cancel_requested = False
        self._run_started = time.perf_counter()
        self.worker = threading.Thread(target=self._work, args=(settings, batch_groups, batch_output_dir), daemon=True)
        self.worker.start()

    def _work(self, settings: PipelineSettings, batch_groups: list[FrameGroup] | None = None, batch_output_dir: Path | None = None) -> None:
        status = lambda m: self.events.put(("log", m))  # noqa: E731
        progress = lambda f, m: self.events.put(("progress", (f, m)))  # noqa: E731
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
                self.events.put(("done", (f"Batch finished: {done}/{len(batch.outputs)} group(s) -> {out_dir}", out_dir, lines)))
            else:
                self.pipeline = StackingPipeline(settings, status, progress)
                result = self.pipeline.run()
                lines = [integration_summary(result, len(settings.lights)), f"Master: {result.output.name}"]
                for label, path in (("Drizzle", result.drizzle_output), ("MFDeconv", result.mfdeconv_output)):
                    if path is not None:
                        lines.append(f"{label}: {path.name}")
                self.events.put(("done", (f"Finished in {result.seconds:.1f}s -> {result.output}", result.output.parent, lines)))
        except Exception as exc:  # surface everything in the GUI log
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))

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

    def _open_help(self) -> None:
        HelpWindow(tk.Toplevel(self.root))

    def _show_finished(self, folder: Path, lines: list[str], elapsed: float) -> None:
        win = tk.Toplevel(self.root)
        win.title("GPUStacker — finished")
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
                messagebox.showerror("GPUStacker", f"Could not open {folder}: {exc}", parent=win)

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
                    self._show_finished(Path(folder), lines, elapsed)
                elif kind == "error":
                    self._append_log(payload)
                    self.status_var.set("Failed")
                    self._set_running(False)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _set_running(self, running: bool) -> None:
        self.run_btn.configure(state="disabled" if running else "normal")
        self.cancel_btn.configure(state="normal" if running else "disabled")
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
        win.title("GPUStacker — Help")
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
