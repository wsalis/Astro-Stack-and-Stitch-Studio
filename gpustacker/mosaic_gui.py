"""Mosaic window: pick plate-solved master tiles, set blend options, build and watch the log."""

from __future__ import annotations

import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .astro_dark_theme import DARK_BG, DARK_BORDER, DARK_FIELD, DARK_TEXT, configure_dark_theme
from .io import discover_frames
from .mosaic import BLEND_MODES, INTERPOLATIONS, ORIENTATIONS, MosaicBuilder, MosaicCancelled, MosaicResult, MosaicSettings

TILE_TYPES = [("Master tiles", "*.fit *.fits *.fts *.xisf"), ("All files", "*.*")]
SETTINGS_PATH = Path.home() / ".gpustacker" / "mosaic_settings.json"
NEUTRAL = "#9aa3b2"
_TRANSIENT = {"status_var", "count_var"}
_MAP_SUFFIXES = ("_coverage", "_rejection", "_drizzle_weight")


def _is_map(path: Path) -> bool:
    return any(path.stem.endswith(s) for s in _MAP_SUFFIXES)


class MosaicWindow:
    def __init__(self, root: tk.Misc, tiles: list[Path] | None = None, output: Path | None = None) -> None:
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.tiles: list[Path] = []
        self.builder: MosaicBuilder | None = None
        self.worker: threading.Thread | None = None
        self._started = 0.0
        self._cancelled = False
        root.title("GPUStacker — mosaic")
        if isinstance(root, (tk.Tk, tk.Toplevel)):
            root.geometry("1120x760")
            root.minsize(900, 600)
        configure_dark_theme(root)
        self._build()
        self._load_settings()
        if tiles:
            self._add(tiles)
        if output is not None:
            self.output_var.set(str(output))
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(100, self._poll)

    # ------------------------------------------------------------------ persistence

    def _setting_vars(self) -> dict[str, tk.Variable]:
        return {k: v for k, v in vars(self).items() if isinstance(v, tk.Variable) and k not in _TRANSIENT}

    def _save_settings(self) -> None:
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps({k: v.get() for k, v in self._setting_vars().items()}, indent=2), encoding="utf-8")
        except (OSError, tk.TclError, TypeError):
            pass

    def _load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for name, var in self._setting_vars().items():
            if name in data and name != "output_var":
                try:
                    var.set(data[name])
                except tk.TclError:
                    pass

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(1, weight=1)

        bar = ttk.Frame(outer)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Button(bar, text="Add tiles…", command=self._browse_files).pack(side="left")
        ttk.Button(bar, text="Add folder…", command=self._browse_folder).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Remove", command=self._remove).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Clear", command=self._clear).pack(side="left", padx=(6, 12))
        ttk.Button(bar, text="▲", width=3, command=lambda: self._move(-1)).pack(side="left")
        ttk.Button(bar, text="▼", width=3, command=lambda: self._move(1)).pack(side="left", padx=(2, 12))
        self.count_var = tk.StringVar(value="0 tiles")
        ttk.Label(bar, textvariable=self.count_var, foreground=NEUTRAL).pack(side="left")
        ttk.Label(bar, text="The first tile is the photometric and orientation reference.", foreground=NEUTRAL).pack(side="right")

        left = ttk.Frame(outer)
        left.grid(row=1, column=0, sticky="ns", padx=(0, 10))
        left.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(left, width=46, bg=DARK_FIELD, fg=DARK_TEXT, selectbackground="#285f9e", selectforeground="#ffffff", relief="flat", highlightthickness=1, highlightbackground=DARK_BORDER, exportselection=False, font=("Consolas", 9), activestyle="none")
        self.listbox.grid(row=0, column=0, sticky="ns")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scroll.set)

        right = ttk.Frame(outer)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        self._build_options(right)
        self._build_run(right)

    def _build_options(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Mosaic", padding=8)
        box.grid(row=0, column=0, sticky="ew")
        for col in (1, 3, 5):
            box.columnconfigure(col, weight=1)

        ttk.Label(box, text="Output FITS").grid(row=0, column=0, sticky="w")
        self.output_var = tk.StringVar()
        ttk.Entry(box, textvariable=self.output_var).grid(row=0, column=1, columnspan=4, sticky="ew", padx=4)
        ttk.Button(box, text="Browse…", command=self._browse_output).grid(row=0, column=5, sticky="e")

        ttk.Label(box, text="Gradient degree").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.gradient_var = tk.IntVar(value=2)
        ttk.Combobox(box, textvariable=self.gradient_var, values=[0, 1, 2, 3], width=5, state="readonly").grid(row=1, column=1, sticky="w", padx=4, pady=(8, 0))
        ttk.Label(box, text="Sky block px").grid(row=1, column=2, sticky="e", pady=(8, 0))
        self.block_var = tk.IntVar(value=64)
        ttk.Spinbox(box, from_=16, to=512, increment=16, textvariable=self.block_var, width=6).grid(row=1, column=3, sticky="w", padx=4, pady=(8, 0))
        ttk.Label(box, text="Pixel scale \"/px (0 = finest)").grid(row=1, column=4, sticky="e", pady=(8, 0))
        self.scale_var = tk.DoubleVar(value=0.0)
        ttk.Entry(box, textvariable=self.scale_var, width=8).grid(row=1, column=5, sticky="w", padx=4, pady=(8, 0))

        ttk.Label(box, text="Orientation").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.orientation_var = tk.StringVar(value="first")
        ttk.Combobox(box, textvariable=self.orientation_var, values=list(ORIENTATIONS), width=8, state="readonly").grid(row=2, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(box, text="Interpolation").grid(row=2, column=2, sticky="e", pady=(6, 0))
        self.interp_var = tk.StringVar(value="lanczos3")
        ttk.Combobox(box, textvariable=self.interp_var, values=list(INTERPOLATIONS), width=9, state="readonly").grid(row=2, column=3, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(box, text="Blend").grid(row=2, column=4, sticky="e", pady=(6, 0))
        self.blend_var = tk.StringVar(value="feather")
        ttk.Combobox(box, textvariable=self.blend_var, values=list(BLEND_MODES), width=8, state="readonly").grid(row=2, column=5, sticky="w", padx=4, pady=(6, 0))

        ttk.Label(box, text="Feather px").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.feather_var = tk.DoubleVar(value=100.0)
        ttk.Entry(box, textvariable=self.feather_var, width=8).grid(row=3, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(box, text="Seam px").grid(row=3, column=2, sticky="e", pady=(6, 0))
        self.seam_var = tk.DoubleVar(value=4.0)
        ttk.Entry(box, textvariable=self.seam_var, width=8).grid(row=3, column=3, sticky="w", padx=4, pady=(6, 0))

        checks = ttk.Frame(box)
        checks.grid(row=4, column=0, columnspan=6, sticky="w", pady=(8, 0))
        self.refine_var = tk.BooleanVar(value=True)
        self.photometric_var = tk.BooleanVar(value=True)
        self.plane_var = tk.BooleanVar(value=True)
        self.solve_var = tk.BooleanVar(value=True)
        self.coverage_var = tk.BooleanVar(value=True)
        self.autocrop_var = tk.BooleanVar(value=True)
        self.cpu_var = tk.BooleanVar(value=False)
        for index, (text, var) in enumerate((
            ("Refine registration from overlap stars", self.refine_var),
            ("Photometric match", self.photometric_var),
            ("Residual plane per tile", self.plane_var),
            ("Plate solve tiles without WCS (ASTAP)", self.solve_var),
            ("Save coverage map", self.coverage_var),
            ("Auto-crop black borders", self.autocrop_var),
            ("CPU only", self.cpu_var),
        )):
            ttk.Checkbutton(checks, text=text, variable=var).grid(row=index // 3, column=index % 3, sticky="w", padx=(0, 12), pady=(0, 2))

    def _build_run(self, parent: ttk.Frame) -> None:
        box = ttk.Frame(parent)
        box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.run_btn = ttk.Button(box, text="Build mosaic", style="Primary.TButton", command=self._start)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(box, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="Add at least two plate-solved master tiles")
        ttk.Label(box, textvariable=self.status_var).pack(side="right")
        self.progress = ttk.Progressbar(parent, maximum=1000)
        self.progress.grid(row=3, column=0, sticky="ew", pady=6)
        self.log = tk.Text(parent, bg=DARK_FIELD, fg=DARK_TEXT, insertbackground=DARK_TEXT, relief="flat", highlightthickness=1, highlightbackground=DARK_BORDER, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        scroll = ttk.Scrollbar(parent, orient="vertical", command=self.log.yview)
        scroll.grid(row=2, column=1, sticky="ns", pady=(8, 0))
        self.log.configure(yscrollcommand=scroll.set)

    # ------------------------------------------------------------------ tile list

    def _browse_files(self) -> None:
        chosen = filedialog.askopenfilenames(filetypes=TILE_TYPES, parent=self.root)
        if chosen:
            self._add([Path(p) for p in chosen])

    def _browse_folder(self) -> None:
        chosen = filedialog.askdirectory(parent=self.root)
        if chosen:
            self._add([p for p in discover_frames(chosen, skip_outputs=False) if not _is_map(p)])

    def _browse_output(self) -> None:
        initial = Path(self.output_var.get()) if self.output_var.get() else (self.tiles[0].parent / "mosaic.fit" if self.tiles else None)
        chosen = filedialog.asksaveasfilename(defaultextension=".fit", filetypes=[("FITS", "*.fit *.fits")], initialfile=initial.name if initial else "mosaic.fit", initialdir=str(initial.parent) if initial else None, parent=self.root)
        if chosen:
            self.output_var.set(chosen)

    def _add(self, paths: list[Path]) -> None:
        known = {p.resolve() for p in self.tiles}
        for p in paths:
            if p.resolve() not in known:
                self.tiles.append(p)
                known.add(p.resolve())
        self._refresh_list()
        if self.tiles and not self.output_var.get():
            self.output_var.set(str(self.tiles[0].parent / "mosaic.fit"))

    def _remove(self) -> None:
        sel = list(self.listbox.curselection())
        for idx in reversed(sel):
            del self.tiles[idx]
        self._refresh_list()

    def _clear(self) -> None:
        self.tiles.clear()
        self._refresh_list()

    def _move(self, step: int) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        new = idx + step
        if not 0 <= new < len(self.tiles):
            return
        self.tiles[idx], self.tiles[new] = self.tiles[new], self.tiles[idx]
        self._refresh_list()
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(new)

    def _refresh_list(self) -> None:
        self.listbox.delete(0, "end")
        for i, p in enumerate(self.tiles):
            self.listbox.insert("end", f"{i + 1:2d}  {p.name}")
        self.count_var.set(f"{len(self.tiles)} tiles")

    # ------------------------------------------------------------------ run

    def _settings(self) -> MosaicSettings:
        out = self.output_var.get().strip()
        if not out:
            raise ValueError("Choose an output FITS path")
        return MosaicSettings(
            tiles=list(self.tiles),
            output=Path(out),
            gradient_degree=int(self.gradient_var.get()),
            gradient_block=int(self.block_var.get()),
            pixel_scale=float(self.scale_var.get()),
            orientation=self.orientation_var.get(),
            interpolation=self.interp_var.get(),
            refine=bool(self.refine_var.get()),
            photometric=bool(self.photometric_var.get()),
            match_gradient=bool(self.plane_var.get()),
            feather=float(self.feather_var.get()),
            seam_width=float(self.seam_var.get()),
            blend_mode=self.blend_var.get(),
            auto_crop=bool(self.autocrop_var.get()),
            plate_solve=bool(self.solve_var.get()),
            save_coverage=bool(self.coverage_var.get()),
            device="cpu" if self.cpu_var.get() else "auto",
        )

    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        if len(self.tiles) < 2:
            messagebox.showerror("Mosaic", "Add at least two tiles", parent=self.root)
            return
        try:
            settings = self._settings()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Mosaic", str(exc), parent=self.root)
            return
        if settings.output.exists() and not messagebox.askyesno("Mosaic", f"{settings.output.name} exists. Overwrite?", parent=self.root):
            return
        self._save_settings()
        self._set_running(True)
        self._clear_log()
        self._cancelled = False
        self._started = time.perf_counter()
        self.worker = threading.Thread(target=self._work, args=(settings,), daemon=True)
        self.worker.start()

    def _work(self, settings: MosaicSettings) -> None:
        status = lambda m: self.events.put(("log", m))  # noqa: E731
        progress = lambda f, m: self.events.put(("progress", (f, m)))  # noqa: E731
        try:
            self.builder = MosaicBuilder(settings, status, progress)
            result = self.builder.run()
            self.events.put(("done", result))
        except MosaicCancelled:
            self.events.put(("cancelled", None))
        except Exception as exc:  # surface everything in the log
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))

    def _cancel(self) -> None:
        self._cancelled = True
        if self.builder is not None:
            self.builder.cancel()
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
                    self._finished(payload)
                elif kind == "cancelled":
                    self._append_log("Cancelled")
                    self.status_var.set("Cancelled")
                    self._set_running(False)
                elif kind == "error":
                    self._append_log(payload)
                    self.status_var.set("Failed")
                    self._set_running(False)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _finished(self, result: MosaicResult) -> None:
        from .gui import format_duration, open_in_file_manager

        elapsed = time.perf_counter() - self._started
        self._append_log(f"Total wall time: {format_duration(elapsed)}")
        self.status_var.set("Done")
        self._set_running(False)
        win = tk.Toplevel(self.root)
        win.title("GPUStacker — mosaic finished")
        win.configure(bg=DARK_BG)
        win.transient(self.root)
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Mosaic complete", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(frame, text=f"Total time: {format_duration(elapsed)}").pack(anchor="w", pady=(6, 0))
        ttk.Label(frame, text=f"{len(result.tiles)} tiles -> {result.shape[1]} x {result.shape[0]} px at {result.pixel_scale:.3f}\"/px").pack(anchor="w")
        findings = result.analysis.findings
        if findings:
            ttk.Label(frame, text=f"Analyzer: {len(findings)} item(s) to review", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
            for finding in findings[:3]:
                ttk.Label(
                    frame,
                    text=f"{finding.message}\nTry: {finding.suggestion}",
                    wraplength=560,
                    justify="left",
                ).pack(anchor="w", pady=(4, 0))
            if len(findings) > 3:
                ttk.Label(frame, text="Additional findings are in the analysis report.", foreground=NEUTRAL).pack(anchor="w", pady=(4, 0))
        else:
            ttk.Label(frame, text="Analyzer: no localized overlap or star-alignment risks detected").pack(anchor="w", pady=(8, 0))
        if result.analysis_output is not None:
            ttk.Label(frame, text=f"Analysis report: {result.analysis_output.name}", foreground=NEUTRAL).pack(anchor="w", pady=(4, 0))
        worst = max((o.rms for o in result.overlaps if o.rms == o.rms), default=float("nan"))
        if worst == worst:
            ttk.Label(frame, text=f"Worst overlap star scatter: {worst:.2f} px").pack(anchor="w")
        after = [o.residual_after for o in result.overlaps if o.residual_after == o.residual_after]
        if after:
            ttk.Label(frame, text=f"Largest overlap background mismatch after matching: {max(after):.2f} ADU").pack(anchor="w")
        ttk.Label(frame, text=str(result.output), foreground=NEUTRAL).pack(anchor="w", pady=(6, 0))
        buttons = ttk.Frame(frame)
        buttons.pack(anchor="e", pady=(12, 0))

        def open_folder() -> None:
            try:
                open_in_file_manager(result.output.parent)
            except OSError as exc:
                messagebox.showerror("Mosaic", f"Could not open folder: {exc}", parent=win)

        ttk.Button(buttons, text="Open output folder", style="Primary.TButton", command=open_folder).pack(side="left")
        ttk.Button(buttons, text="Close", command=win.destroy).pack(side="left", padx=(6, 0))
        win.bind("<Escape>", lambda _e: win.destroy())
        win.bell()
        win.lift()
        win.focus_force()

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

    def _on_close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            if not messagebox.askyesno("Mosaic", "A mosaic is still building. Cancel it and close?", parent=self.root):
                return
            self._cancel()
        self._save_settings()
        self.root.destroy()


def main(tiles: list[Path] | None = None, output: Path | None = None) -> int:
    root = tk.Tk()
    MosaicWindow(root, tiles, output)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
