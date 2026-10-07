"""Small dark Tk window for benchmarking a GPUStacker master against a reference stack."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .astro_dark_theme import DARK_BORDER, DARK_FIELD, DARK_TEXT, configure_dark_theme
from .compare import CompareResult, compare_many, write_report

STACK_TYPES = [("Stacks", "*.fit *.fits *.fts *.xisf"), ("All files", "*.*")]
GOOD, BAD, NEUTRAL = "#5fcf80", "#e06c75", "#9aa3b2"


def _verdict(ratio: float, higher_is_better: bool = True, tol: float = 0.02) -> tuple[str, str]:
    if ratio != ratio:  # NaN
        return "n/a", NEUTRAL
    if abs(ratio - 1.0) <= tol:
        return "= same", NEUTRAL
    better = (ratio > 1.0) == higher_is_better
    return ("+ GPU better" if better else "- PI better"), (GOOD if better else BAD)


def _rows(r: CompareResult) -> list[tuple[str, str, str, str, str, str]]:
    """(metric, gpu, ref, ratio, verdict, colour) for the result table."""

    rows = []

    def add(metric, gpu, ref, ratio, hib=True, tol=0.02, fmt="{:.2f}"):
        text, colour = _verdict(ratio, hib, tol)
        rows.append((metric, fmt.format(gpu) if gpu is not None else "—", fmt.format(ref) if ref is not None else "—", f"{ratio:.3f}" if ratio == ratio else "—", text, colour))

    add("Stars detected @5σ", r.stars_gpu, r.stars_ref, r.stars_gpu / max(r.stars_ref, 1), fmt="{:d}")
    add("Median FWHM [px]  (lower = sharper)", r.fwhm_gpu, r.fwhm_ref, r.fwhm_ref / r.fwhm_gpu if r.fwhm_gpu else float("nan"))
    add("Faint-star SNR (aperture, 10th pct)", r.faint_peak_snr_gpu, r.faint_peak_snr_ref, r.faint_peak_snr_gpu / r.faint_peak_snr_ref if r.faint_peak_snr_ref else float("nan"), fmt="{:.1f}")
    add("Matched-star SNR (aperture, median)", r.matched_peak_snr_gpu, r.matched_peak_snr_ref, r.peak_snr_ratio, fmt="{:.1f}")
    add("   …faintest quintile", r.matched_peak_snr_faint_gpu, r.matched_peak_snr_faint_ref, r.peak_snr_ratio_faint, fmt="{:.1f}")
    add("Star / nebula flux balance", None, None, r.star_to_nebula_flux_ratio, tol=0.05)
    for band, ratio in r.nebula_snr_ratio.items():
        add(f"Nebula SNR (15px mean), ref SNR in {band}", r.nebula_snr_gpu.get(band), r.nebula_snr_ref.get(band), ratio, tol=0.03, fmt="{:.1f}")
    for region in sorted(set(r.regional_fwhm_gpu) | set(r.regional_fwhm_ref)):
        gpu, ref = r.regional_fwhm_gpu.get(region), r.regional_fwhm_ref.get(region)
        ratio = ref / gpu if gpu and ref else float("nan")
        add(f"FWHM {region} [px]", gpu, ref, ratio)
    for region in sorted(r.regional_matched_stars):
        gpu, ref = r.regional_star_snr_gpu.get(region), r.regional_star_snr_ref.get(region)
        ratio = gpu / ref if gpu is not None and ref else float("nan")
        count = r.regional_matched_stars[region]
        add(f"Matched-star SNR {region} (n={count})", gpu, ref, ratio, fmt="{:.1f}")
    rows.append(("Stars only in reference: median SNR", "—", f"{r.ref_only_star_median_snr:.1f}", "—", f"{r.stars_ref - r.stars_matched} stars", NEUTRAL))
    rows.append(("Registration rms [px] / common area", "—", "—", f"{r.alignment_rms_px:.2f} / {r.common_fraction:.0%}", "", NEUTRAL))
    return rows


class CompareWindow:
    def __init__(self, root: tk.Misc, gpu: Path | None = None, reference: Path | None = None, gpus: list[Path] | None = None) -> None:
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.result: CompareResult | None = None
        self.results: list[CompareResult] = []
        self.candidate_paths: list[Path] = []
        root.title("GPUStacker — compare masters")
        if isinstance(root, (tk.Tk, tk.Toplevel)):
            root.geometry("1120x620")
            root.minsize(900, 480)
        if isinstance(root, tk.Toplevel):
            root.transient(root.master)
        configure_dark_theme(root)
        self._build()
        root.lift()
        initial = list(gpus or [])
        if gpu is not None:
            initial.insert(0, gpu)
        self._add_candidate_paths(initial)
        if reference:
            self.ref_var.set(str(reference))
        root.after(100, self._poll)

    def _add_candidate_paths(self, paths: list[Path] | tuple[str, ...] | None = None) -> None:
        if paths is None:
            paths = filedialog.askopenfilenames(title="Select GPUStacker masters", filetypes=STACK_TYPES, parent=self.root)
            self.root.lift()
        for path in paths:
            candidate = Path(path)
            if candidate not in self.candidate_paths:
                self.candidate_paths.append(candidate)
        self._render_candidates()
        if self.candidate_paths and not self.ref_var.get():
            self._auto_pair()

    def _render_candidates(self) -> None:
        self.candidate_list.delete(0, "end")
        for path in self.candidate_paths:
            self.candidate_list.insert("end", str(path))

    def _remove_candidates(self) -> None:
        for index in reversed(self.candidate_list.curselection()):
            del self.candidate_paths[index]
        self._render_candidates()

    def _clear_candidates(self) -> None:
        self.candidate_paths.clear()
        self._render_candidates()

    def _browse_reference(self) -> None:
        chosen = filedialog.askopenfilename(title="Select reference stack", filetypes=STACK_TYPES, parent=self.root)
        self.root.lift()
        if chosen:
            self.ref_var.set(chosen)

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(3, weight=1)

        candidates = ttk.LabelFrame(outer, text="GPUStacker masters", padding=6)
        candidates.grid(row=0, column=0, columnspan=4, sticky="ew")
        candidates.columnconfigure(0, weight=1)
        self.candidate_list = tk.Listbox(candidates, selectmode=tk.EXTENDED, height=3, exportselection=False, bg=DARK_FIELD, fg=DARK_TEXT, relief="flat", highlightthickness=1, highlightbackground=DARK_BORDER, font=("Consolas", 9))
        self.candidate_list.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=(0, 8))
        ttk.Button(candidates, text="Add files…", command=self._add_candidate_paths).grid(row=0, column=1, sticky="ew", pady=1)
        ttk.Button(candidates, text="Remove selected", command=self._remove_candidates).grid(row=1, column=1, sticky="ew", pady=1)
        ttk.Button(candidates, text="Clear", command=self._clear_candidates).grid(row=2, column=1, sticky="ew", pady=1)

        self.ref_var = tk.StringVar()
        ttk.Label(outer, text="Reference stack").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(outer, textvariable=self.ref_var).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(outer, text="…", width=3, command=self._browse_reference).grid(row=1, column=2)
        bar = ttk.Frame(outer)
        bar.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 6))
        self.run_btn = ttk.Button(bar, text="Compare", style="Primary.TButton", command=self._start)
        self.run_btn.pack(side="left")
        self.save_btn = ttk.Button(bar, text="Save JSON…", command=self._save, state="disabled")
        self.save_btn.pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="Select one or more masters and one matching reference stack")
        ttk.Label(bar, textvariable=self.status_var, foreground=NEUTRAL).pack(side="left", padx=8)

        self.tree = ttk.Treeview(outer, columns=("metric", "reference"), show="headings", height=14)
        self.tree.heading("metric", text="Metric")
        self.tree.column("metric", width=300, anchor="w", stretch=True)
        self.tree.heading("reference", text="Reference")
        self.tree.column("reference", width=110, anchor="e")
        self.tree.grid(row=3, column=0, columnspan=3, sticky="nsew")
        vertical = ttk.Scrollbar(outer, orient="vertical", command=self.tree.yview)
        vertical.grid(row=3, column=3, sticky="ns")
        horizontal = ttk.Scrollbar(outer, orient="horizontal", command=self.tree.xview)
        horizontal.grid(row=4, column=0, columnspan=3, sticky="ew")
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.log = tk.Text(outer, height=4, bg=DARK_FIELD, fg=DARK_TEXT, relief="flat", highlightthickness=1, highlightbackground=DARK_BORDER, state="disabled", font=("Consolas", 9), wrap="word")
        self.log.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(6, 0))

    def _auto_pair(self) -> None:
        """Pick the matching reference for the first candidate from a sibling Pix Stacks folder."""

        if not self.candidate_paths:
            return
        gpu = self.candidate_paths[0]
        pix = gpu.parent / "Pix Stacks"
        if not pix.is_dir():
            return
        panel = next((tok for tok in gpu.stem.replace("IC_1805_", "").split("_") if "-" in tok), None)
        drizzle = "_drizzle" in gpu.stem
        current = Path(self.ref_var.get()).name if self.ref_var.get() else ""
        if current and (panel is None or panel in current) and (("drizzle" in current) == drizzle):
            return
        for cand in sorted(pix.iterdir()):
            name = cand.name
            if panel and panel in name and (("drizzle" in name) == drizzle):
                self.ref_var.set(str(cand))
                return

    def _start(self) -> None:
        self._auto_pair()
        candidates = list(self.candidate_paths)
        reference = Path(self.ref_var.get()) if self.ref_var.get() else None
        if not candidates or any(not path.is_file() for path in candidates) or reference is None or not reference.is_file():
            messagebox.showerror("Compare", "Select existing candidate masters and a reference stack", parent=self.root)
            return
        self.run_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.tree.delete(*self.tree.get_children())
        self._log_clear()
        self.status_var.set(f"Comparing {len(candidates)} master(s) against {reference.name}…")
        threading.Thread(target=self._work, args=(candidates, reference), daemon=True).start()

    def _work(self, candidates: list[Path], reference: Path) -> None:
        try:
            results = compare_many(candidates, reference, lambda message: self.events.put(("log", message)))
            self.events.put(("done", results))
        except Exception as exc:  # surface in the window
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))

    def _populate_results(self, results: list[CompareResult]) -> None:
        rows_by_result = [_rows(result) for result in results]
        metric_names = list(dict.fromkeys(row[0] for rows in rows_by_result for row in rows))
        lookup = [{row[0]: row for row in rows} for rows in rows_by_result]
        candidate_names = [path.name for path in self.candidate_paths]
        candidate_names.extend(f"Candidate {index + 1}" for index in range(len(candidate_names), len(results)))
        columns = ("metric", "reference", *(f"candidate_{i}" for i in range(len(results))))
        self.tree.configure(columns=columns)
        self.tree.heading("metric", text="Metric")
        self.tree.column("metric", width=300, anchor="w", stretch=True)
        self.tree.heading("reference", text="Reference")
        self.tree.column("reference", width=110, anchor="e", stretch=False)
        for index, name in enumerate(candidate_names):
            column = f"candidate_{index}"
            self.tree.heading(column, text=name)
            self.tree.column(column, width=190, anchor="e", stretch=False)

        for metric in metric_names:
            reference_value = next((rows[metric][2] for rows in lookup if metric in rows), "—")
            values = [metric, reference_value]
            for rows in lookup:
                row = rows.get(metric)
                if row is None:
                    values.append("—")
                elif row[1] != "—":
                    ratio = f" ({row[3]}x)" if row[3] != "—" else ""
                    values.append(f"{row[1]}{ratio}")
                else:
                    values.append(row[3] if row[3] != "—" else row[4] or "—")
            self.tree.insert("", "end", values=values)

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "done":
                    self.results = payload
                    self.result = payload[0] if len(payload) == 1 else None
                    self._populate_results(payload)
                    wins = sum(row[5] == GOOD for result in payload for row in _rows(result))
                    losses = sum(row[5] == BAD for result in payload for row in _rows(result))
                    self.status_var.set(f"Compared {len(payload)} masters — {wins} metrics above and {losses} below reference")
                    self.run_btn.configure(state="normal")
                    self.save_btn.configure(state="normal")
                elif kind == "error":
                    self._log(payload)
                    self.status_var.set("Failed")
                    self.run_btn.configure(state="normal")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _save(self) -> None:
        if not self.results:
            return
        default = self.candidate_paths[0].with_suffix(".compare.json").name if len(self.results) == 1 else "compare_all.compare.json"
        chosen = filedialog.asksaveasfilename(defaultextension=".json", initialfile=default, filetypes=[("JSON", "*.json")], parent=self.root)
        self.root.lift()
        if chosen:
            write_report(self.results[0] if len(self.results) == 1 else self.results, Path(chosen))
            self._log(f"Saved {chosen}")

    def _log_clear(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


def main() -> int:
    root = tk.Tk()
    CompareWindow(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
