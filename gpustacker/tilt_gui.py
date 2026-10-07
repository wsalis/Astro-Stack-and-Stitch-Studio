"""Tilt inspector: per-frame FWHM / star-shape map of the sensor plus a session trend."""

from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .astro_dark_theme import DARK_BG, DARK_BORDER, DARK_FIELD, DARK_SURFACE, DARK_TEXT, configure_dark_theme
from .io import discover_frames
from .tilt import TiltMap, analyse_tilt, describe, frame_scatter, session_median, write_tilt_csv

FRAME_TYPES = [("Light frames", "*.fit *.fits *.fts *.xisf"), ("All files", "*.*")]
NEUTRAL = "#9aa3b2"
SERIES = (("TL", "#e06c75"), ("TR", "#e5c07b"), ("BL", "#56b6c2"), ("BR", "#61afef"), ("C", "#e8eaed"))
SESSION_KEY = "<< session median >>"


def _heat(ratio: float) -> str:
    """Cell colour for FWHM / sharpest-cell FWHM: 1.00 green, 1.15 yellow, >= 1.30 red."""

    if not np.isfinite(ratio):
        return DARK_SURFACE
    t = min(max((ratio - 1.0) / 0.30, 0.0), 1.0)
    if t < 0.5:
        r, g, b = _lerp((0x3f, 0xb9, 0x50), (0xe5, 0xc0, 0x7b), t * 2.0)
    else:
        r, g, b = _lerp((0xe5, 0xc0, 0x7b), (0xe0, 0x6c, 0x75), (t - 0.5) * 2.0)
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))  # type: ignore[return-value]


class TiltWindow:
    def __init__(self, root: tk.Misc, lights: list[Path] | None = None) -> None:
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.frames: list[Path] = []
        self.maps: dict[Path, TiltMap] = {}
        self.session: TiltMap | None = None
        self.scatter = float("nan")
        self.busy = False
        root.title("GPUStacker — tilt inspector")
        if isinstance(root, (tk.Tk, tk.Toplevel)):
            root.geometry("1180x760")
            root.minsize(900, 600)
        configure_dark_theme(root)
        self._build()
        if lights:
            self._add(lights)
        root.after(100, self._poll)

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(1, weight=1)

        bar = ttk.Frame(outer)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Button(bar, text="Add frames…", command=self._browse_files).pack(side="left")
        ttk.Button(bar, text="Add folder…", command=self._browse_folder).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Clear", command=self._clear).pack(side="left", padx=(6, 12))
        ttk.Label(bar, text="Grid").pack(side="left")
        self.cells_var = tk.IntVar(value=4)
        ttk.Spinbox(bar, from_=3, to=6, textvariable=self.cells_var, width=3, state="readonly").pack(side="left", padx=(4, 12))
        self.run_btn = ttk.Button(bar, text="Analyse", style="Primary.TButton", command=self._start)
        self.run_btn.pack(side="left")
        self.save_btn = ttk.Button(bar, text="Save CSV…", command=self._save, state="disabled")
        self.save_btn.pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="Add light frames (unregistered subs), then Analyse")
        ttk.Label(bar, textvariable=self.status_var, foreground=NEUTRAL).pack(side="left", padx=8)

        left = ttk.Frame(outer)
        left.grid(row=1, column=0, sticky="ns", padx=(0, 10))
        left.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(left, width=44, bg=DARK_FIELD, fg=DARK_TEXT, selectbackground="#285f9e", selectforeground="#ffffff", relief="flat", highlightthickness=1, highlightbackground=DARK_BORDER, exportselection=False, font=("Consolas", 9), activestyle="none")
        self.listbox.grid(row=0, column=0, sticky="ns")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scroll.set)
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self._redraw())

        right = ttk.Frame(outer)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=0)
        self.map_canvas = tk.Canvas(right, bg=DARK_FIELD, highlightthickness=1, highlightbackground=DARK_BORDER)
        self.map_canvas.grid(row=0, column=0, sticky="nsew")
        self.map_canvas.bind("<Configure>", lambda _e: self._draw_map())
        self.text = tk.Text(right, width=52, bg=DARK_FIELD, fg=DARK_TEXT, relief="flat", highlightthickness=1, highlightbackground=DARK_BORDER, state="disabled", font=("Consolas", 9), wrap="word", padx=8, pady=6)
        self.text.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.trend_canvas = tk.Canvas(right, bg=DARK_FIELD, highlightthickness=1, highlightbackground=DARK_BORDER, height=170)
        self.trend_canvas.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        self.trend_canvas.bind("<Configure>", lambda _e: self._draw_trend())
        self.trend_canvas.bind("<Button-1>", self._trend_click)

    # ------------------------------------------------------------------ frame list

    def _browse_files(self) -> None:
        chosen = filedialog.askopenfilenames(filetypes=FRAME_TYPES)
        if chosen:
            self._add([Path(p) for p in chosen])

    def _browse_folder(self) -> None:
        chosen = filedialog.askdirectory()
        if chosen:
            self._add(discover_frames(chosen))

    def _add(self, paths: list[Path]) -> None:
        known = {p.resolve() for p in self.frames}
        for p in paths:
            if p.resolve() not in known:
                self.frames.append(p)
                known.add(p.resolve())
        self._refresh_list()
        self.status_var.set(f"{len(self.frames)} frame(s) loaded; {len(self.maps)} analysed")

    def _clear(self) -> None:
        if self.busy:
            return
        self.frames.clear()
        self.maps.clear()
        self.session = None
        self._refresh_list()
        self._redraw()
        self.save_btn.configure(state="disabled")
        self.status_var.set("Cleared")

    def _refresh_list(self) -> None:
        sel = self._selected_key()
        self.listbox.delete(0, "end")
        if self.session is not None:
            self.listbox.insert("end", f"{SESSION_KEY}  tilt {self.session.tilt_px:.2f} px {self.session.soft_side}")
        for p in self.frames:
            m = self.maps.get(p)
            tail = p.stem if len(p.stem) <= 24 else "…" + p.stem[-23:]
            label = tail if m is None else f"{tail:<24} {m.median_fwhm:5.2f}px  tilt {m.tilt_px:4.2f} {m.soft_side}"
            self.listbox.insert("end", label)
        if sel is not None:
            self._select_key(sel)
        elif self.listbox.size():
            self.listbox.selection_set(0)

    def _selected_key(self) -> Path | str | None:
        sel = self.listbox.curselection()
        if not sel:
            return None
        idx = int(sel[0])
        if self.session is not None:
            if idx == 0:
                return SESSION_KEY
            idx -= 1
        return self.frames[idx] if idx < len(self.frames) else None

    def _select_key(self, key: Path | str) -> None:
        offset = 1 if self.session is not None else 0
        idx = 0 if key == SESSION_KEY else (self.frames.index(key) + offset if key in self.frames else None)  # type: ignore[arg-type]
        if idx is None:
            return
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(idx)
        self.listbox.see(idx)

    def _current_map(self) -> TiltMap | None:
        key = self._selected_key()
        if key == SESSION_KEY:
            return self.session
        return self.maps.get(key) if isinstance(key, Path) else None

    # ------------------------------------------------------------------ analysis

    def _start(self) -> None:
        if self.busy:
            return
        todo = [p for p in self.frames if p not in self.maps or self.maps[p].cells != int(self.cells_var.get())]
        if not todo:
            messagebox.showinfo("Tilt inspector", "Nothing to analyse — add frames first")
            return
        self.busy = True
        self.run_btn.configure(state="disabled")
        self.status_var.set(f"Analysing {len(todo)} frame(s)…")
        threading.Thread(target=self._work, args=(todo, int(self.cells_var.get())), daemon=True).start()

    def _work(self, todo: list[Path], cells: int) -> None:
        for i, p in enumerate(todo):
            try:
                self.events.put(("map", (p, analyse_tilt(p, cells))))
            except Exception as exc:  # keep going; report per frame
                self.events.put(("fail", (p, f"{type(exc).__name__}: {exc}")))
            self.events.put(("progress", (i + 1, len(todo))))
        self.events.put(("done", None))

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "map":
                    p, m = payload
                    self.maps[p] = m
                    self._refresh_list()
                    if self._selected_key() in (None, p):
                        self._redraw()
                elif kind == "fail":
                    p, msg = payload
                    self.status_var.set(f"{p.name}: {msg}")
                elif kind == "progress":
                    done, total = payload
                    self.status_var.set(f"Analysing {done}/{total}…")
                elif kind == "done":
                    self.busy = False
                    self.run_btn.configure(state="normal")
                    ordered = [self.maps[p] for p in self.frames if p in self.maps]
                    self.session = session_median(ordered) if len(ordered) >= 2 else None
                    self.scatter = frame_scatter(ordered)
                    self._refresh_list()
                    if self.session is not None:
                        self._select_key(SESSION_KEY)
                    self._redraw()
                    self.save_btn.configure(state="normal" if ordered else "disabled")
                    self.status_var.set(f"Done — {len(ordered)} frame(s) analysed")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _save(self) -> None:
        ordered = [self.maps[p] for p in self.frames if p in self.maps]
        if not ordered:
            return
        rows = ordered + ([self.session] if self.session is not None else [])
        default = (self.frames[0].parent / "tilt_report.csv").name
        chosen = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=default, initialdir=str(self.frames[0].parent), filetypes=[("CSV", "*.csv")])
        if chosen:
            write_tilt_csv(rows, Path(chosen))
            self.status_var.set(f"Saved {chosen}")

    # ------------------------------------------------------------------ drawing

    def _redraw(self) -> None:
        self._draw_map()
        self._draw_text()
        self._draw_trend()

    def _draw_map(self) -> None:
        cv = self.map_canvas
        cv.delete("all")
        m = self._current_map()
        W, H = cv.winfo_width(), cv.winfo_height()
        if m is None or W < 50 or H < 50:
            cv.create_text(W // 2, H // 2, text="FWHM map appears here after Analyse", fill=NEUTRAL, font=("Segoe UI", 11))
            return
        margin = 36
        aspect = m.shape[1] / max(m.shape[0], 1)
        avail_w, avail_h = W - 2 * margin, H - 2 * margin
        if avail_w / avail_h > aspect:
            draw_h = avail_h
            draw_w = draw_h * aspect
        else:
            draw_w = avail_w
            draw_h = draw_w / aspect
        x0 = (W - draw_w) / 2
        y0 = (H - draw_h) / 2
        n = m.cells
        cw, ch = draw_w / n, draw_h / n
        best = float(np.nanmin(m.fwhm)) if np.isfinite(m.fwhm).any() else float("nan")
        big = ("Segoe UI", max(9, int(min(cw, ch) / 7)), "bold")
        small = ("Segoe UI", max(8, int(min(cw, ch) / 11)))
        for r in range(n):
            for c in range(n):
                cx0, cy0 = x0 + c * cw, y0 + r * ch
                v = float(m.fwhm[r, c])
                fill = _heat(v / best) if np.isfinite(v) and np.isfinite(best) and best > 0 else DARK_SURFACE
                cv.create_rectangle(cx0, cy0, cx0 + cw, cy0 + ch, fill=fill, outline=DARK_BG, width=2)
                if not np.isfinite(v):
                    cv.create_text(cx0 + cw / 2, cy0 + ch / 2, text=f"n/a\n{int(m.count[r, c])} stars", fill=DARK_BG, font=small, justify="center")
                    continue
                self._ellipse(cv, cx0 + cw / 2, cy0 + ch / 2, min(cw, ch) * 0.30, float(m.elongation[r, c]), float(m.angle_deg[r, c]), float(m.coherence[r, c]))
                cv.create_text(cx0 + cw / 2, cy0 + ch * 0.22, text=f"{v:.2f}", fill="#101216", font=big)
                e = m.elongation[r, c]
                cv.create_text(cx0 + cw / 2, cy0 + ch * 0.82, text=f"e {e:.2f}  n {int(m.count[r, c])}" if np.isfinite(e) else f"n {int(m.count[r, c])}", fill="#101216", font=small)
        cv.create_rectangle(x0, y0, x0 + draw_w, y0 + draw_h, outline=DARK_BORDER, width=1)
        cv.create_text(x0, y0 - 10, text="sensor top-left", anchor="sw", fill=NEUTRAL, font=small)
        cv.create_text(W / 2, H - 6, anchor="s", width=W - 20, justify="center", text=f"{m.name}   FWHM px; colour vs sharpest cell (green 1.00, yellow 1.15, red 1.30+); ellipse = median star shape, axis ratio x3", fill=NEUTRAL, font=small)
        # tilt arrow from centre toward the soft side
        if np.isfinite(m.tilt_px) and np.isfinite(m.median_fwhm) and m.median_fwhm > 0 and m.tilt_px / m.median_fwhm >= 0.03:
            ang = math.radians(m.tilt_angle_deg)
            length = min(draw_w, draw_h) * min(0.45, 0.15 + 2.0 * m.tilt_px / m.median_fwhm)
            cx, cy = x0 + draw_w / 2, y0 + draw_h / 2
            cv.create_line(cx, cy, cx + length * math.cos(ang), cy + length * math.sin(ang), fill="#ffffff", width=3, arrow="last", arrowshape=(14, 16, 6))
            cv.create_text(x0 + draw_w, y0 - 10, text=f"arrow: soft side {m.soft_side}, {m.tilt_px:.2f} px across sensor", anchor="se", fill="#ffffff", font=small)

    @staticmethod
    def _ellipse(cv: tk.Canvas, cx: float, cy: float, radius: float, elong: float, angle_deg: float, coherence: float) -> None:
        if not np.isfinite(elong):
            return
        vis = 1.0 + 3.0 * max(elong - 1.0, 0.0)  # exaggerate so a 1.1 axis ratio is visible
        a, b = radius * math.sqrt(vis), radius / math.sqrt(vis)
        th = math.radians(angle_deg) if np.isfinite(angle_deg) else 0.0
        pts = []
        for i in range(36):
            t = 2 * math.pi * i / 36
            ex, ey = a * math.cos(t), b * math.sin(t)
            pts.extend((cx + ex * math.cos(th) - ey * math.sin(th), cy + ex * math.sin(th) + ey * math.cos(th)))
        dash = () if (np.isfinite(coherence) and coherence >= 0.4) else (4, 3)
        cv.create_polygon(*pts, outline="#101216", fill="", width=2, dash=dash, smooth=True)

    def _draw_text(self) -> None:
        m = self._current_map()
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        if m is not None:
            scatter = self.scatter if m is self.session else float("nan")
            self.text.insert("end", "\n".join(describe(m, scatter)))
        self.text.configure(state="disabled")

    def _draw_trend(self) -> None:
        cv = self.trend_canvas
        cv.delete("all")
        W, H = cv.winfo_width(), cv.winfo_height()
        ordered = [(p, self.maps[p]) for p in self.frames if p in self.maps]
        if len(ordered) < 2 or W < 50 or H < 50:
            cv.create_text(W // 2, H // 2, text="Corner FWHM trend over the session appears here (2+ frames)", fill=NEUTRAL, font=("Segoe UI", 10))
            return
        series = {k: np.array([m.corners()[k] for _, m in ordered]) for k, _ in SERIES}
        allv = np.concatenate([v[np.isfinite(v)] for v in series.values()])
        if allv.size == 0:
            return
        lo, hi = float(allv.min()), float(allv.max())
        pad = max(0.05, (hi - lo) * 0.1)
        lo, hi = lo - pad, hi + pad
        left, right, top, bottom = 48, 12, 10, 22
        pw, ph = W - left - right, H - top - bottom
        n = len(ordered)

        def px(i: int) -> float:
            return left + (pw * i / max(n - 1, 1))

        def py(v: float) -> float:
            return top + ph * (1.0 - (v - lo) / max(hi - lo, 1e-9))

        cv.create_rectangle(left, top, left + pw, top + ph, outline=DARK_BORDER)
        for v in np.linspace(lo, hi, 4):
            y = py(v)
            cv.create_line(left, y, left + pw, y, fill="#2a2e36")
            cv.create_text(left - 6, y, text=f"{v:.2f}", anchor="e", fill=NEUTRAL, font=("Segoe UI", 8))
        for key, colour in SERIES:
            vals = series[key]
            pts = [(px(i), py(float(v))) for i, v in enumerate(vals) if np.isfinite(v)]
            if len(pts) >= 2:
                cv.create_line(*[c for pt in pts for c in pt], fill=colour, width=2 if key != "C" else 1)
        lx = left + 8
        for key, colour in SERIES:
            cv.create_line(lx, top + 10, lx + 14, top + 10, fill=colour, width=2)
            cv.create_text(lx + 18, top + 10, text=key, anchor="w", fill=colour, font=("Segoe UI", 8))
            lx += 44
        cv.create_text(left + pw / 2, H - 8, text=f"frame 1 … {n}   (corner FWHM px; lines that cross = tilt changing with pointing/flexure)", fill=NEUTRAL, font=("Segoe UI", 8))
        key = self._selected_key()
        if isinstance(key, Path) and key in self.maps:
            idx = [p for p, _ in ordered].index(key)
            x = px(idx)
            cv.create_line(x, top, x, top + ph, fill="#ffffff", dash=(3, 3))

    def _trend_click(self, event: tk.Event) -> None:
        ordered = [p for p in self.frames if p in self.maps]
        if len(ordered) < 2:
            return
        left, right = 48, 12
        pw = self.trend_canvas.winfo_width() - left - right
        idx = int(round((event.x - left) / max(pw, 1) * (len(ordered) - 1)))
        idx = min(max(idx, 0), len(ordered) - 1)
        self._select_key(ordered[idx])
        self._redraw()


def main(lights: list[Path] | None = None) -> int:
    root = tk.Tk()
    TiltWindow(root, lights)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
