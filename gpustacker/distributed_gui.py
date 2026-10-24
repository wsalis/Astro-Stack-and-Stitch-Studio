"""GUI for configuring a worker and probing a remote worker."""

from __future__ import annotations

import base64
import ctypes
import json
import logging
import os
import queue
import secrets
import socketserver
import threading
import time
import tkinter as tk
from logging.handlers import RotatingFileHandler
from ctypes import wintypes
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import APP_NAME, __version__
from .astro_dark_theme import DARK_BG, DARK_FIELD, DARK_TEXT, configure_dark_theme
from .distributed import WorkerProbeError, create_worker_server, probe_worker

SETTINGS_PATH = Path.home() / ".gpustacker" / "distributed_settings.json"
WORKER_LOG_PATH = Path.home() / ".gpustacker" / "worker.log"


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _crypt_token(data: bytes, protect: bool) -> bytes:
    if os.name != "nt":
        raise OSError("Secure token storage is available only on Windows")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    input_buffer = ctypes.create_string_buffer(data, len(data))
    input_blob = _DataBlob(len(data), ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    output_blob = _DataBlob()
    if protect:
        transform = crypt32.CryptProtectData
        transform.argtypes = [
            ctypes.POINTER(_DataBlob), wintypes.LPCWSTR, ctypes.POINTER(_DataBlob), ctypes.c_void_p,
            ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob),
        ]
        arguments = (ctypes.byref(input_blob), "GPUStacker multi-system token", None, None, None, 1, ctypes.byref(output_blob))
    else:
        transform = crypt32.CryptUnprotectData
        transform.argtypes = [
            ctypes.POINTER(_DataBlob), ctypes.POINTER(wintypes.LPWSTR), ctypes.POINTER(_DataBlob),
            ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob),
        ]
        arguments = (ctypes.byref(input_blob), None, None, None, None, 1, ctypes.byref(output_blob))
    transform.restype = wintypes.BOOL
    if not transform(*arguments):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        kernel32.LocalFree(ctypes.cast(output_blob.pbData, ctypes.c_void_p))


def _protect_token(token: str) -> str:
    return base64.b64encode(_crypt_token(token.encode("utf-8"), protect=True)).decode("ascii")


def _unprotect_token(protected: str) -> str:
    encrypted = base64.b64decode(protected, validate=True)
    return _crypt_token(encrypted, protect=False).decode("utf-8")


class DistributedWindow:
    def __init__(self, root: tk.Misc, shared_store: Path | None = None) -> None:
        self.root = root
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.server: socketserver.ThreadingTCPServer | None = None
        self._closing = False
        root.title(f"{APP_NAME} — Multi-system setup")
        if isinstance(root, (tk.Tk, tk.Toplevel)):
            root.geometry("820x700")
            root.minsize(700, 590)
        configure_dark_theme(root)

        initial_store = str(shared_store) if shared_store is not None else ""
        self.worker_store_var = tk.StringVar(value=initial_store)
        self.coordinator_store_var = tk.StringVar(value=initial_store)
        self.token_var = tk.StringVar()
        self.show_token_var = tk.BooleanVar(value=False)
        self.remember_token_var = tk.BooleanVar(value=os.name == "nt")
        self.worker_bind_var = tk.StringVar(value="This computer only")
        self.worker_port_var = tk.StringVar(value="47861")
        self.worker_rows_var = tk.StringVar(value="8")
        self.host_var = tk.StringVar()
        self.probe_port_var = tk.StringVar(value="47861")
        self.probe_timeout_var = tk.StringVar(value="10")
        self.require_cuda_var = tk.BooleanVar(value=True)
        self.worker_execution_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Capability checks only; worker execution is opt-in.")

        self._build()
        self._load_settings()
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(100, self._poll)

    def _build(self) -> None:
        canvas = tk.Canvas(self.root, bg=DARK_BG, highlightthickness=0)
        page_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        page_scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=page_scrollbar.set)
        outer = ttk.Frame(canvas, padding=12)
        content = canvas.create_window((0, 0), window=outer, anchor="nw")

        def resize_content(_event) -> None:
            canvas.itemconfigure(content, width=canvas.winfo_width(), height=max(canvas.winfo_height(), outer.winfo_reqheight()))
            canvas.configure(scrollregion=canvas.bbox("all"))

        canvas.bind("<Configure>", resize_content)
        outer.bind("<Configure>", resize_content)
        outer.columnconfigure(0, weight=1)

        security = ttk.LabelFrame(outer, text="Shared authentication token", padding=8)
        security.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        security.columnconfigure(0, weight=1)
        self.token_entry = ttk.Entry(security, textvariable=self.token_var, show="*")
        self.token_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(security, text="Generate", command=self._generate_token).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(security, text="Copy", command=self._copy_token).grid(row=0, column=2, padx=(6, 0))
        ttk.Checkbutton(security, text="Show", variable=self.show_token_var, command=self._toggle_token).grid(row=0, column=3, padx=(8, 0))
        ttk.Checkbutton(
            security,
            text="Remember token securely on this computer",
            variable=self.remember_token_var,
            state="normal" if os.name == "nt" else "disabled",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(5, 0))
        security_help = ttk.Label(
            security,
            text="Use the same random token on both computers. When enabled, Windows encrypts it for your account.",
            foreground="#9aa3b2",
            wraplength=760,
        )
        security_help.grid(row=2, column=0, columnspan=4, sticky="w", pady=(3, 0))
        security.bind("<Configure>", lambda event: security_help.configure(wraplength=max(240, event.width - 20)))

        worker = ttk.LabelFrame(outer, text="Worker computer", padding=8)
        worker.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        worker.columnconfigure(1, weight=1)
        ttk.Label(worker, text="Registered store").grid(row=0, column=0, sticky="w")
        ttk.Entry(worker, textvariable=self.worker_store_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(worker, text="Browse…", command=lambda: self._browse_store(self.worker_store_var)).grid(row=0, column=2)
        ttk.Label(worker, text="Listen on").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            worker,
            textvariable=self.worker_bind_var,
            values=("This computer only", "Other computers on trusted LAN"),
            state="readonly",
            width=34,
        ).grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(worker, text="Port").grid(row=1, column=2, sticky="e", pady=(6, 0))
        ttk.Entry(worker, textvariable=self.worker_port_var, width=8).grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Label(worker, text="Benchmark rows per frame").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(worker, textvariable=self.worker_rows_var, width=8).grid(row=2, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Checkbutton(
            worker,
            text="Allow authenticated stack-band execution (experimental)",
            variable=self.worker_execution_var,
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))
        worker_actions = ttk.Frame(worker)
        worker_actions.grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.start_worker_button = ttk.Button(worker_actions, text="Start worker service", command=self._start_worker)
        self.start_worker_button.pack(side="left")
        self.stop_worker_button = ttk.Button(worker_actions, text="Stop service", command=self._stop_worker, state="disabled")
        self.stop_worker_button.pack(side="left", padx=6)
        worker_help = ttk.Label(
            worker,
            text="For remote access, choose trusted LAN and allow this port through the worker's firewall only from the coordinator. Traffic is not encrypted.",
            foreground="#9aa3b2",
            wraplength=760,
        )
        worker_help.grid(row=5, column=0, columnspan=4, sticky="w", pady=(6, 0))
        worker.bind("<Configure>", lambda event: worker_help.configure(wraplength=max(240, event.width - 20)))

        coordinator = ttk.LabelFrame(outer, text="Coordinator computer", padding=8)
        coordinator.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        coordinator.columnconfigure(1, weight=1)
        ttk.Label(coordinator, text="Worker host").grid(row=0, column=0, sticky="w")
        ttk.Entry(coordinator, textvariable=self.host_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(coordinator, text="Port").grid(row=0, column=2, sticky="e")
        ttk.Entry(coordinator, textvariable=self.probe_port_var, width=8).grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Label(coordinator, text="Coordinator's path to the same registered store").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(coordinator, textvariable=self.coordinator_store_var).grid(row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(coordinator, text="Browse…", command=lambda: self._browse_store(self.coordinator_store_var)).grid(row=1, column=3, pady=(6, 0))
        ttk.Checkbutton(coordinator, text="Require a CUDA device", variable=self.require_cuda_var).grid(row=2, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(coordinator, text="Store check timeout (minutes)").grid(row=2, column=2, sticky="e", pady=(6, 0))
        ttk.Spinbox(coordinator, textvariable=self.probe_timeout_var, from_=1, to=120, width=6).grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(6, 0))
        self.probe_button = ttk.Button(coordinator, text="Check worker", style="Primary.TButton", command=self._probe_worker)
        self.probe_button.grid(row=3, column=3, sticky="e", pady=(6, 0))

        result = ttk.LabelFrame(outer, text="Status", padding=8)
        result.grid(row=3, column=0, sticky="nsew")
        outer.rowconfigure(3, weight=1)
        result.columnconfigure(0, weight=1)
        result.rowconfigure(1, weight=1)
        status_label = ttk.Label(result, textvariable=self.status_var, wraplength=760)
        status_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        result.bind("<Configure>", lambda event: status_label.configure(wraplength=max(240, event.width - 20)))
        self.result_text = tk.Text(
            result,
            height=9,
            bg=DARK_FIELD,
            fg=DARK_TEXT,
            insertbackground=DARK_TEXT,
            relief="flat",
            highlightthickness=1,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
        )
        self.result_text.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(result, orient="vertical", command=self.result_text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.result_text.configure(yscrollcommand=scrollbar.set)

    def _browse_store(self, variable: tk.StringVar) -> None:
        selected = filedialog.askdirectory(title="Select registered-frame store", parent=self.root, mustexist=True)
        if selected:
            variable.set(selected)

    def _generate_token(self) -> None:
        self.token_var.set(secrets.token_urlsafe(32))
        self.status_var.set("Generated a new token. Use this same value on both computers.")

    def _copy_token(self) -> None:
        token = self.token_var.get()
        if not token:
            messagebox.showinfo(APP_NAME, "Generate or enter a token first.", parent=self.root)
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(token)
        self.status_var.set("Token copied. Share it only through a trusted channel.")

    def _toggle_token(self) -> None:
        self.token_entry.configure(show="" if self.show_token_var.get() else "*")

    def _save_settings(self) -> None:
        settings = {
            "worker_store": self.worker_store_var.get(),
            "worker_bind": self.worker_bind_var.get(),
            "worker_port": self.worker_port_var.get(),
            "worker_rows": self.worker_rows_var.get(),
            "worker_execution": self.worker_execution_var.get(),
            "worker_host": self.host_var.get(),
            "probe_port": self.probe_port_var.get(),
            "probe_timeout_minutes": self.probe_timeout_var.get(),
            "coordinator_store": self.coordinator_store_var.get(),
            "require_cuda": self.require_cuda_var.get(),
            "remember_token": self.remember_token_var.get(),
        }
        if self.remember_token_var.get() and self.token_var.get():
            try:
                settings["protected_token"] = _protect_token(self.token_var.get())
            except (OSError, AttributeError) as exc:
                self.status_var.set(f"Could not securely remember token: {exc}")
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        except (OSError, tk.TclError, TypeError) as exc:
            self.status_var.set(f"Could not save multi-system settings: {exc}")

    def _load_settings(self) -> None:
        if not SETTINGS_PATH.is_file():
            return
        try:
            settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(settings, dict):
            return
        variables = {
            "worker_store": self.worker_store_var,
            "worker_bind": self.worker_bind_var,
            "worker_port": self.worker_port_var,
            "worker_rows": self.worker_rows_var,
            "worker_execution": self.worker_execution_var,
            "worker_host": self.host_var,
            "probe_port": self.probe_port_var,
            "probe_timeout_minutes": self.probe_timeout_var,
            "coordinator_store": self.coordinator_store_var,
            "require_cuda": self.require_cuda_var,
            "remember_token": self.remember_token_var,
        }
        for name, variable in variables.items():
            if name in settings:
                try:
                    variable.set(settings[name])
                except (tk.TclError, ValueError):
                    pass
        if self.remember_token_var.get() and settings.get("protected_token"):
            try:
                self.token_var.set(_unprotect_token(settings["protected_token"]))
            except (OSError, ValueError, UnicodeDecodeError, AttributeError) as exc:
                self.status_var.set(f"Saved token could not be restored; enter it again. ({exc})")

    def _start_worker(self) -> None:
        token = self.token_var.get()
        store = Path(self.worker_store_var.get())
        try:
            if len(token.encode("utf-8")) < 32:
                raise ValueError("Generate a token of at least 32 bytes first.")
            if not store.is_dir():
                raise ValueError("Select an accessible registered-store folder.")
            port = int(self.worker_port_var.get())
            rows = int(self.worker_rows_var.get())
            if rows < 1:
                raise ValueError("Rows to read must be at least 1.")
            logger = logging.getLogger("gpustacker.worker")
            if not logger.handlers:
                WORKER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                handler = RotatingFileHandler(WORKER_LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
                logger.propagate = False
            bind_host = "127.0.0.1" if self.worker_bind_var.get() == "This computer only" else "0.0.0.0"
            server = create_worker_server(
                bind_host,
                port,
                store,
                token,
                rows,
                status_callback=self._worker_status,
                enable_task_execution=self.worker_execution_var.get(),
            )
        except (OSError, ValueError, WorkerProbeError) as exc:
            messagebox.showerror(APP_NAME, str(exc), parent=self.root)
            return

        self._save_settings()
        self.server = server
        self.start_worker_button.configure(state="disabled")
        self.stop_worker_button.configure(state="normal")
        self.status_var.set(f"Worker service listening on {server.server_address[0]}:{server.server_address[1]}.")
        threading.Thread(target=self._serve_worker, args=(server,), daemon=True).start()

    def _worker_status(self, message: str) -> None:
        logging.getLogger("gpustacker.worker").info(message)
        self.events.put(("worker_status", f"{time.strftime('%H:%M:%S')} {message}"))

    def _serve_worker(self, server: socketserver.ThreadingTCPServer) -> None:
        try:
            server.serve_forever()
        finally:
            server.server_close()
            self.events.put(("worker_stopped", server))

    def _stop_worker(self) -> None:
        server = self.server
        if server is not None:
            self.status_var.set("Stopping worker service…")
            threading.Thread(target=server.shutdown, daemon=True).start()

    def _probe_worker(self) -> None:
        try:
            token = self.token_var.get()
            if len(token.encode("utf-8")) < 32:
                raise ValueError("Generate or enter the same token used by the worker.")
            host = self.host_var.get().strip()
            if not host:
                raise ValueError("Enter the worker computer's name or IP address.")
            store = Path(self.coordinator_store_var.get())
            if not store.is_dir():
                raise ValueError("Select the coordinator's accessible path to the same registered store.")
            port = int(self.probe_port_var.get())
            if not 1 <= port <= 65535:
                raise ValueError("Port must be between 1 and 65535.")
            timeout_minutes = int(self.probe_timeout_var.get())
            if not 1 <= timeout_minutes <= 120:
                raise ValueError("Store check timeout must be between 1 and 120 minutes.")
        except (OSError, ValueError) as exc:
            messagebox.showerror(APP_NAME, str(exc), parent=self.root)
            return

        self._save_settings()
        self.probe_button.configure(state="disabled")
        self.status_var.set(f"Checking {host}:{port}; allowing up to {timeout_minutes} minutes for the store check…")
        threading.Thread(
            target=self._probe_worker_thread,
            args=(host, port, token, store, self.require_cuda_var.get(), timeout_minutes * 60),
            daemon=True,
        ).start()

    def _probe_worker_thread(self, host: str, port: int, token: str, store: Path, require_cuda: bool, response_timeout: float) -> None:
        try:
            capabilities = probe_worker(
                host,
                port,
                token,
                expected_store=store,
                require_cuda=require_cuda,
                response_timeout=response_timeout,
            )
            version_match = capabilities.get("gpustacker_version") == __version__
            devices = capabilities.get("cuda_devices") or []
            summary = "\n".join((
                f"Worker: {capabilities.get('hostname', host)}",
                f"GPUStacker version: {capabilities.get('gpustacker_version')} ({'match' if version_match else 'MISMATCH'})",
                f"Python / PyTorch: {capabilities.get('python_version')} / {capabilities.get('torch_version')}",
                f"CUDA devices: {', '.join(devices) if devices else 'none'}",
                f"Store: {capabilities.get('store_frame_count')} frames, shape {capabilities.get('store_shape')}",
                f"Shared read: {capabilities.get('store_read_mib_per_second')} MiB/s across {capabilities.get('store_read_sample_count')} sampled frames; write probe passed",
                "Coordinator and worker store check: matched",
                f"Stack-band task protocol: v{capabilities.get('stack_band_task_protocol')}; execution {'enabled' if capabilities.get('stack_task_execution_enabled') else 'disabled'}",
                "No stack tasks were sent.",
            ))
            self.events.put(("probe_result", (True, version_match, summary)))
        except (WorkerProbeError, OSError, ValueError) as exc:
            self.events.put(("probe_result", (False, False, str(exc))))

    def _set_result(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def _append_result(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.insert("end", text + "\n")
        if int(self.result_text.index("end-1c").split(".")[0]) > 2000:
            self.result_text.delete("1.0", "2.0")
        self.result_text.see("end")
        self.result_text.configure(state="disabled")

    def _poll(self) -> None:
        if self._closing:
            return
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "worker_stopped" and payload is self.server:
                    self.server = None
                    self.start_worker_button.configure(state="normal")
                    self.stop_worker_button.configure(state="disabled")
                    self.status_var.set("Worker service stopped.")
                elif event == "worker_status":
                    self.status_var.set(str(payload))
                    self._append_result(str(payload))
                elif event == "probe_result":
                    success, version_match, summary = payload
                    self.probe_button.configure(state="normal")
                    self._set_result(str(summary))
                    if success and version_match:
                        self.status_var.set("Worker checked successfully; capability discovery only.")
                    elif success:
                        self.status_var.set("Worker responded, but GPUStacker versions differ. Use matching versions.")
                    else:
                        self.status_var.set("Worker check failed.")
                        self._set_result(f"Check failed: {summary}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _close(self) -> None:
        self._closing = True
        self._save_settings()
        if self.server is not None:
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        self.root.destroy()