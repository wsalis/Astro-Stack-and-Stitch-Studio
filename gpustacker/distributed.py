"""Shared-store preflight and authenticated worker capability probes."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import re
import secrets
import select
import socket
import socketserver
import tempfile
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
import torch

from . import __version__
from .backend import select_backend
from .stacking import FrameStore, StackBandResult, StackResult, StackSettings, _stack_band_halo, _stack_band_rows, band_overhead, read_store_band, stack_store_band

WORKER_PROTOCOL_VERSION = 1
MAX_WORKER_MESSAGE_BYTES = 16 * 1024
WORKER_PROBE_SAMPLE_FRAMES = 32
WORKER_TASKS_IN_FLIGHT = 2
MAX_WORKER_PREFETCH_BYTES = 1024 * 1024 * 1024
WORKER_TASK_TIMEOUT_SECONDS = 300.0


class WorkerProbeError(RuntimeError):
    """A worker could not be authenticated or did not return a valid capability response."""


class _HashingWriter:
    def __init__(self, stream) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data) -> int:
        written = self.stream.write(data)
        self.digest.update(memoryview(data)[:written])
        return written

    def tell(self) -> int:
        return self.stream.tell()

    def read(self, size: int = -1) -> bytes:
        raise OSError("Checksum stream is write-only")

    def flush(self) -> None:
        self.stream.flush()


@dataclass(frozen=True)
class StoreReadBenchmark:
    frame_count: int
    sampled_frame_count: int
    shape: tuple[int, int, int]
    start_row: int
    rows: int
    bytes_read: int
    validation_seconds: float
    read_seconds: float
    checksum: float

    @property
    def mib_per_second(self) -> float:
        return self.bytes_read / (1024 * 1024) / max(self.read_seconds, 1e-9)


@dataclass(frozen=True)
class StackBandTask:
    """Wire contract for a future globally or locally normalized stack band."""

    task_id: str
    run_id: str
    pass_name: str
    row_start: int
    row_stop: int
    store_fingerprint: str
    settings_fingerprint: str
    stack_settings: dict[str, Any]
    reject_masks: bool
    vram_fraction: float


@dataclass(frozen=True)
class WorkerTaskAcknowledgement:
    task_id: str
    status: str
    reason_code: str
    message: str
    artifact_path: str | None = None
    artifact_sha256: str | None = None


def benchmark_store_read(
    directory: Path,
    rows: int = 8,
    start_row: int | None = None,
    store: FrameStore | None = None,
    validation_seconds: float | None = None,
    sample_indices: list[int] | None = None,
) -> StoreReadBenchmark:
    """Validate a store and read a row band from every or selected registered frames."""

    if rows < 1:
        raise ValueError("Benchmark row count must be positive")
    if store is None:
        validation_start = time.perf_counter()
        store = FrameStore.load_index(directory)
        validation_seconds = time.perf_counter() - validation_start
    elif validation_seconds is None:
        validation_seconds = 0.0
    channels, height, width = store.shape
    if rows > height:
        raise ValueError(f"Benchmark row count {rows} exceeds store height {height}")
    if start_row is None:
        start_row = (height - rows) // 2
    if start_row < 0 or start_row + rows > height:
        raise ValueError(f"Benchmark rows {start_row}:{start_row + rows} are outside store height {height}")
    if sample_indices is None:
        sample_paths = store.paths
    else:
        if any(index < 0 or index >= len(store) for index in sample_indices) or len(set(sample_indices)) != len(sample_indices):
            raise ValueError("Benchmark frame sample contains invalid indices")
        sample_paths = [store.paths[index] for index in sample_indices]
    if not sample_paths:
        raise ValueError("Benchmark frame sample is empty")

    buffer = np.empty((channels, rows, width), dtype=np.float32)
    checksum = 0.0
    read_start = time.perf_counter()
    for path in sample_paths:
        frame = np.load(path, mmap_mode="r", allow_pickle=False)
        if frame.shape != store.shape or frame.dtype != np.float32:
            raise ValueError(f"Registered frame has invalid shape or dtype: {path}")
        np.copyto(buffer, frame[:, start_row : start_row + rows, :])
        sample = buffer[0, rows // 2, width // 2]
        if np.isfinite(sample):
            checksum += float(sample)
        del frame
    read_seconds = time.perf_counter() - read_start
    bytes_read = len(sample_paths) * channels * rows * width * np.dtype(np.float32).itemsize
    return StoreReadBenchmark(
        frame_count=len(store),
        sampled_frame_count=len(sample_paths),
        shape=store.shape,
        start_row=start_row,
        rows=rows,
        bytes_read=bytes_read,
        validation_seconds=validation_seconds,
        read_seconds=read_seconds,
        checksum=checksum,
    )


def _canonical_json(document: dict[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _message_signature(token: str, message: bytes) -> str:
    if not token:
        raise WorkerProbeError("A shared worker token is required")
    key = token.encode("utf-8")
    if len(key) < 32:
        raise WorkerProbeError("The shared worker token must be at least 32 UTF-8 bytes")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _task_signature(token: str, document: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in document.items() if key != "signature"}
    return _message_signature(token, _canonical_json(unsigned))


def _validated_stack_band_task(document: Any) -> StackBandTask:
    required = {
        "task_id", "run_id", "pass_name", "row_start", "row_stop",
        "store_fingerprint", "settings_fingerprint", "stack_settings", "reject_masks", "vram_fraction",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError("Stack-band task fields are invalid")
    try:
        task = StackBandTask(**document)
    except TypeError as exc:
        raise ValueError(f"Stack-band task fields are invalid: {exc}") from exc
    for name, value in (("task_id", task.task_id), ("run_id", task.run_id)):
        try:
            if str(uuid.UUID(value)) != value:
                raise ValueError
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError(f"{name} must be a canonical UUID") from exc
    if task.pass_name not in ("global", "local"):
        raise ValueError("Stack-band pass_name must be 'global' or 'local'")
    if isinstance(task.row_start, bool) or isinstance(task.row_stop, bool) or not isinstance(task.row_start, int) or not isinstance(task.row_stop, int):
        raise ValueError("Stack-band row bounds must be integers")
    if task.row_start < 0 or task.row_stop <= task.row_start:
        raise ValueError("Stack-band row range must be non-empty and zero-based")
    for name, value in (("store_fingerprint", task.store_fingerprint), ("settings_fingerprint", task.settings_fingerprint)):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{name} must be a SHA-256 hex digest")
    if not isinstance(task.stack_settings, dict) or set(task.stack_settings) != set(asdict(StackSettings())):
        raise ValueError("Stack-band settings fields do not match this GPUStacker version")
    try:
        StackSettings(**task.stack_settings)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Stack-band settings are invalid: {exc}") from exc
    if hashlib.sha256(_canonical_json(task.stack_settings)).hexdigest() != task.settings_fingerprint:
        raise ValueError("Stack-band settings fingerprint does not match its settings")
    if not isinstance(task.reject_masks, bool):
        raise ValueError("Stack-band reject_masks must be boolean")
    if isinstance(task.vram_fraction, bool) or not isinstance(task.vram_fraction, (int, float)) or not 0.1 <= task.vram_fraction <= 1.0:
        raise ValueError("Stack-band vram_fraction must be between 0.1 and 1.0")
    return task


def _store_metadata_fingerprint(directory: Path) -> str:
    digest = hashlib.sha256()
    for name in ("registered_store.json", "index.json"):
        digest.update(name.encode("ascii"))
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def _probe_sample_indices(frame_count: int) -> list[int]:
    sample_count = min(frame_count, WORKER_PROBE_SAMPLE_FRAMES)
    if sample_count < 1:
        raise ValueError("Registered store has no frames")
    if sample_count == 1:
        return [0]
    return [(index * (frame_count - 1)) // (sample_count - 1) for index in range(sample_count)]


def _load_probe_store(directory: Path) -> FrameStore:
    """Validate store metadata and file presence without opening every frame array."""

    store_path = directory.resolve(strict=True)
    if (store_path / "cleanup_pending.json").exists() or (store_path / "cleanup_pending.json").is_symlink():
        raise ValueError("Registered-store cleanup is pending")
    manifest_path = store_path / "registered_store.json"
    index_path = store_path / "index.json"
    if manifest_path.is_symlink() or index_path.is_symlink():
        raise ValueError("Registered-store metadata cannot be symlinks")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != 1 or manifest.get("stage") != "registered_before_stack":
        raise ValueError("Unsupported or incomplete registered-store manifest")
    if not isinstance(index, dict):
        raise ValueError("Registered-store index is invalid")
    try:
        shape = tuple(int(value) for value in index["shape"])
        frames = index["frames"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Registered-store index is invalid: {exc}") from exc
    if len(shape) != 3 or any(value < 1 for value in shape) or not isinstance(frames, list) or not frames:
        raise ValueError("Registered-store index has an invalid shape or no frames")
    paths = []
    names = set()
    for entry in frames:
        stored_name = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(stored_name, str):
            raise ValueError("Registered-store frame entry has no path")
        stored_path = Path(stored_name)
        name = stored_path.name
        if not re.fullmatch(r"reg_\d{4}\.npy", name) or (not stored_path.is_absolute() and len(stored_path.parts) != 1):
            raise ValueError(f"Registered-store frame path is not an owned array: {stored_name}")
        if name in names:
            raise ValueError(f"Registered-store index repeats frame path: {name}")
        names.add(name)
        paths.append(store_path / name)
    if manifest.get("index") != "index.json" or manifest.get("shape") != list(shape) or manifest.get("frame_count") != len(paths):
        raise ValueError("Registered-store manifest does not match its index")
    with os.scandir(store_path) as entries:
        present = {entry.name for entry in entries if entry.is_file(follow_symlinks=False)}
    missing = [path.name for path in paths if path.name not in present]
    if missing:
        raise ValueError(f"Registered store is missing {len(missing)} indexed frame file(s), including {missing[0]}")
    store = FrameStore(store_path, shape)
    store.paths.extend(paths)
    return store


def _worker_memory_budget() -> int:
    return max(1, min(MAX_WORKER_PREFETCH_BYTES, int(psutil.virtual_memory().available) // 8))


def _worker_task_rows(store: FrameStore, budget: int) -> int:
    channels, _, width = store.shape
    return max(1, (budget // 2) // (len(store) * channels * width * np.dtype(np.float32).itemsize))


def _remote_task_can_finish_in_time(remaining_rows: int, local_seconds_per_row: float, remote_task_seconds: float) -> bool:
    return remote_task_seconds * 1.25 < remaining_rows * local_seconds_per_row


def worker_capabilities(directory: Path, rows: int = 8) -> dict[str, Any]:
    """Inspect the configured store and local worker capabilities without scheduling work."""

    store_path = directory.resolve(strict=True)
    validation_start = time.perf_counter()
    store = _load_probe_store(store_path)
    validation_seconds = time.perf_counter() - validation_start
    sample_indices = _probe_sample_indices(len(store))
    benchmark = benchmark_store_read(store_path, rows, store=store, validation_seconds=validation_seconds, sample_indices=sample_indices)
    probe_path = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=store_path, prefix=".gpustacker-worker-probe-", delete=False) as stream:
            probe_path = Path(stream.name)
            stream.write(b"gpustacker worker write probe\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)

    try:
        import torch

        cuda_devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        torch_version = torch.__version__
    except Exception as exc:
        cuda_devices = []
        torch_version = f"unavailable: {type(exc).__name__}: {exc}"
    return {
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "gpustacker_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch_version": torch_version,
        "cuda_devices": cuda_devices,
        "store_path": str(directory),
        "store_path_is_unc": str(directory).startswith("\\\\"),
        "stack_band_task_protocol": 1,
        "stack_task_execution_enabled": False,
        "stack_task_max_rows": _worker_task_rows(store, _worker_memory_budget()),
        "store_shape": list(benchmark.shape),
        "store_frame_count": benchmark.frame_count,
        "store_metadata_sha256": _store_metadata_fingerprint(store_path),
        "store_read_sample_indices": sample_indices,
        "store_read_sample_count": benchmark.sampled_frame_count,
        "store_read_rows": benchmark.rows,
        "store_read_start_row": benchmark.start_row,
        "store_read_mib_per_second": round(benchmark.mib_per_second, 3),
        "store_read_bytes": benchmark.bytes_read,
        "store_read_checksum": benchmark.checksum,
        "store_validation_seconds": round(benchmark.validation_seconds, 3),
        "store_read_seconds": round(benchmark.read_seconds, 3),
        "store_write_probe": "passed",
    }


def _store_sample_fingerprint(store: FrameStore) -> str:
    digest = hashlib.sha256()
    _, height, width = store.shape
    for index in _probe_sample_indices(len(store)):
        frame = np.load(store.paths[index], mmap_mode="r", allow_pickle=False)
        if frame.shape != store.shape or frame.dtype != np.float32:
            raise ValueError(f"Registered frame has invalid shape or dtype: {store.paths[index]}")
        digest.update(np.ascontiguousarray(frame[:, height // 2, width // 2]).tobytes())
        del frame
    return digest.hexdigest()


def worker_readiness(directory: Path) -> dict[str, Any]:
    """Validate metadata and sampled pixels without a frame-band benchmark."""

    store_path = directory.resolve(strict=True)
    store = _load_probe_store(store_path)
    return {
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "gpustacker_version": __version__,
        "stack_task_max_rows": _worker_task_rows(store, _worker_memory_budget()),
        "store_shape": list(store.shape),
        "store_frame_count": len(store),
        "store_metadata_sha256": _store_metadata_fingerprint(store_path),
        "store_sample_sha256": _store_sample_fingerprint(store),
    }


def _execute_stack_band_task(
    directory: Path,
    task: StackBandTask,
    status_callback: Callable[[str], None] | None,
    store_cache: dict[str, Any],
    read_lock: threading.Lock,
    gpu_lock: threading.Lock,
    cancel_callback: Callable[[], None] | None = None,
) -> WorkerTaskAcknowledgement:
    task_started = time.perf_counter()
    if status_callback is not None:
        status_callback(f"Received {task.pass_name} stack task rows {task.row_start}:{task.row_stop}; waiting for store access.")
    store_path = directory.resolve(strict=True)
    with read_lock:
        if cancel_callback is not None:
            cancel_callback()
        current_fingerprint = _store_metadata_fingerprint(store_path)
        if current_fingerprint != task.store_fingerprint:
            raise WorkerProbeError("Registered-store fingerprint changed before task execution")
        if store_cache.get("fingerprint") != current_fingerprint:
            if status_callback is not None:
                status_callback("Loading registered-store index and normalization maps; shared-storage access may take time.")
            previous_store = store_cache.get("store")
            store_cache["store"] = FrameStore.load_index(
                store_path,
                header_workers=8,
                header_cache=None if previous_store is None else previous_store._validated_headers,
            )
            if status_callback is not None:
                status_callback(f"Preparing band readers for {len(store_cache['store'])} registered frames.")
            store_cache["maps"] = store_cache["store"].band_reader()
            store_cache["fingerprint"] = current_fingerprint
        store = store_cache["store"]
        frame_maps = store_cache["maps"]
        if task.row_stop > store.shape[1]:
            raise WorkerProbeError("Stack-band range exceeds registered-store height")
        settings = StackSettings(**task.stack_settings)
        halo = _stack_band_halo(settings)
        read_start = max(0, task.row_start - halo)
        read_stop = min(store.shape[1], task.row_stop + halo)
        memory_budget = _worker_memory_budget()
        input_bytes = len(store) * store.shape[0] * (read_stop - read_start) * store.shape[2] * 4
        output_bytes = (task.row_stop - task.row_start) * store.shape[2] * (store.shape[0] * 4 + 6 + (len(store) if task.reject_masks else 0))
        if output_bytes > memory_budget:
            raise WorkerProbeError("Task result exceeds the worker RAM budget; reduce rows per task")
        prefetch_budget_bytes = memory_budget
        setup_seconds = time.perf_counter() - task_started
        read_started = time.perf_counter()
        host_data = None
        if input_bytes <= memory_budget:
            if status_callback is not None:
                status_callback(f"Reading {task.pass_name} stack band rows {task.row_start}:{task.row_stop}.")
            host_data = read_store_band(frame_maps, store.shape, read_start, read_stop)
        read_seconds = time.perf_counter() - read_started
        input_mode = "prefetched" if host_data is not None else "streamed"
    gpu_requested = time.perf_counter()
    with gpu_lock:
        gpu_wait_seconds = time.perf_counter() - gpu_requested
        compute_started = time.perf_counter()
        if cancel_callback is not None:
            cancel_callback()
        if store_cache.get("vram_fraction") != task.vram_fraction:
            store_cache["backend"] = select_backend(vram_fraction=float(task.vram_fraction))
            store_cache["vram_fraction"] = task.vram_fraction
        backend = store_cache["backend"]
        memory_budget = min(memory_budget, _worker_memory_budget())
        bytes_per_row = len(store) * store.shape[0] * store.shape[2] * 4
        if bytes_per_row > memory_budget:
            raise WorkerProbeError("One stack row exceeds the worker RAM budget")
        backend = replace(backend, budget_bytes=min(backend.budget_bytes, int(memory_budget * band_overhead(settings))))
        if status_callback is not None:
            status_callback(f"Executing {task.pass_name} stack band rows {task.row_start}:{task.row_stop} on {backend.name}.")
        result = _stack_worker_band_with_memory_retry(
            store,
            backend,
            settings,
            task.row_start,
            task.row_stop,
            task.reject_masks,
            host_data,
            status_callback,
            frame_maps,
            cancel_callback,
        )
        compute_seconds = time.perf_counter() - compute_started
    del host_data
    if cancel_callback is not None:
        cancel_callback()
    publish_started = time.perf_counter()
    artifact_dir = store_path / "gpustacker_tasks" / task.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"{task.task_id}.npz"
    if artifact.is_symlink():
        raise WorkerProbeError("Task artifact path cannot be a symlink")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=artifact_dir, prefix=f".{task.task_id}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = _HashingWriter(stream)
            np.savez(
                writer,
                image=result.image,
                coverage=result.coverage,
                rejection=result.rejection,
                rejection_masks=result.rejection_masks if result.rejection_masks is not None else np.empty((0,), dtype=np.uint8),
                rejected_low=np.asarray(result.rejected_low, dtype=np.int64),
                rejected_high=np.asarray(result.rejected_high, dtype=np.int64),
                total_samples=np.asarray(result.total_samples, dtype=np.int64),
            )
            stream.flush()
            os.fsync(stream.fileno())
            digest = writer.digest.hexdigest()
        if cancel_callback is not None:
            cancel_callback()
        os.replace(temporary, artifact)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    relative_path = artifact.relative_to(store_path).as_posix()
    message = (
        f"Worker timing {task.pass_name} rows {task.row_start}:{task.row_stop}: "
        f"setup={setup_seconds:.3f}s read={read_seconds:.3f}s gpu_wait={gpu_wait_seconds:.3f}s "
        f"compute={compute_seconds:.3f}s publish={time.perf_counter() - publish_started:.3f}s "
        f"total={time.perf_counter() - task_started:.3f}s mode={input_mode} "
        f"input={input_bytes / 2**20:.1f}MiB budget={prefetch_budget_bytes / 2**20:.1f}MiB"
    )
    if status_callback is not None:
        status_callback(message)
    return WorkerTaskAcknowledgement(task.task_id, "accepted", "complete", message, relative_path, digest)


def _stack_worker_band_with_memory_retry(
    store: FrameStore,
    backend,
    settings: StackSettings,
    row_start: int,
    row_stop: int,
    reject_masks: bool,
    host_data: np.ndarray | None,
    status_callback: Callable[[str], None] | None,
    frame_maps=None,
    cancel_callback: Callable[[], None] | None = None,
) -> StackBandResult:
    try:
        return stack_store_band(
            store, backend, settings, row_start, row_stop, reject_masks,
            frame_maps=frame_maps, host_data=host_data, cancel_cb=cancel_callback,
        )
    except torch.cuda.OutOfMemoryError:
        if settings.large_scale or row_stop - row_start <= 1:
            raise
    # Retrying outside the handler releases the traceback that pins the failed attempt's GPU tensors.
    backend.empty_cache()
    middle = row_start + (row_stop - row_start) // 2
    split = middle - row_start
    if status_callback is not None:
        status_callback(f"CUDA memory limit reached; retrying rows {row_start}:{row_stop} as smaller bands.")
    upper = _stack_worker_band_with_memory_retry(
        store, backend, settings, row_start, middle, reject_masks,
        None if host_data is None else host_data[:, :, :split], status_callback, frame_maps, cancel_callback,
    )
    lower = _stack_worker_band_with_memory_retry(
        store, backend, settings, middle, row_stop, reject_masks,
        None if host_data is None else host_data[:, :, split:], status_callback, frame_maps, cancel_callback,
    )
    masks = None
    if upper.rejection_masks is not None and lower.rejection_masks is not None:
        masks = np.concatenate((upper.rejection_masks, lower.rejection_masks), axis=1)
    return StackBandResult(
        image=np.concatenate((upper.image, lower.image), axis=1),
        rejected_low=upper.rejected_low + lower.rejected_low,
        rejected_high=upper.rejected_high + lower.rejected_high,
        total_samples=upper.total_samples + lower.total_samples,
        coverage=np.concatenate((upper.coverage, lower.coverage), axis=0),
        rejection=np.concatenate((upper.rejection, lower.rejection), axis=0),
        rejection_masks=masks,
    )


def create_worker_server(
    host: str,
    port: int,
    directory: Path,
    token: str,
    rows: int = 8,
    status_callback: Callable[[str], None] | None = None,
    enable_task_execution: bool = False,
) -> socketserver.ThreadingTCPServer:
    """Create an authenticated worker; stack execution must be explicitly enabled."""

    _message_signature(token, b"validate-token")

    class WorkerProbeServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    class Handler(socketserver.StreamRequestHandler):
        def check_connection(self) -> None:
            readable, _, _ = select.select([self.request], [], [], 0)
            if readable and not self.request.recv(1, socket.MSG_PEEK):
                raise ConnectionAbortedError("Coordinator disconnected before task completion")

        def handle(self) -> None:
            self.request.settimeout(10.0)
            raw_request = self.rfile.readline(MAX_WORKER_MESSAGE_BYTES + 1)
            self.request.settimeout(None)
            operation = "probe"
            try:
                if len(raw_request) > MAX_WORKER_MESSAGE_BYTES or not raw_request.endswith(b"\n"):
                    raise WorkerProbeError("invalid request size")
                request = json.loads(raw_request)
                if not isinstance(request, dict) or request.get("protocol_version") != WORKER_PROTOCOL_VERSION:
                    raise WorkerProbeError("unsupported protocol")
                operation = request.get("operation", "probe")
                if operation == "stack_band_task":
                    nonce = request.get("nonce")
                    signature = request.get("signature")
                    if not isinstance(nonce, str) or len(nonce) != 32 or not isinstance(signature, str):
                        raise WorkerProbeError("invalid task authentication envelope")
                    expected = _task_signature(token, request)
                    if not hmac.compare_digest(signature, expected):
                        raise WorkerProbeError("unauthorized")
                    task = _validated_stack_band_task(request.get("task"))
                    if not enable_task_execution:
                        acknowledgement = WorkerTaskAcknowledgement(task.task_id, "rejected", "execution_disabled", "Stack-band execution is disabled on this worker pending opt-in and output-equivalence validation.")
                    else:
                        acknowledgement = _execute_stack_band_task(directory, task, status_callback, store_cache, read_lock, gpu_lock, self.check_connection)
                    response = {"nonce": nonce, **asdict(acknowledgement)}
                    response["signature"] = _message_signature(token, nonce.encode("ascii") + _canonical_json(response))
                    if status_callback is not None:
                        status_callback(f"Stack-band task {task.task_id}: {acknowledgement.status} ({acknowledgement.reason_code}).")
                    try:
                        self.wfile.write(_canonical_json(response) + b"\n")
                    except OSError as exc:
                        if status_callback is not None:
                            status_callback(f"Coordinator disconnected before task acknowledgement: {type(exc).__name__}: {exc}")
                    return
                if operation not in ("probe", "ready"):
                    raise WorkerProbeError("unsupported operation")
                nonce = request.get("nonce")
                signature = request.get("signature")
                if not isinstance(nonce, str) or len(nonce) != 32 or not isinstance(signature, str):
                    raise WorkerProbeError("invalid authentication envelope")
                expected = _message_signature(token, nonce.encode("ascii", errors="strict"))
                if not hmac.compare_digest(signature, expected):
                    raise WorkerProbeError("unauthorized")
                if status_callback is not None:
                    status_callback(f"Probe received from {self.client_address[0]}; checking registered store…")
                capabilities = worker_readiness(directory) if operation == "ready" else worker_capabilities(directory, rows)
                capabilities["stack_task_execution_enabled"] = enable_task_execution
                response = {"nonce": nonce, "capabilities": capabilities}
                response["signature"] = _message_signature(token, nonce.encode("ascii") + _canonical_json(response))
                self.wfile.write(_canonical_json(response) + b"\n")
                if status_callback is not None:
                    status_callback("Capability probe complete; no stack tasks were received.")
            except Exception as exc:
                if isinstance(exc, WorkerProbeError):
                    error = "unauthorized"
                else:
                    error = "task failed" if operation == "stack_band_task" else "probe failed"
                    if status_callback is not None:
                        action = "Stack-band task" if operation == "stack_band_task" else "Capability probe"
                        status_callback(f"{action} failed: {type(exc).__name__}: {exc}")
                self.wfile.write(_canonical_json({"error": error}) + b"\n")

    read_lock = threading.Lock()
    gpu_lock = threading.Lock()
    store_cache: dict[str, Any] = {}
    return WorkerProbeServer((host, port), Handler)


def probe_worker(
    host: str,
    port: int,
    token: str,
    timeout: float = 5.0,
    expected_store: Path | None = None,
    require_cuda: bool = False,
    response_timeout: float | None = None,
    cancel_callback: Callable[[], None] | None = None,
    readiness_only: bool = False,
) -> dict[str, Any]:
    """Authenticate to a worker and verify its signed capability response."""

    nonce = secrets.token_hex(16)
    request = {
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "operation": "ready" if readiness_only else "probe",
        "nonce": nonce,
        "signature": _message_signature(token, nonce.encode("ascii")),
    }
    try:
        connection = socket.create_connection((host, port), timeout=timeout)
    except OSError as exc:
        raise WorkerProbeError(f"Could not connect to worker {host}:{port}: {exc}") from exc
    response_wait = timeout if response_timeout is None else response_timeout
    try:
        with connection:
            connection.settimeout(response_wait)
            connection.sendall(_canonical_json(request) + b"\n")
            raw_response = _read_worker_response(connection, response_wait, cancel_callback)
        if len(raw_response) > MAX_WORKER_MESSAGE_BYTES or not raw_response.endswith(b"\n"):
            raise WorkerProbeError("Worker returned an invalid response size")
        response = json.loads(raw_response)
    except socket.timeout as exc:
        raise WorkerProbeError(f"Worker {host}:{port} accepted the connection but did not finish its store check within {response_wait:g} seconds") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerProbeError(f"Could not read the worker response from {host}:{port}: {exc}") from exc
    if not isinstance(response, dict):
        raise WorkerProbeError("Worker returned an invalid response")
    if response.get("error"):
        raise WorkerProbeError(f"Worker probe failed: {response['error']}")
    capabilities = response.get("capabilities")
    signature = response.get("signature")
    if response.get("nonce") != nonce or not isinstance(capabilities, dict) or not isinstance(signature, str):
        raise WorkerProbeError("Worker response is incomplete or stale")
    expected = _message_signature(token, nonce.encode("ascii") + _canonical_json({"nonce": nonce, "capabilities": capabilities}))
    if not hmac.compare_digest(signature, expected):
        raise WorkerProbeError("Worker response authentication failed")
    if capabilities.get("protocol_version") != WORKER_PROTOCOL_VERSION:
        raise WorkerProbeError("Worker protocol version is incompatible")
    if require_cuda and not capabilities.get("cuda_devices"):
        raise WorkerProbeError("Worker has no available CUDA device")
    if expected_store is not None:
        try:
            store = _load_probe_store(expected_store)
            local_fingerprint = _store_metadata_fingerprint(expected_store)
            if readiness_only:
                if (
                    capabilities.get("store_metadata_sha256") != local_fingerprint
                    or capabilities.get("store_sample_sha256") != _store_sample_fingerprint(store)
                ):
                    raise WorkerProbeError("Worker and coordinator do not see the same registered store")
            else:
                sample_indices = _probe_sample_indices(len(store))
                if capabilities.get("store_read_sample_indices") != sample_indices:
                    raise WorkerProbeError("Worker and coordinator selected different shared-store frame samples")
                local_read = benchmark_store_read(
                    expected_store,
                    int(capabilities["store_read_rows"]),
                    int(capabilities["store_read_start_row"]),
                    store=store,
                    sample_indices=sample_indices,
                )
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise WorkerProbeError(f"Could not validate coordinator shared store: {exc}") from exc
        matching_store = readiness_only or (
            local_fingerprint == capabilities.get("store_metadata_sha256")
            and list(local_read.shape) == capabilities.get("store_shape")
            and local_read.frame_count == capabilities.get("store_frame_count")
            and local_read.sampled_frame_count == capabilities.get("store_read_sample_count")
            and local_read.bytes_read == capabilities.get("store_read_bytes")
            and local_read.checksum == capabilities.get("store_read_checksum")
        )
        if not matching_store:
            raise WorkerProbeError("Worker and coordinator do not see the same registered store")
        capabilities["coordinator_store_match"] = True
    return capabilities


def _read_worker_response(connection: socket.socket, timeout: float, cancel_callback: Callable[[], None] | None) -> bytes:
    deadline = time.monotonic() + timeout
    response = bytearray()
    while len(response) <= MAX_WORKER_MESSAGE_BYTES and not response.endswith(b"\n"):
        if cancel_callback is not None:
            cancel_callback()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise socket.timeout("Worker response deadline exceeded")
        connection.settimeout(min(0.25, remaining))
        try:
            chunk = connection.recv(min(4096, MAX_WORKER_MESSAGE_BYTES + 1 - len(response)))
        except socket.timeout:
            continue
        if not chunk:
            break
        response.extend(chunk)
    return bytes(response)


def send_stack_band_task(
    host: str,
    port: int,
    token: str,
    task: StackBandTask,
    timeout: float = WORKER_TASK_TIMEOUT_SECONDS,
    *,
    cancel_callback: Callable[[], None] | None = None,
) -> WorkerTaskAcknowledgement:
    """Send a signed stack-band envelope and verify the worker's signed acknowledgement."""

    validated_task = _validated_stack_band_task(asdict(task))
    nonce = secrets.token_hex(16)
    request: dict[str, Any] = {
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "operation": "stack_band_task",
        "nonce": nonce,
        "task": asdict(validated_task),
    }
    request["signature"] = _task_signature(token, request)
    try:
        with socket.create_connection((host, port), timeout=min(timeout, 5.0)) as connection:
            connection.settimeout(timeout)
            connection.sendall(_canonical_json(request) + b"\n")
            raw_response = _read_worker_response(connection, timeout, cancel_callback)
        if len(raw_response) > MAX_WORKER_MESSAGE_BYTES or not raw_response.endswith(b"\n"):
            raise WorkerProbeError("Worker returned an invalid task acknowledgement size")
        response = json.loads(raw_response)
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerProbeError(f"Could not send task to worker {host}:{port}: {exc}") from exc
    if not isinstance(response, dict):
        raise WorkerProbeError("Worker returned an invalid task acknowledgement")
    if response.get("error"):
        raise WorkerProbeError(f"Worker rejected task request: {response['error']}")
    signature = response.get("signature")
    if response.get("nonce") != nonce or response.get("task_id") != validated_task.task_id or not isinstance(signature, str):
        raise WorkerProbeError("Worker task acknowledgement does not match the request")
    unsigned_response = {key: value for key, value in response.items() if key != "signature"}
    expected = _message_signature(token, nonce.encode("ascii") + _canonical_json(unsigned_response))
    if not hmac.compare_digest(signature, expected):
        raise WorkerProbeError("Worker task acknowledgement authentication failed")
    status = response.get("status")
    reason_code = response.get("reason_code")
    message = response.get("message")
    if status not in ("accepted", "rejected") or not isinstance(reason_code, str) or not isinstance(message, str):
        raise WorkerProbeError("Worker task acknowledgement has invalid fields")
    artifact_path = response.get("artifact_path")
    artifact_sha256 = response.get("artifact_sha256")
    if status == "accepted":
        if not isinstance(artifact_path, str) or not isinstance(artifact_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None:
            raise WorkerProbeError("Worker accepted a task without a valid result artifact")
        pure_path = Path(artifact_path)
        if pure_path.is_absolute() or ".." in pure_path.parts or pure_path.suffix != ".npz":
            raise WorkerProbeError("Worker returned an unsafe result artifact path")
    elif artifact_path is not None or artifact_sha256 is not None:
        raise WorkerProbeError("Rejected task acknowledgement unexpectedly contains an artifact")
    return WorkerTaskAcknowledgement(validated_task.task_id, status, reason_code, message, artifact_path, artifact_sha256)


def stack_store_distributed(
    store: FrameStore,
    backend,
    settings: StackSettings,
    host: str,
    port: int,
    token: str,
    *,
    pass_name: str = "global",
    reject_masks: bool = False,
    vram_fraction: float = 0.75,
    rows_per_task: int = 64,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_callback: Callable[[], None] | None = None,
    task_timeout: float = WORKER_TASK_TIMEOUT_SECONDS,
) -> StackResult:
    """Dynamically assign row bands to this computer and an authenticated worker."""

    if not 1 <= rows_per_task <= 4096:
        raise ValueError("Distributed task band height must be between 1 and 4096 rows")
    if not 0 < task_timeout < float("inf"):
        raise ValueError("Worker task timeout must be finite and positive")
    channels, height, width = store.shape
    settings_document = asdict(settings)
    settings_fingerprint = hashlib.sha256(_canonical_json(settings_document)).hexdigest()
    store_fingerprint = _store_metadata_fingerprint(store.directory)
    run_id = str(uuid.uuid4())
    tasks = [
        StackBandTask(
            task_id=str(uuid.uuid4()),
            run_id=run_id,
            pass_name=pass_name,
            row_start=row_start,
            row_stop=min(height, row_start + rows_per_task),
            store_fingerprint=store_fingerprint,
            settings_fingerprint=settings_fingerprint,
            stack_settings=settings_document,
            reject_masks=reject_masks,
            vram_fraction=vram_fraction,
        )
        for row_start in range(0, height, rows_per_task)
    ]
    completed_tasks: set[str] = set()
    pending = deque(tasks)
    retry_locally: deque[StackBandTask] = deque()
    completed_rows = 0
    lock = threading.Lock()
    stopped = threading.Event()
    frame_maps = store.band_reader()
    remote_enabled = True
    image = np.empty((channels, height, width), dtype=np.float32)
    coverage = np.empty((height, width), dtype=np.uint16)
    rejection = np.empty((height, width), dtype=np.float32)
    mask_maps: list[np.memmap] = []
    if reject_masks:
        store.mask_paths = [path.with_name(path.name.replace("reg_", "rej_")) for path in store.paths]
        mask_maps = [np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(height, width)) for path in store.mask_paths]
    rejected_low = rejected_high = total_samples = 0
    local_seconds_per_row: float | None = None
    remote_task_seconds: float | None = None

    def next_task(local: bool) -> StackBandTask | None:
        with lock:
            if local and retry_locally:
                return retry_locally.popleft()
            if not local and local_seconds_per_row is not None and remote_task_seconds is not None:
                remaining_rows = sum(task.row_stop - task.row_start for task in pending)
                if not _remote_task_can_finish_in_time(remaining_rows, local_seconds_per_row, remote_task_seconds):
                    return None
            return pending.popleft() if pending else None

    def record(task: StackBandTask, result: StackBandResult, low: int, high: int, source: str, batch: list[StackBandTask] | None = None) -> None:
        nonlocal completed_rows, rejected_low, rejected_high, total_samples
        with lock:
            identifiers = {item.task_id for item in batch} if batch else {task.task_id}
            if completed_tasks.intersection(identifiers):
                raise WorkerProbeError(f"Duplicate completion for stack task {task.task_id}")
            row_start, row_stop = task.row_start, task.row_stop
            image[:, row_start:row_stop] = result.image
            coverage[row_start:row_stop] = result.coverage
            rejection[row_start:row_stop] = result.rejection
            if reject_masks and result.rejection_masks is not None:
                for index, mask_map in enumerate(mask_maps):
                    mask_map[row_start:row_stop] = result.rejection_masks[index]
            completed_tasks.update(identifiers)
            rejected_low += low
            rejected_high += high
            total_samples += result.total_samples
            completed_rows += task.row_stop - task.row_start
            done = completed_rows
        if progress_callback is not None:
            progress_callback(done / height, f"Distributed stack: {done}/{height} rows complete ({source})")

    def check_cancel() -> None:
        if cancel_callback is not None:
            cancel_callback()
        if stopped.is_set():
            raise RuntimeError("Distributed stacking stopped")

    def compute_local(task: StackBandTask, host_data: np.ndarray | None = None, batch: list[StackBandTask] | None = None) -> None:
        nonlocal local_seconds_per_row
        check_cancel()
        started = time.perf_counter()
        result = stack_store_band(store, backend, settings, task.row_start, task.row_stop, reject_masks, frame_maps, host_data=host_data, cancel_cb=check_cancel)
        record(task, result, result.rejected_low, result.rejected_high, "this computer", batch)
        seconds_per_row = (time.perf_counter() - started) / (task.row_stop - task.row_start)
        with lock:
            local_seconds_per_row = seconds_per_row if local_seconds_per_row is None else 0.75 * local_seconds_per_row + 0.25 * seconds_per_row

    def compute_remote(task: StackBandTask) -> bool:
        nonlocal remote_enabled, remote_task_seconds
        started = time.perf_counter()
        try:
            check_cancel()
            if progress_callback is not None:
                progress_callback(completed_rows / height, f"Sending {task.pass_name} stack band rows {task.row_start}:{task.row_stop} to worker {host}:{port}")
            acknowledgement = send_stack_band_task(host, port, token, task, timeout=task_timeout, cancel_callback=check_cancel)
            check_cancel()
            if acknowledgement.status != "accepted" or acknowledgement.artifact_path is None or acknowledgement.artifact_sha256 is None:
                remote_enabled = False
                if progress_callback is not None:
                    progress_callback(completed_rows / height, f"Worker unavailable for this run ({acknowledgement.reason_code}); finishing locally")
                return False
            if progress_callback is not None:
                progress_callback(completed_rows / height, acknowledgement.message)
            artifact_candidate = store.directory / acknowledgement.artifact_path
            if artifact_candidate.is_symlink():
                raise WorkerProbeError("Worker result artifact cannot be a symlink")
            artifact = artifact_candidate.resolve(strict=True)
            root = store.directory.resolve(strict=True)
            if not artifact.is_relative_to(root) or not artifact.is_file():
                raise WorkerProbeError("Worker result artifact escapes the registered store")
            digest_builder = hashlib.sha256()
            with artifact.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    check_cancel()
                    digest_builder.update(chunk)
            digest = digest_builder.hexdigest()
            if not hmac.compare_digest(digest, acknowledgement.artifact_sha256):
                raise WorkerProbeError("Worker result artifact checksum does not match its acknowledgement")
            with np.load(artifact, allow_pickle=False) as data:
                expected_rows = task.row_stop - task.row_start
                image = np.asarray(data["image"], dtype=np.float32)
                coverage = np.asarray(data["coverage"], dtype=np.uint16)
                rejection = np.asarray(data["rejection"], dtype=np.float32)
                masks_array = np.asarray(data["rejection_masks"], dtype=np.uint8)
                masks = masks_array if reject_masks else None
                if image.shape != (channels, expected_rows, width) or coverage.shape != (expected_rows, width) or rejection.shape != (expected_rows, width):
                    raise WorkerProbeError("Worker result artifact has invalid band dimensions")
                if not np.isfinite(image).all() or not np.isfinite(rejection).all():
                    raise WorkerProbeError("Worker result artifact contains non-finite values")
                if reject_masks and (masks is None or masks.shape != (len(store), expected_rows, width)):
                    raise WorkerProbeError("Worker result rejection masks have invalid dimensions")
                low = int(data["rejected_low"])
                high = int(data["rejected_high"])
                total = int(data["total_samples"])
                if low < 0 or high < 0 or total != len(store) * channels * expected_rows * width:
                    raise WorkerProbeError("Worker result counters are invalid")
            result = StackBandResult(image, low, high, total, coverage, rejection, masks)
            record(task, result, low, high, "worker")
            try:
                artifact.unlink(missing_ok=True)
            except OSError:
                pass
            elapsed = time.perf_counter() - started
            with lock:
                remote_task_seconds = elapsed if remote_task_seconds is None else 0.75 * remote_task_seconds + 0.25 * elapsed
            return True
        except (OSError, ValueError, KeyError, WorkerProbeError) as exc:
            remote_enabled = False
            if progress_callback is not None:
                progress_callback(completed_rows / height, f"Worker task failed verification; finishing its band locally ({exc})")
            return False

    def remote_loop(first_task: StackBandTask | None) -> None:
        task = first_task
        while task is not None:
            if not remote_enabled or not compute_remote(task):
                with lock:
                    retry_locally.append(task)
                return
            task = next_task(local=False)
            if task is None and progress_callback is not None:
                with lock:
                    remaining = bool(pending)
                if remaining:
                    progress_callback(completed_rows / height, "Holding worker dispatch near pass end: measured remote turnaround exceeds remaining local work")

    def local_loop() -> None:
        halo = _stack_band_halo(settings)
        bytes_per_row = len(store) * channels * width * 4
        read_budget = _worker_memory_budget()
        local_rows = max(rows_per_task, min(_stack_band_rows(store, backend, settings), max(1, read_budget // bytes_per_row - 2 * halo)))
        if progress_callback is not None:
            progress_callback(completed_rows / height, f"Distributed scheduling: remote up to {rows_per_task} rows/task; coordinator up to {local_rows} rows/batch, one read ahead")

        def reserve_batch() -> list[StackBandTask]:
            with lock:
                queue = retry_locally if retry_locally else pending
                if not queue:
                    return []
                batch = [queue.popleft()]
                while queue and queue[0].row_start == batch[-1].row_stop and queue[0].row_stop - batch[0].row_start <= local_rows:
                    batch.append(queue.popleft())
                return batch

        def prefetch(batch: list[StackBandTask]) -> tuple[np.ndarray | None, float]:
            check_cancel()
            started = time.perf_counter()
            start = max(0, batch[0].row_start - halo)
            stop = min(height, batch[-1].row_stop + halo)
            if bytes_per_row * (stop - start) > read_budget:
                return None, 0.0
            data = read_store_band(frame_maps, store.shape, start, stop)
            return data, time.perf_counter() - started

        batch = reserve_batch()
        if not batch:
            return
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpustacker-local-prefetch") as reader:
            future = reader.submit(prefetch, batch)
            while batch:
                host_data, read_seconds = future.result()
                following = reserve_batch()
                if following:
                    future = reader.submit(prefetch, following)
                task = replace(batch[0], row_stop=batch[-1].row_stop)
                compute_started = time.perf_counter()
                compute_local(task, host_data, batch)
                if progress_callback is not None:
                    progress_callback(completed_rows / height, f"Coordinator timing {pass_name} rows {task.row_start}:{task.row_stop}: read={read_seconds:.3f}s compute_and_merge={time.perf_counter() - compute_started:.3f}s")
                del host_data
                batch = following

    def run_loop(callback, *args) -> None:
        try:
            callback(*args)
        except Exception:
            stopped.set()
            raise

    initial_remote_tasks = [next_task(local=False) for _ in range(WORKER_TASKS_IN_FLIGHT)]
    with ThreadPoolExecutor(max_workers=WORKER_TASKS_IN_FLIGHT + 1, thread_name_prefix="gpustacker-distributed") as executor:
        futures = [executor.submit(run_loop, remote_loop, task) for task in initial_remote_tasks if task is not None]
        futures.append(executor.submit(run_loop, local_loop))
        for future in as_completed(futures):
            future.result()
    # A worker failure can requeue a band after the local loop has already drained the queue.
    while retry_locally:
        compute_local(retry_locally.popleft())
    if len(completed_tasks) != len(tasks):
        raise WorkerProbeError("Distributed stacking did not complete every row band")
    for mask_map in mask_maps:
        mask_map.flush()
    backend.empty_cache()
    return StackResult(image, rejected_low, rejected_high, total_samples, len(tasks), coverage, rejection)