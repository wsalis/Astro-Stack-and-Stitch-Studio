from __future__ import annotations

import json
import hashlib
import os
import queue
import shutil
import socketserver
import threading
import time
import weakref
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpustacker.distributed import (
    WORKER_PROBE_SAMPLE_FRAMES,
    StackBandTask,
    WorkerProbeError,
    _load_probe_store,
    _probe_sample_indices,
    _stack_worker_band_with_memory_retry,
    benchmark_store_read,
    create_worker_server,
    probe_worker,
    send_stack_band_task,
    stack_store_distributed,
)
from gpustacker.backend import select_backend
from gpustacker.cli import build_parser
from gpustacker import __version__
import gpustacker.distributed_gui as distributed_gui
import gpustacker.distributed as distributed_module
from gpustacker.distributed_gui import _protect_token, _unprotect_token
from gpustacker.stacking import FrameStore, StackBandResult, StackSettings, read_store_band, stack_store, stack_store_band


def test_worker_probe_reports_store_and_backend_capabilities(tmp_path):
    directory = tmp_path / "registered"
    store = FrameStore(directory, (1, 4, 5))
    store.add("frame-0.fit", np.ones((1, 4, 5), dtype=np.float32))
    store.add("frame-1.fit", np.full((1, 4, 5), 2.0, dtype=np.float32))
    store.save_index()
    (directory / "registered_store.json").write_text(json.dumps({
        "schema": 1,
        "stage": "registered_before_stack",
        "shape": list(store.shape),
        "frame_count": len(store),
        "index": "index.json",
    }), encoding="utf-8")
    token = "test-worker-token-with-sufficient-entropy"
    worker_status = []
    server = create_worker_server("127.0.0.1", 0, directory, token, rows=2, status_callback=worker_status.append)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address[:2]
        capabilities = probe_worker(host, port, token, expected_store=directory)

        assert capabilities["protocol_version"] == 1
        assert capabilities["stack_task_execution_enabled"] is False
        assert capabilities["gpustacker_version"]
        assert capabilities["hostname"]
        assert capabilities["store_shape"] == [1, 4, 5]
        assert capabilities["store_frame_count"] == 2
        assert capabilities["store_read_bytes"] == 2 * 1 * 2 * 5 * 4
        assert capabilities["store_write_probe"] == "passed"
        assert capabilities["coordinator_store_match"] is True
        assert worker_status == [
            "Probe received from 127.0.0.1; checking registered store…",
            "Capability probe complete; no stack tasks were received.",
        ]
        assert list(directory.glob(".gpustacker-worker-probe-*")) == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_worker_probe_rejects_different_shared_store(tmp_path):
    directory = tmp_path / "registered"
    store = FrameStore(directory, (1, 4, 5))
    store.add("frame.fit", np.ones((1, 4, 5), dtype=np.float32))
    store.save_index()
    (directory / "registered_store.json").write_text(json.dumps({
        "schema": 1,
        "stage": "registered_before_stack",
        "shape": list(store.shape),
        "frame_count": len(store),
        "index": "index.json",
    }), encoding="utf-8")
    different_directory = tmp_path / "different"
    shutil.copytree(directory, different_directory)
    different_frame = np.load(different_directory / "reg_0000.npy")
    different_frame[0, 1, 2] = 3.0
    np.save(different_directory / "reg_0000.npy", different_frame)

    server = create_worker_server("127.0.0.1", 0, directory, "correct-worker-token-with-at-least-32-bytes", rows=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address[:2]
        with pytest.raises(WorkerProbeError, match="do not see the same registered store"):
            probe_worker(host, port, "correct-worker-token-with-at-least-32-bytes", expected_store=different_directory)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_worker_probe_rejects_wrong_token(tmp_path):
    directory = tmp_path / "registered"
    store = FrameStore(directory, (1, 4, 5))
    store.add("frame.fit", np.ones((1, 4, 5), dtype=np.float32))
    store.save_index()
    (directory / "registered_store.json").write_text(json.dumps({
        "schema": 1,
        "stage": "registered_before_stack",
        "shape": list(store.shape),
        "frame_count": len(store),
        "index": "index.json",
    }), encoding="utf-8")
    server = create_worker_server("127.0.0.1", 0, directory, "correct-worker-token-with-at-least-32-bytes", rows=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address[:2]
        with pytest.raises(WorkerProbeError, match="unauthorized"):
            probe_worker(host, port, "incorrect-worker-token-with-at-least-32-bytes", expected_store=directory)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_worker_cli_requires_coordinator_store_path(tmp_path):
    args = build_parser().parse_args([
        "worker-probe",
        "worker-host",
        "--shared-store",
        str(tmp_path),
        "--require-cuda",
    ])

    assert args.command == "worker-probe"
    assert args.shared_store == tmp_path
    assert args.require_cuda is True


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI is required")
def test_remembered_worker_token_is_encrypted_for_current_user():
    token = "remember-this-worker-token-only-in-encrypted-form"

    protected = _protect_token(token)

    assert token not in protected
    assert _unprotect_token(protected) == token


def test_gui_passes_configured_response_timeout(monkeypatch, tmp_path):
    window = distributed_gui.DistributedWindow.__new__(distributed_gui.DistributedWindow)
    window.events = queue.Queue()
    received = {}

    def fake_probe(host, port, token, **kwargs):
        received.update(kwargs)
        return {"gpustacker_version": __version__, "cuda_devices": []}

    monkeypatch.setattr(distributed_gui, "probe_worker", fake_probe)

    window._probe_worker_thread("worker", 47861, "token", tmp_path, False, 1800)

    assert received["response_timeout"] == 1800
    assert window.events.get_nowait()[0] == "probe_result"


def test_probe_reads_bounded_evenly_spaced_frame_sample(tmp_path, monkeypatch):
    directory = tmp_path / "registered"
    store = FrameStore(directory, (1, 4, 5))
    for index in range(WORKER_PROBE_SAMPLE_FRAMES + 8):
        store.add(f"frame-{index}.fit", np.full((1, 4, 5), index, dtype=np.float32))
    store.save_index()
    (directory / "registered_store.json").write_text(json.dumps({
        "schema": 1,
        "stage": "registered_before_stack",
        "shape": list(store.shape),
        "frame_count": len(store),
        "index": "index.json",
    }), encoding="utf-8")
    real_load = np.load
    loaded_paths = []

    def tracking_load(path, *args, **kwargs):
        loaded_paths.append(Path(path).name)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", tracking_load)
    probe_store = _load_probe_store(directory)
    sample_indices = _probe_sample_indices(len(probe_store))
    benchmark = benchmark_store_read(directory, rows=1, store=probe_store, sample_indices=sample_indices)

    assert len(probe_store) == WORKER_PROBE_SAMPLE_FRAMES + 8
    assert len(sample_indices) == WORKER_PROBE_SAMPLE_FRAMES
    assert sample_indices[0] == 0
    assert sample_indices[-1] == len(probe_store) - 1
    assert benchmark.sampled_frame_count == WORKER_PROBE_SAMPLE_FRAMES
    assert len(loaded_paths) == WORKER_PROBE_SAMPLE_FRAMES

    (directory / "reg_0004.npy").unlink()
    with pytest.raises(ValueError, match="missing 1 indexed frame"):
        _load_probe_store(directory)


def test_stack_band_task_is_signed_and_explicitly_rejected_until_execution_is_enabled(tmp_path):
    directory = tmp_path / "registered"
    store = FrameStore(directory, (1, 4, 5))
    store.add("frame.fit", np.ones((1, 4, 5), dtype=np.float32))
    store.save_index()
    (directory / "registered_store.json").write_text(json.dumps({
        "schema": 1,
        "stage": "registered_before_stack",
        "shape": list(store.shape),
        "frame_count": len(store),
        "index": "index.json",
    }), encoding="utf-8")
    token = "task-envelope-token-with-at-least-32-bytes"
    worker_status = []
    server = create_worker_server("127.0.0.1", 0, directory, token, status_callback=worker_status.append)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stack_settings = asdict(StackSettings())
    settings_fingerprint = hashlib.sha256(json.dumps(stack_settings, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    task = StackBandTask(
        task_id="0b4c64e8-19f0-4d96-8f77-5e79f61552a4",
        run_id="7c2879d0-7c4a-431a-9c50-d6adce8c50ac",
        pass_name="global",
        row_start=0,
        row_stop=2,
        store_fingerprint="a" * 64,
        settings_fingerprint=settings_fingerprint,
        stack_settings=stack_settings,
        reject_masks=False,
        vram_fraction=0.75,
    )

    try:
        host, port = server.server_address[:2]
        acknowledgement = send_stack_band_task(host, port, token, task)

        assert acknowledgement.task_id == task.task_id
        assert acknowledgement.status == "rejected"
        assert acknowledgement.reason_code == "execution_disabled"
        assert "output-equivalence" in acknowledgement.message
        assert worker_status == [f"Stack-band task {task.task_id}: rejected (execution_disabled)."]
        with pytest.raises(WorkerProbeError, match="unauthorized"):
            send_stack_band_task(host, port, "incorrect-task-token-with-at-least-32-bytes", task)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("mode", ["normal", "stream", "spatial"])
def test_enabled_worker_computes_bands_and_matches_local_stack(tmp_path, monkeypatch, mode):
    if mode == "stream":
        monkeypatch.setattr(distributed_module, "_worker_memory_budget", lambda: 512)
        monkeypatch.setattr(distributed_module, "read_store_band", lambda *args: pytest.fail("Oversized input was prefetched"))
    directory = tmp_path / "registered"
    store = FrameStore(directory, (1, 9, 11))
    rng = np.random.default_rng(91)
    for index in range(5):
        frame = rng.normal(index, 0.2, store.shape).astype(np.float32)
        if index == 4:
            frame[0, 4, 5] += 20
        store.add(f"frame-{index}.fit", frame)
    store.save_index()
    (directory / "registered_store.json").write_text(json.dumps({
        "schema": 1,
        "stage": "registered_before_stack",
        "shape": list(store.shape),
        "frame_count": len(store),
        "index": "index.json",
    }), encoding="utf-8")
    token = "enabled-worker-task-token-with-at-least-32-bytes"
    worker_status = []
    load_index = FrameStore.load_index

    def load_reported_index(cls, *args, **kwargs):
        assert any(message.startswith("Received global stack task") for message in worker_status)
        assert any(message.startswith("Loading registered-store index") for message in worker_status)
        return load_index(*args, **kwargs)

    monkeypatch.setattr(FrameStore, "load_index", classmethod(load_reported_index))
    server = create_worker_server(
        "127.0.0.1", 0, directory, token,
        status_callback=worker_status.append,
        enable_task_execution=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address[:2]
        capabilities = probe_worker(host, port, token, expected_store=directory)
        assert capabilities["stack_task_execution_enabled"] is True

        settings = StackSettings(
            method="sigma", sigma_low=1.5, sigma_high=1.5, normalize=False, use_weights=False,
            large_scale=mode == "spatial", large_scale_box=3, large_scale_grow=1,
        )
        expected = stack_store(store, select_backend("cpu"), settings, reject_masks=True)
        expected_masks = [np.load(path, allow_pickle=False).copy() for path in store.mask_paths]
        actual = stack_store_distributed(
            store,
            select_backend("cpu"),
            settings,
            host,
            port,
            token,
            reject_masks=True,
            rows_per_task=3,
        )

        assert np.allclose(actual.image, expected.image, atol=2e-5, rtol=2e-5)
        assert np.array_equal(actual.coverage, expected.coverage)
        assert np.allclose(actual.rejection, expected.rejection, atol=2e-5, rtol=2e-5)
        assert actual.total_samples == expected.total_samples
        assert actual.bands == 3
        assert all(np.array_equal(np.load(path, allow_pickle=False), expected_mask) for path, expected_mask in zip(store.mask_paths, expected_masks))
        assert list((directory / "gpustacker_tasks").rglob("*.npz")) == []
        assert any("accepted (complete)" in message for message in worker_status)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_worker_activity_replaces_stale_probe_result():
    window = distributed_gui.DistributedWindow.__new__(distributed_gui.DistributedWindow)
    window._closing = False
    window.events = queue.Queue()
    statuses = []
    results = []
    window.status_var = SimpleNamespace(set=statuses.append)
    window._set_result = results.append
    window.root = SimpleNamespace(after=lambda *args: None)
    message = "Received global stack task rows 0:32; waiting for store access."
    window.events.put(("worker_status", message))

    window._poll()

    assert statuses == [message]
    assert results == [message]


def test_worker_retries_cuda_oom_as_smaller_bands(monkeypatch, tmp_path):
    import torch
    import gpustacker.distributed as distributed_module

    store = FrameStore(tmp_path / "registered", (1, 8, 3))
    store.add("frame.fit", np.ones(store.shape, dtype=np.float32))
    backend = select_backend("cpu")
    calls = []
    statuses = []
    live_failed_attempts = []

    class FailedAttemptMemory:
        pass

    def stack_band(_store, _backend, _settings, row_start, row_stop, reject_masks, host_data=None, frame_maps=None, cancel_cb=None):
        assert host_data is not None and host_data.shape[2] == row_stop - row_start
        assert not [ref for ref in live_failed_attempts if ref() is not None], "failed attempt memory is still referenced"
        calls.append((row_start, row_stop))
        rows = row_stop - row_start
        if rows > 2:
            pinned = FailedAttemptMemory()
            live_failed_attempts.append(weakref.ref(pinned))
            raise torch.cuda.OutOfMemoryError("simulated CUDA allocation failure")
        masks = np.full((len(store), rows, store.shape[2]), row_start, dtype=np.uint8) if reject_masks else None
        return StackBandResult(
            np.full((1, rows, store.shape[2]), row_start, dtype=np.float32),
            rows,
            rows * 2,
            rows * store.shape[2],
            np.ones((rows, store.shape[2]), dtype=np.uint16),
            np.zeros((rows, store.shape[2]), dtype=np.float32),
            masks,
        )

    monkeypatch.setattr(distributed_module, "stack_store_band", stack_band)

    result = _stack_worker_band_with_memory_retry(
        store, backend, StackSettings(), 0, 8, True, np.zeros((1, 1, 8, 3), dtype=np.float32), statuses.append,
    )

    assert calls == [(0, 8), (0, 4), (0, 2), (2, 4), (4, 8), (4, 6), (6, 8)]
    assert result.image.shape == (1, 8, 3)
    assert result.rejection_masks is not None and result.rejection_masks.shape == (1, 8, 3)
    assert result.rejected_low == 8
    assert result.rejected_high == 16
    assert result.total_samples == 24
    assert len(statuses) == 3


def test_loaded_store_reuses_validated_frame_headers(tmp_path, monkeypatch):
    store = FrameStore(tmp_path / "registered", (2, 10, 7))
    expected = []
    for index in range(4):
        frame = np.full(store.shape, index, dtype=np.float32)
        expected.append(frame[:, 3:7])
        store.add(str(index), frame)
    store.save_index()
    restored = FrameStore.load_index(store.directory)

    def unexpected_load(*args, **kwargs):
        pytest.fail("Band reader reopened an already validated frame header")

    monkeypatch.setattr(np, "load", unexpected_load)
    reader = restored.band_reader()
    np.testing.assert_array_equal(reader.read(3, 7), np.stack(expected))
    np.testing.assert_array_equal(restored.band_reader().read(3, 7), np.stack(expected))


def test_pre_read_band_matches_memmap_band(tmp_path):
    store = FrameStore(tmp_path / "registered", (2, 10, 7))
    rng = np.random.default_rng(5)
    for index in range(6):
        store.add(f"frame-{index}.fit", rng.normal(index, 0.3, store.shape).astype(np.float32))
    backend = select_backend("cpu")
    settings = StackSettings(method="sigma", sigma_low=2.0, sigma_high=2.0)
    maps = store.memmaps()

    expected = stack_store_band(store, backend, settings, 3, 7, reject_masks=True, frame_maps=maps)
    actual = stack_store_band(store, backend, settings, 3, 7, reject_masks=True, host_data=read_store_band(maps, store.shape, 3, 7))

    np.testing.assert_array_equal(actual.image, expected.image)
    np.testing.assert_array_equal(actual.coverage, expected.coverage)
    np.testing.assert_array_equal(actual.rejection_masks, expected.rejection_masks)
    assert actual.total_samples == expected.total_samples


@pytest.mark.parametrize("pre_read", [False, True])
@pytest.mark.parametrize("capacity_rows", [12, 4096])
def test_large_scale_rejection_matches_across_task_boundaries(tmp_path, pre_read, capacity_rows):
    store = FrameStore(tmp_path / "registered", (1, 32, 32))
    rng = np.random.default_rng(12)
    for index in range(10):
        frame = rng.normal(100, 1, store.shape).astype(np.float32)
        if index == 0:
            frame[0, 9:20, 8:24] += 100
        store.add(str(index), frame)
    settings = StackSettings(
        method="sigma", sigma_high=2.0, sigma_low=3.0,
        normalize=False, use_weights=False, large_scale=True,
        large_scale_box=5, large_scale_density=0.2, large_scale_grow=2,
    )
    backend = select_backend("cpu")
    expected = stack_store_band(store, backend, settings, 0, 32, True)
    backend = replace(backend, budget_bytes=capacity_rows * len(store) * 32 * 4 * 8)
    maps = store.band_reader()
    parts = []
    for row in range(0, 32, 8):
        host_data = read_store_band(maps, store.shape, max(0, row - 4), min(32, row + 12)) if pre_read else None
        parts.append(stack_store_band(store, backend, settings, row, row + 8, True, maps, host_data))

    np.testing.assert_array_equal(np.concatenate([part.image for part in parts], axis=1), expected.image)
    np.testing.assert_array_equal(np.concatenate([part.rejection_masks for part in parts], axis=1), expected.rejection_masks)
    assert sum(part.rejected_low for part in parts) == expected.rejected_low
    assert sum(part.rejected_high for part in parts) == expected.rejected_high
    assert sum(part.total_samples for part in parts) == expected.total_samples
    actual = stack_store(store, backend, settings, reject_masks=True)
    np.testing.assert_array_equal(actual.image, expected.image)
    np.testing.assert_array_equal(actual.rejection, expected.rejection)
    np.testing.assert_array_equal(actual.coverage, expected.coverage)
    np.testing.assert_array_equal(np.stack([np.load(path) for path in store.mask_paths]), expected.rejection_masks)
    assert actual.rejected_low == expected.rejected_low
    assert actual.rejected_high == expected.rejected_high


def test_worker_memory_budget_and_advertised_rows_are_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(distributed_module.psutil, "virtual_memory", lambda: SimpleNamespace(available=4 * 1024**3))
    budget = distributed_module._worker_memory_budget()
    assert budget == 512 * 1024**2
    store = FrameStore(tmp_path, (3, 2160, 3840))
    store.paths = [tmp_path / f"reg_{index:04d}.npy" for index in range(7000)]
    rows = distributed_module._worker_task_rows(store, budget)
    assert rows * len(store) * 3 * 3840 * 4 <= budget
    assert rows < 64


def test_large_scale_rejection_fails_clearly_without_overlap_memory(tmp_path):
    store = FrameStore(tmp_path / "registered", (1, 32, 32))
    store.add("frame.fit", np.ones(store.shape, dtype=np.float32))
    backend = replace(select_backend("cpu"), budget_bytes=8 * 32 * 4 * 8)
    settings = StackSettings(method="sigma", large_scale=True, large_scale_box=5, large_scale_grow=2)
    with pytest.raises(ValueError, match="Insufficient device memory.*overlap"):
        stack_store_band(store, backend, settings, 0, 32)


@pytest.fixture
def stalled_worker():
    received = threading.Event()
    release = threading.Event()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.rfile.readline()
            received.set()
            release.wait(timeout=5)

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield (*server.server_address[:2], received)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("cancel", [False, True])
def test_stalled_worker_times_out_or_cancels(tmp_path, stalled_worker, cancel):
    store = FrameStore(tmp_path / "registered", (1, 9, 11))
    for index in range(4):
        store.add(str(index), np.full(store.shape, index, dtype=np.float32))
    store.save_index()
    (store.directory / "registered_store.json").write_text("{}", encoding="utf-8")
    host, port, received = stalled_worker
    settings = StackSettings(method="none", normalize=False, use_weights=False)
    backend = select_backend("cpu")
    expected = stack_store(store, backend, settings)

    def check_cancel():
        if cancel and received.is_set():
            raise RuntimeError("Cancelled")

    start = time.monotonic()
    kwargs = dict(rows_per_task=2, task_timeout=2.0 if cancel else 0.2, cancel_callback=check_cancel)
    if cancel:
        with pytest.raises(RuntimeError, match="Cancelled"):
            stack_store_distributed(store, backend, settings, host, port, "test-stalled-worker-token-at-least-32-bytes", **kwargs)
    else:
        actual = stack_store_distributed(store, backend, settings, host, port, "test-stalled-worker-token-at-least-32-bytes", **kwargs)
        np.testing.assert_array_equal(actual.image, expected.image)
        np.testing.assert_array_equal(actual.coverage, expected.coverage)
        assert actual.total_samples == expected.total_samples
    assert time.monotonic() - start < 5


def test_local_failure_interrupts_stalled_remote_waits(tmp_path, monkeypatch, stalled_worker):
    store = FrameStore(tmp_path / "registered", (1, 9, 11))
    store.add("frame.fit", np.ones(store.shape, dtype=np.float32))
    store.save_index()
    (store.directory / "registered_store.json").write_text("{}", encoding="utf-8")
    host, port, received = stalled_worker

    def fail_local(*args, **kwargs):
        assert received.wait(timeout=2)
        raise RuntimeError("Local stack failed")

    monkeypatch.setattr(distributed_module, "stack_store_band", fail_local)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="Local stack failed"):
        stack_store_distributed(
            store, select_backend("cpu"), StackSettings(), host, port,
            "test-local-failure-token-at-least-32-bytes", rows_per_task=2,
        )
    assert time.monotonic() - start < 5


def test_rejected_worker_bands_are_stacked_by_the_coordinator(tmp_path):
    directory = tmp_path / "registered"
    store = FrameStore(directory, (1, 9, 11))
    rng = np.random.default_rng(17)
    for index in range(4):
        store.add(f"frame-{index}.fit", rng.normal(index, 0.2, store.shape).astype(np.float32))
    store.save_index()
    (directory / "registered_store.json").write_text(json.dumps({
        "schema": 1,
        "stage": "registered_before_stack",
        "shape": list(store.shape),
        "frame_count": len(store),
        "index": "index.json",
    }), encoding="utf-8")
    token = "disabled-worker-task-token-with-at-least-32-bytes"
    server = create_worker_server("127.0.0.1", 0, directory, token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address[:2]
        settings = StackSettings(method="none", normalize=False, use_weights=False)
        expected = stack_store(store, select_backend("cpu"), settings)
        actual = stack_store_distributed(store, select_backend("cpu"), settings, host, port, token, rows_per_task=2)

        assert np.allclose(actual.image, expected.image, atol=2e-5, rtol=2e-5)
        assert np.array_equal(actual.coverage, expected.coverage)
        assert actual.bands == 5
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)