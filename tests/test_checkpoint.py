from __future__ import annotations

import json

import pytest

import numpy as np

from gpustacker.checkpoint import CheckpointError, load_manifest, write_manifest, write_npy_atomic


def test_manifest_round_trips_and_matches_inputs_and_settings(tmp_path):
    source = tmp_path / "light.fit"
    source.write_bytes(b"light data")
    path = tmp_path / "work" / "checkpoint.json"
    settings = {"weighting": "psfsw", "reference": "light.fit"}

    write_manifest(path, "registered", settings, [{"path": str(source.resolve()), "size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}], {"index": "index.json"})

    document = load_manifest(path, settings, [source])
    assert document["stage"] == "registered"
    assert document["payload"] == {"index": "index.json"}
    with pytest.raises(CheckpointError, match="settings"):
        load_manifest(path, {"weighting": "noise"}, [source])


def test_manifest_rejects_changed_or_missing_inputs(tmp_path):
    source = tmp_path / "light.fit"
    source.write_bytes(b"light data")
    path = tmp_path / "checkpoint.json"
    settings = {"stack": "winsorized"}
    inputs = [{"path": str(source.resolve()), "size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}]
    write_manifest(path, "registered", settings, inputs, {})

    source.write_bytes(b"changed light data")
    with pytest.raises(CheckpointError, match="inputs have changed"):
        load_manifest(path, settings, [source])
    with pytest.raises(CheckpointError, match="input is unavailable"):
        load_manifest(path, settings, [tmp_path / "missing.fit"])


def test_manifest_rejects_partial_json(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text('{"schema":', encoding="utf-8")

    with pytest.raises(CheckpointError, match="Cannot read checkpoint manifest"):
        load_manifest(path, {}, [])


def test_manifest_json_is_parseable_after_atomic_write(tmp_path):
    path = tmp_path / "checkpoint.json"
    write_manifest(path, "registered", {}, [], {"ok": True})

    assert json.loads(path.read_text(encoding="utf-8"))["payload"] == {"ok": True}


def test_numpy_checkpoint_write_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "global_stack.npy"
    write_npy_atomic(path, np.ones((2, 3), dtype=np.float32))
    original = path.read_bytes()

    def fail_save(*_args, **_kwargs):
        raise OSError("simulated array failure")

    monkeypatch.setattr(np, "save", fail_save)
    with pytest.raises(OSError, match="simulated array failure"):
        write_npy_atomic(path, np.zeros((2, 3), dtype=np.float32))

    assert path.read_bytes() == original
    assert list(tmp_path.glob("global_stack.npy.*.tmp")) == []