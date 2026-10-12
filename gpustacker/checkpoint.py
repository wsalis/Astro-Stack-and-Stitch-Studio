"""Validated, atomic manifests for resumable pipeline checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CHECKPOINT_SCHEMA = 1


class CheckpointError(ValueError):
    """A checkpoint is missing, incomplete, stale, or incompatible with this run."""


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    """Write JSON through a same-directory temporary file and atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f"{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, sort_keys=True, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_npy_atomic(path: Path, array: np.ndarray) -> None:
    """Write a NumPy artifact durably before atomically replacing its destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f"{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, np.asarray(array), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fingerprint_inputs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Capture inexpensive file identities for validating a saved checkpoint."""

    fingerprints = []
    for path in paths:
        try:
            resolved = path.resolve(strict=True)
            stat = resolved.stat()
        except OSError as exc:
            raise CheckpointError(f"Checkpoint input is unavailable: {path}: {exc}") from exc
        if not resolved.is_file():
            raise CheckpointError(f"Checkpoint input is not a file: {resolved}")
        fingerprints.append({"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return fingerprints


def settings_digest(settings: dict[str, Any]) -> str:
    """Hash canonical JSON settings so resume cannot silently mix run configurations."""

    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_manifest(path: Path, stage: str, settings: dict[str, Any], inputs: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    """Atomically replace a checkpoint manifest after its payload artifacts are durable."""

    document = {
        "schema": CHECKPOINT_SCHEMA,
        "stage": stage,
        "settings_sha256": settings_digest(settings),
        "inputs": inputs,
        "payload": payload,
    }
    write_json_atomic(path, document)


def load_manifest(path: Path, settings: dict[str, Any], input_paths: Iterable[Path]) -> dict[str, Any]:
    """Load and validate a manifest against the current files and run settings."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Cannot read checkpoint manifest {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise CheckpointError("Checkpoint manifest is incomplete")
    if document.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError(f"Unsupported checkpoint schema: {document.get('schema')!r}")
    if document.get("settings_sha256") != settings_digest(settings):
        raise CheckpointError("Checkpoint settings do not match this run")
    expected_inputs = fingerprint_inputs(input_paths)
    if document.get("inputs") != expected_inputs:
        raise CheckpointError("Checkpoint inputs have changed since it was created")
    if not isinstance(document.get("stage"), str) or not isinstance(document.get("payload"), dict):
        raise CheckpointError("Checkpoint manifest is incomplete")
    return document