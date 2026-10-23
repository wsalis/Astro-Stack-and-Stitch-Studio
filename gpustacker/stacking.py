"""Chunked (row-band) GPU stacking with outlier rejection.

Registered frames live on disk as float32 ``.npy`` memmaps shaped (C, H, W) with NaN
marking uncovered pixels. Bands of rows are streamed to the device so the stack never
has to fit in VRAM all at once.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .backend import Backend, rows_per_band
from .checkpoint import write_json_atomic

RejectionMethod = Literal["none", "median", "sigma", "winsorized", "percentile", "gesd"]
ProgressCallback = Callable[[float, str], None]


def _noop_progress(_: float, __: str) -> None:
    pass


@dataclass
class FrameNorm:
    offset: float | np.ndarray = 0.0  # background level of this frame; scalar or per-channel (C,)
    scale: float | np.ndarray = 1.0  # multiplicative factor to bring noise/level onto the reference; scalar or (C,)
    weight: float = 1.0
    # optional local normalisation: low-res (C, bh, bw) maps, frame' = scale_map * frame + offset_map
    scale_map: np.ndarray | None = None
    offset_map: np.ndarray | None = None
    weight_map: np.ndarray | None = None  # low-resolution spatial quality weights on the reference grid
    # (median, MAD-sigma, valid fraction) per local-normalisation block, (C, bh, bw) each, taken at registration
    block_stats: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def norm_vec(value: float | np.ndarray, channels: int) -> np.ndarray:
    """Broadcast a scalar or (C,) normalisation parameter to a float32 (C,) array."""

    return np.broadcast_to(np.asarray(value, dtype=np.float32).reshape(-1), (channels,)).copy()


def _storage_volume_id(path: Path) -> int:
    return path.stat().st_dev


def _allocated_file_bytes(size_bytes: int) -> int:
    return ((size_bytes + 256 + 4095) // 4096) * 4096


@dataclass
class FrameStore:
    """Registered frames on disk plus their normalisation parameters."""

    directory: Path
    shape: tuple[int, int, int]  # (C, H, W)
    paths: list[Path] = field(default_factory=list)
    norms: list[FrameNorm] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    mask_paths: list[Path] = field(default_factory=list)  # per-frame rejection masks (optional)
    storage_roots: list[Path] = field(default_factory=list)
    reserve_mask_bytes: int = 0
    storage_dirs: list[Path] = field(default_factory=list, init=False)
    _remaining_bytes: list[int] = field(default_factory=list, init=False, repr=False)
    _data_offsets: dict[Path, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        roots = [Path(root).expanduser().resolve() for root in self.storage_roots]
        if roots:
            seen_volumes = set()
            selected_roots = []
            for root in roots:
                root.mkdir(parents=True, exist_ok=True)
                volume = _storage_volume_id(root)
                if volume not in seen_volumes:
                    seen_volumes.add(volume)
                    selected_roots.append(root)
            store_id = uuid.uuid4().hex
            self.storage_dirs = [root / f".gpustacker_registered_{store_id}" for root in selected_roots]
            for directory in self.storage_dirs:
                directory.mkdir()
        else:
            self.storage_dirs = [self.directory]
        self._remaining_bytes = [int(shutil.disk_usage(directory).free) for directory in self.storage_dirs]

    def __len__(self) -> int:
        return len(self.paths)

    def add(self, name: str, data: np.ndarray, norm: FrameNorm | None = None) -> Path:
        arr = np.ascontiguousarray(data, dtype=np.float32)
        path = self._reserve(name, norm, arr.nbytes)
        np.save(path, arr)
        return path

    def add_async(self, name: str, data: np.ndarray, norm: FrameNorm | None, executor) -> "Future[Path]":
        """Reserve the slot now (keeps frame order) and write on the executor."""

        arr = np.ascontiguousarray(data, dtype=np.float32)
        path = self._reserve(name, norm, arr.nbytes)

        def write() -> Path:
            try:
                np.save(path, arr)
            except OSError as exc:
                raise OSError(f"Failed writing registered frame {name!r} to {path} ({arr.nbytes} bytes): {exc}") from exc
            return path

        return executor.submit(write)

    def _reserve(self, name: str, norm: FrameNorm | None, size_bytes: int) -> Path:
        reserved_bytes = _allocated_file_bytes(size_bytes)
        if self.reserve_mask_bytes:
            reserved_bytes += _allocated_file_bytes(self.reserve_mask_bytes)
        available = [
            min(remaining, int(shutil.disk_usage(directory).free))
            for directory, remaining in zip(self.storage_dirs, self._remaining_bytes)
        ]
        candidates = [index for index, free in enumerate(available) if free >= reserved_bytes]
        if not candidates:
            details = ", ".join(f"{directory}: {free} bytes free" for directory, free in zip(self.storage_dirs, available))
            raise OSError(f"No selected registered-frame drive has space for a {size_bytes}-byte frame ({details})")
        storage_index = max(candidates, key=lambda index: available[index])
        self._remaining_bytes[storage_index] = available[storage_index] - reserved_bytes
        path = self.storage_dirs[storage_index] / f"reg_{len(self.paths):04d}.npy"
        self.paths.append(path)
        self.norms.append(norm or FrameNorm())
        self.names.append(name)
        return path

    def memmaps(self) -> list[np.memmap]:
        return [np.load(p, mmap_mode="r") for p in self.paths]

    def band_reader(self, workers: int = 8) -> "BandReader":
        offsets = [self._data_offsets[path] for path in self.paths] if all(path in self._data_offsets for path in self.paths) else None
        reader = BandReader(self.paths, self.shape, workers, data_offsets=offsets)
        self._data_offsets.update(zip(self.paths, reader.data_offsets))
        return reader

    @classmethod
    def load_index(
        cls,
        directory: Path,
        allow_missing: bool = False,
        allowed_storage_dirs: list[Path] | None = None,
    ) -> "FrameStore":
        """Restore a registered store, validating that every frame belongs to it and matches its shape."""

        try:
            index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
            shape = tuple(int(v) for v in index["shape"])
            frames = index["frames"]
            stored_dirs = index.get("storage_dirs", [])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot read registered-frame index in {directory}: {exc}") from exc
        if len(shape) != 3 or any(v < 1 for v in shape) or not isinstance(frames, list) or not frames:
            raise ValueError("Registered-frame index has an invalid shape or no frames")
        root = directory.resolve()
        if not isinstance(stored_dirs, list) or any(not isinstance(path, str) for path in stored_dirs):
            raise ValueError("Registered-frame index has invalid storage directories")
        stored_paths = [Path(path) for path in stored_dirs]
        if any(path.is_symlink() for path in stored_paths):
            raise ValueError("Registered-frame index contains a symlinked storage directory")
        external_dirs = [path.resolve(strict=not allow_missing) for path in stored_paths]
        if any(path == root or re.fullmatch(r"\.gpustacker_registered_[0-9a-f]{32}", path.name) is None for path in external_dirs):
            raise ValueError("Registered-frame index contains an invalid storage directory")
        if allowed_storage_dirs is not None:
            allowed = {Path(path).resolve(strict=not allow_missing) for path in allowed_storage_dirs}
            if set(external_dirs) != allowed:
                raise ValueError("Registered-frame storage directories do not match the store manifest")
        store = cls(directory, shape)
        store.storage_dirs = [root, *external_dirs]
        for frame_index, entry in enumerate(frames):
            try:
                stored_path = Path(entry["path"])
                is_registered_array = stored_path.name.startswith("reg_") and stored_path.name.endswith(".npy") and stored_path.name[4:-4].isdigit()
                if stored_path.is_absolute():
                    try:
                        path = stored_path.resolve(strict=not allow_missing)
                    except FileNotFoundError:
                        if not is_registered_array:
                            raise
                        path = (directory / stored_path.name).resolve(strict=not allow_missing)
                else:
                    path = (directory / stored_path).resolve(strict=not allow_missing)
                if not path.is_relative_to(root) and not any(path.parent == storage_dir for storage_dir in external_dirs):
                    if not stored_path.is_absolute() or not is_registered_array or external_dirs:
                        raise ValueError(f"Registered frame is outside its store: {path}")
                    path = (directory / stored_path.name).resolve(strict=not allow_missing)
                    if not path.is_relative_to(root):
                        raise ValueError(f"Registered frame is outside its store: {path}")
                if path.exists():
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    valid_shape = array.shape == shape and array.dtype == np.float32
                    data_offset = int(array.offset)
                    del array
                    if not valid_shape:
                        raise ValueError(f"Registered frame has invalid shape or dtype: {path}")
                    store._data_offsets[path] = data_offset
                offset_value = entry["offset"]
                scale_value = entry["scale"]
                offset = float(offset_value) if np.isscalar(offset_value) else np.asarray(offset_value, dtype=np.float32)
                scale = float(scale_value) if np.isscalar(scale_value) else np.asarray(scale_value, dtype=np.float32)
                weight_map_value = entry.get("weight_map")
                weight_map = None
                if weight_map_value is not None:
                    weight_path = (directory / str(weight_map_value)).resolve(strict=not allow_missing)
                    if not weight_path.is_relative_to(root):
                        raise ValueError(f"Weight map is outside its store: {weight_path}")
                    if weight_path.exists():
                        weight_map = np.load(weight_path, allow_pickle=False)
                        if weight_map.ndim != 2 or weight_map.dtype != np.float32 or not np.isfinite(weight_map).all():
                            raise ValueError(f"Registered weight map has invalid shape or dtype: {weight_path}")
                scale_map_value = entry.get("scale_map")
                offset_map_value = entry.get("offset_map")
                if (scale_map_value is None) != (offset_map_value is None):
                    raise ValueError("Registered local-normalisation maps are incomplete")
                scale_map = offset_map = None
                if scale_map_value is not None:
                    scale_map_path = (directory / str(scale_map_value)).resolve(strict=not allow_missing)
                    offset_map_path = (directory / str(offset_map_value)).resolve(strict=not allow_missing)
                    if not scale_map_path.is_relative_to(root) or not offset_map_path.is_relative_to(root):
                        raise ValueError("Registered local-normalisation map is outside its store")
                    if scale_map_path.exists() and offset_map_path.exists():
                        scale_map = np.load(scale_map_path, allow_pickle=False)
                        offset_map = np.load(offset_map_path, allow_pickle=False)
                        valid_maps = (
                            scale_map.ndim == 3
                            and offset_map.shape == scale_map.shape
                            and scale_map.shape[0] == shape[0]
                            and scale_map.dtype == np.float32
                            and offset_map.dtype == np.float32
                            and np.isfinite(scale_map).all()
                            and np.isfinite(offset_map).all()
                        )
                        if not valid_maps:
                            raise ValueError("Registered local-normalisation maps have invalid shape or dtype")
                    elif not allow_missing:
                        raise ValueError("Registered local-normalisation maps are missing")
                store.paths.append(path)
                store.names.append(str(entry["name"]))
                store.norms.append(FrameNorm(offset=offset, scale=scale, weight=float(entry["weight"]), scale_map=scale_map, offset_map=offset_map, weight_map=weight_map))
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid registered-frame entry in {directory}: {exc}") from exc
        return store

    def drop(self, index: int) -> None:
        """Remove a registered frame (e.g. failed a post-registration filter)."""

        path = self.paths.pop(index)
        self.norms.pop(index)
        self.names.pop(index)
        try:
            path.unlink()
        except OSError:
            pass

    def save_index(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        frames = []
        for i, (name, path, norm) in enumerate(zip(self.names, self.paths, self.norms)):
            weight_map_path = None
            if norm.weight_map is not None:
                weight_map_path = self.directory / f"weight_{i:04d}.npy"
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile("wb", dir=self.directory, prefix=f"{weight_map_path.name}.", suffix=".tmp", delete=False) as stream:
                        temporary = Path(stream.name)
                        np.save(stream, np.ascontiguousarray(norm.weight_map, dtype=np.float32), allow_pickle=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, weight_map_path)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            scale_map_path = offset_map_path = None
            if norm.scale_map is not None or norm.offset_map is not None:
                if norm.scale_map is None or norm.offset_map is None:
                    raise ValueError(f"Incomplete local-normalisation maps for {name}")
                scale_map_path = self.directory / f"scale_{i:04d}.npy"
                offset_map_path = self.directory / f"offset_{i:04d}.npy"
                for map_file_path, array in ((scale_map_path, norm.scale_map), (offset_map_path, norm.offset_map)):
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile("wb", dir=self.directory, prefix=f"{map_file_path.name}.", suffix=".tmp", delete=False) as stream:
                            temporary = Path(stream.name)
                            np.save(stream, np.ascontiguousarray(array, dtype=np.float32), allow_pickle=False)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary, map_file_path)
                    finally:
                        if temporary is not None:
                            temporary.unlink(missing_ok=True)
            stored_frame_path = path.name if path.parent == self.directory.resolve() else str(path.resolve())
            frames.append({
                "name": name,
                "path": stored_frame_path,
                "offset": np.asarray(norm.offset, dtype=float).tolist(),
                "scale": np.asarray(norm.scale, dtype=float).tolist(),
                "weight": norm.weight,
                "weight_map": None if weight_map_path is None else weight_map_path.name,
                "scale_map": None if scale_map_path is None else scale_map_path.name,
                "offset_map": None if offset_map_path is None else offset_map_path.name,
            })
        external_dirs = [str(path.resolve()) for path in self.storage_dirs if path.resolve() != self.directory.resolve()]
        index = {"shape": list(self.shape), "storage_dirs": external_dirs, "frames": frames}
        out = self.directory / "index.json"
        write_json_atomic(out, index)
        return out

    def cleanup(self) -> None:
        for p in self.paths + self.mask_paths:
            try:
                p.unlink()
            except OSError:
                pass
        for directory in self.storage_dirs:
            if directory != self.directory:
                try:
                    directory.rmdir()
                except OSError:
                    pass


def registered_store_cleanup_plan(directory: Path) -> list[Path]:
    """Return owned artifacts only after validating the retained-store manifest and index."""

    root = directory.resolve(strict=True)
    cleanup_marker = root / "cleanup_pending.json"
    if cleanup_marker.is_symlink():
        raise ValueError(f"Registered-store cleanup marker is a symlink: {cleanup_marker}")
    if cleanup_marker.exists():
        try:
            marker = json.loads(cleanup_marker.read_text(encoding="utf-8"))
            if not isinstance(marker, dict):
                raise ValueError("Invalid registered-store cleanup marker")
            names = marker["files"]
            schema = marker.get("schema")
            if schema not in (1, 2) or not isinstance(names, list):
                raise ValueError("Invalid registered-store cleanup marker")
            if schema == 1:
                if any(
                    not isinstance(name, str)
                    or name not in ("index.json", "registered_store.json")
                    and re.fullmatch(r"(?:reg|rej|weight|scale|offset)_\d{4}\.npy", name) is None
                    for name in names
                ):
                    raise ValueError("Cleanup marker contains an unowned filename")
                storage_dirs = []
                paths = [root / name for name in names]
            else:
                raw_dirs = marker.get("storage_dirs")
                if not isinstance(raw_dirs, list) or any(not isinstance(path, str) for path in raw_dirs):
                    raise ValueError("Cleanup marker contains invalid storage directories")
                raw_storage_dirs = [Path(path) for path in raw_dirs]
                if any(path.is_symlink() for path in raw_storage_dirs):
                    raise ValueError("Cleanup marker contains a symlinked storage directory")
                storage_dirs = [path.resolve() for path in raw_storage_dirs]
                if any(re.fullmatch(r"\.gpustacker_registered_[0-9a-f]{32}", path.name) is None for path in storage_dirs):
                    raise ValueError("Cleanup marker contains an invalid storage directory")
                paths = [Path(path) for path in names if isinstance(path, str)]
                if len(paths) != len(names):
                    raise ValueError("Cleanup marker contains an invalid artifact path")
        except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot validate cleanup marker in {root}: {exc}") from exc
        validated = []
        for path in [*paths, cleanup_marker]:
            if path.is_symlink():
                raise ValueError(f"Registered-store artifact is a symlink: {path}")
            if not path.exists():
                continue
            resolved = path.resolve(strict=True)
            root_artifact = resolved.parent == root and (
                resolved.name in ("index.json", "registered_store.json", "cleanup_pending.json")
                or re.fullmatch(r"(?:reg|rej|weight|scale|offset)_\d{4}\.npy", resolved.name) is not None
            )
            external_artifact = any(resolved.parent == storage_dir and re.fullmatch(r"(?:reg|rej)_\d{4}\.npy", resolved.name) is not None for storage_dir in storage_dirs)
            if not (root_artifact or external_artifact) or not resolved.is_file():
                raise ValueError(f"Registered-store artifact is not a regular file inside its store: {path}")
            validated.append(resolved)
        return validated

    manifest_path = root / "registered_store.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema") != 1 or manifest.get("stage") != "registered_before_stack":
            raise ValueError("Unsupported or incomplete registered-store manifest")
        raw_dirs = manifest.get("storage_dirs", [])
        if not isinstance(raw_dirs, list) or any(not isinstance(path, str) for path in raw_dirs):
            raise ValueError("Registered-store manifest has invalid storage directories")
        storage_dirs = []
        for value in raw_dirs:
            path = Path(value)
            if path.is_symlink() or re.fullmatch(r"\.gpustacker_registered_[0-9a-f]{32}", path.name) is None:
                raise ValueError("Registered-store manifest has an invalid storage directory")
            resolved = path.resolve(strict=True)
            if not resolved.is_dir():
                raise ValueError(f"Registered-frame storage directory is missing: {path}")
            storage_dirs.append(resolved)
        store = FrameStore.load_index(root, allow_missing=True, allowed_storage_dirs=storage_dirs)
        if manifest.get("index") != "index.json" or manifest.get("shape") != list(store.shape) or manifest.get("frame_count") != len(store):
            raise ValueError("Registered-store manifest does not match its index")
        index = json.loads((root / "index.json").read_text(encoding="utf-8"))
        frames = index["frames"]
        if len(frames) != len(store):
            raise ValueError("Registered-store index has inconsistent frame metadata")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot validate registered store in {root}: {exc}") from exc

    owned = {root / "index.json", manifest_path, *store.paths}
    for path in store.paths:
        mask_path = path.with_name(path.name.replace("reg_", "rej_", 1))
        if mask_path.exists():
            mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
            valid_mask = mask.shape == store.shape[1:] and mask.dtype == np.uint8
            del mask
            if not valid_mask:
                raise ValueError(f"Registered rejection mask has invalid shape or dtype: {mask_path}")
            owned.add(mask_path)
    for entry in frames:
        for key in ("weight_map", "scale_map", "offset_map"):
            value = entry.get(key)
            if value is not None:
                owned.add(root / str(value))
    validated = []
    for path in owned:
        if path.is_symlink():
            raise ValueError(f"Registered-store artifact is a symlink: {path}")
        if not path.exists():
            continue
        resolved = path.resolve(strict=True)
        root_artifact = resolved.parent == root and (
            resolved.name in ("index.json", "registered_store.json", "cleanup_pending.json")
            or re.fullmatch(r"(?:reg|rej|weight|scale|offset)_\d{4}\.npy", resolved.name) is not None
        )
        external_artifact = any(resolved.parent == storage_dir and re.fullmatch(r"(?:reg|rej)_\d{4}\.npy", resolved.name) is not None for storage_dir in storage_dirs)
        if not (root_artifact or external_artifact) or not resolved.is_file():
            raise ValueError(f"Registered-store artifact is not a regular file inside its store: {path}")
        validated.append(resolved)
    return validated


def cleanup_registered_store(directory: Path) -> tuple[int, int]:
    """Delete only artifacts owned by a validated retained-store manifest."""

    validated = registered_store_cleanup_plan(directory)
    total_bytes = sum(path.stat().st_size for path in validated)
    root = directory.resolve(strict=True)
    cleanup_marker = root / "cleanup_pending.json"
    if cleanup_marker.is_symlink():
        raise ValueError(f"Registered-store cleanup marker is a symlink: {cleanup_marker}")
    storage_dirs = []
    if not cleanup_marker.exists():
        manifest_path = root / "registered_store.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        storage_dirs = [str(Path(value).resolve()) for value in manifest.get("storage_dirs", [])]
        write_json_atomic(cleanup_marker, {
            "schema": 2,
            "storage_dirs": storage_dirs,
            "files": [str(path) for path in validated if path != cleanup_marker],
        })
    else:
        marker = json.loads(cleanup_marker.read_text(encoding="utf-8"))
        storage_dirs = marker.get("storage_dirs", []) if marker.get("schema") == 2 else []
    errors = []
    removed_bytes = 0
    removed_count = 0
    for path in (path for path in validated if path != cleanup_marker):
        try:
            size = path.stat().st_size
            path.unlink()
            removed_count += 1
            removed_bytes += size
        except OSError as exc:
            errors.append(f"{path.name}: {exc}")
    if errors:
        raise OSError(f"Removed {removed_count} of {len(validated)} registered-store files ({removed_bytes} bytes); remaining: {'; '.join(errors)}")
    try:
        cleanup_marker.unlink()
    except OSError as exc:
        raise OSError(f"Removed {removed_count} registered-store files ({removed_bytes} bytes), but cleanup marker remains: {exc}") from exc
    for storage_dir in storage_dirs:
        try:
            Path(storage_dir).rmdir()
        except OSError:
            pass
    return removed_count, total_bytes


@dataclass
class StackSettings:
    method: RejectionMethod = "winsorized"
    sigma_low: float = 3.0
    sigma_high: float = 3.0
    iterations: int = 3
    percentile_low: float = 0.2
    percentile_high: float = 0.1
    gesd_outliers: float = 0.3  # max fraction of frames that may be rejected per pixel
    gesd_significance: float = 0.05
    gesd_low_relax: float = 1.5  # >1 makes low (dark) outliers harder to reject
    use_weights: bool = True
    normalize: bool = True
    large_scale: bool = False  # grow high-rejections across dense structures (satellite/plane trails)
    large_scale_box: int = 15
    large_scale_density: float = 0.3
    large_scale_grow: int = 3


@dataclass
class StackResult:
    image: np.ndarray  # (C, H, W)
    rejected_low: int
    rejected_high: int
    total_samples: int
    bands: int
    coverage: np.ndarray | None = None  # (H, W) valid frames per pixel (channel 0)
    rejection: np.ndarray | None = None  # (H, W) rejected fraction per pixel, mean over channels
    crop_box: tuple[int, int, int, int] | None = None  # (y0, y1, x0, x1) applied to image/maps

    @property
    def rejection_fraction(self) -> float:
        return (self.rejected_low + self.rejected_high) / max(1, self.total_samples)


@dataclass
class StackBandResult:
    image: np.ndarray
    rejected_low: int
    rejected_high: int
    total_samples: int
    coverage: np.ndarray
    rejection: np.ndarray
    rejection_masks: np.ndarray | None = None


def _nanstd(x: torch.Tensor, mean: torch.Tensor, dim: int = 0) -> torch.Tensor:
    valid = ~torch.isnan(x)
    count = valid.sum(dim=dim).clamp_min(1)
    diff = torch.where(valid, x - mean, torch.zeros_like(x))
    return torch.sqrt((diff * diff).sum(dim=dim) / count.clamp_min(2).sub(1))


def _sigma_clip(x: torch.Tensor, lo: float, hi: float, iters: int, winsorize: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (clipped stack with NaN rejections, low-reject mask, high-reject mask)."""

    work = x.clone()
    rej_low = torch.zeros_like(x, dtype=torch.bool)
    rej_high = torch.zeros_like(x, dtype=torch.bool)
    for _ in range(max(1, iters)):
        center = torch.nanmedian(work, dim=0).values
        if winsorize:
            # Huber-style: start from MAD, then iterate winsorised mean/sigma (1.134 bias correction)
            std = 1.4826 * torch.nanmedian((work - center).abs(), dim=0).values
            for _w in range(3):
                lo_b, hi_b = center - 1.5 * std, center + 1.5 * std
                wins = torch.minimum(torch.maximum(work, lo_b), hi_b)
                center = torch.nanmean(wins, dim=0)
                std = 1.134 * _nanstd(wins, center)
        else:
            std = _nanstd(work, center)
        std = std.clamp_min(1e-12)
        dev = work - center
        new_low = dev < -lo * std
        new_high = dev > hi * std
        if not bool((new_low | new_high).any()):
            break
        rej_low |= new_low
        rej_high |= new_high
        work = torch.where(new_low | new_high, torch.full_like(work, float("nan")), work)
    return work, rej_low, rej_high


def _percentile_clip(x: torch.Tensor, lo: float, hi: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center = torch.nanmedian(x, dim=0).values
    denom = center.abs().clamp_min(1e-6)
    rel = (x - center) / denom
    rej_low = rel < -lo
    rej_high = rel > hi
    work = torch.where(rej_low | rej_high, torch.full_like(x, float("nan")), x)
    return work, rej_low, rej_high


_GESD_TABLE_CACHE: dict[tuple[int, int, float], np.ndarray] = {}


def gesd_critical_table(n_max: int, k: int, alpha: float) -> np.ndarray:
    """lambda[i-1, n] for i=1..k outliers among n samples (Rosner 1983); inf where undefined."""

    key = (n_max, k, alpha)
    cached = _GESD_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    from scipy.stats import t as student_t

    # lambda depends only on m = n - i, so evaluate the t quantile once per m instead of per (i, n)
    # (the per-cell loop was 14.7M scalar ppf calls ≈ 6 min at n_max=7000, k=2100).
    m = np.arange(2, n_max + 1, dtype=np.float64)
    dof = m - 1.0
    tv = student_t.ppf(1.0 - alpha / (2.0 * (m + 1.0)), dof)
    lam = np.full(n_max + 1, np.inf)
    lam[2:] = m * tv / np.sqrt((dof + tv * tv) * (m + 1.0))
    mi = np.arange(n_max + 1)[None, :] - np.arange(1, k + 1)[:, None]
    table = np.where(mi >= 2, lam[np.clip(mi, 0, n_max)], np.inf).astype(np.float32)
    _GESD_TABLE_CACHE[key] = table
    return table


# Graph replay measured slower than eager (7.1 s vs 5.0 s at N=853; 357 s vs 15 s at N=7000 once the
# graph pool pushed the device past its 8 GB) so it is disabled unless explicitly lowered.
GRAPH_MIN_STEPS = 1 << 62
_graph_stats = {"captured": 0, "fallback": 0, "last_error": None}  # diagnostics for _run_steps


def _run_steps(body: Callable[[], None], steps: int, state: tuple[torch.Tensor, ...]) -> None:
    """Run ``body`` ``steps`` times. On CUDA the body (a few dozen tiny kernels on in-place state) is
    captured once into a CUDA graph and replayed, which removes the per-kernel launch cost that
    otherwise dominates for thousands of steps per band. Falls back to an eager loop anywhere else."""

    if steps <= 0:
        return
    device = state[0].device
    if device.type != "cuda" or steps < GRAPH_MIN_STEPS or not hasattr(torch.cuda, "CUDAGraph"):
        for _ in range(steps):
            body()
        return
    saved = [t.clone() for t in state]
    try:
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            body()  # warm-up (lazy kernel loading) on a side stream, as capture requires
        torch.cuda.current_stream(device).wait_stream(side)
        for t, s in zip(state, saved):
            t.copy_(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            body()
    except Exception as exc:
        _graph_stats["fallback"] += 1
        _graph_stats["last_error"] = exc
        for t, s in zip(state, saved):
            t.copy_(s)
        for _ in range(steps):
            body()
        return
    _graph_stats["captured"] += 1
    for _ in range(steps):
        graph.replay()


def _gesd_clip(x: torch.Tensor, max_frac: float, alpha: float, low_relax: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generalized Extreme Studentized Deviate test, vectorised over pixels.

    The most extreme sample is always the current min or max, so each pixel is sorted once and the
    test walks inward with running sums: O(N log N) sort + k cheap steps on (pixels,) vectors.
    Only the accepted step count and its low/high split are kept per pixel, so device memory stays
    at ~4x the band regardless of k. Steps past a pixel's valid sample count are neutralised by the
    critical table (lambda = inf there), which removes all per-step masking from the loop.
    """

    n = x.shape[0]
    k = int(max(1, min(n - 3, np.floor(n * max_frac))))
    if n < 4 or k < 1:
        return x, torch.zeros_like(x, dtype=torch.bool), torch.zeros_like(x, dtype=torch.bool)
    flat = x.reshape(n, -1)
    pixels = flat.shape[1]
    device = x.device
    n_valid = (~torch.isnan(flat)).sum(dim=0)  # (P,) int64
    sorted_vals, sort_idx = torch.sort(flat, dim=0)  # NaN sorts last
    # lower median of the valid samples (what torch.nanmedian returns); NaN where nothing is valid
    median = sorted_vals.gather(0, ((n_valid - 1).clamp_min(0) // 2).view(1, -1))[0]
    sorted_vals.sub_(median)  # centred values keep the running sums well conditioned
    # running sums in float64: downdating the sum of squares after removing a saturated outlier
    # cancels catastrophically in float32 (the per-step vectors are tiny, so this costs little)
    count = n_valid.to(torch.float64)
    s1 = torch.nansum(sorted_vals, dim=0, dtype=torch.float64)
    s2 = torch.empty_like(s1)
    chunk = max(1, pixels // 8)  # bounds the squared temporary while the int64 sort indices are alive
    for a in range(0, pixels, chunk):
        part = sorted_vals[:, a : a + chunk]
        s2[a : a + chunk] = torch.nansum(part * part, dim=0, dtype=torch.float64)
    lo = torch.zeros(pixels, dtype=torch.long, device=device)
    hi = (n_valid - 1).clamp_min(0)
    n_out = torch.zeros(pixels, dtype=torch.long, device=device)
    low_out = torch.zeros(pixels, dtype=torch.long, device=device)
    table = torch.from_numpy(gesd_critical_table(n, k, alpha)).to(device)  # (k, n + 1), inf where undefined
    inv_relax = 1.0 / low_relax
    step = torch.zeros(1, dtype=torch.long, device=device)  # steps taken so far (device-side loop counter)

    def walk_step() -> None:
        """One GESD step, entirely in place on the state tensors (no host syncs, so it can be graph-captured)."""

        c = count.clamp_min(1.0)
        mean = s1 / c
        inv_std = ((s2 - s1 * mean) / (c - 1.0).clamp_min(1.0)).clamp_min(1e-24).rsqrt()
        ends = sorted_vals.gather(0, torch.stack((lo, hi))).to(torch.float64)  # (2, P): current min, max
        vlo, vhi = ends[0], ends[1]
        dlo = (mean - vlo) * inv_std
        if low_relax != 1.0:
            dlo = dlo * inv_relax
        dhi = (vhi - mean) * inv_std
        pick_hi = dhi >= dlo
        r_i = torch.where(pick_hi, dhi, dlo)
        lam = table.index_select(0, step)[0].index_select(0, n_valid)  # inf for exhausted pixels
        exceeds = r_i > lam  # False for NaN r_i
        v = torch.where(pick_hi, vhi, vlo)
        s1.sub_(v)
        s2.addcmul_(v, v, value=-1.0)
        count.sub_(1.0)
        took_hi = pick_hi.to(torch.long)
        hi.sub_(took_hi).clamp_min_(0)
        lo.add_(1 - took_hi).clamp_max_(n - 1)
        step.add_(1)
        torch.where(exceeds, step, n_out, out=n_out)  # accepted step count (step already = i + 1)
        torch.where(exceeds, lo, low_out, out=low_out)  # low-side removals among the accepted steps

    _run_steps(walk_step, k, (s1, s2, count, lo, hi, n_out, low_out, step))
    del sorted_vals, s1, s2, count, lo, hi
    # the removed samples are the lowest low_out and the highest (n_out - low_out) sorted positions
    rank = torch.arange(n, device=device).view(n, 1)
    reject_sorted = (rank < low_out) | ((rank >= n_valid - (n_out - low_out)) & (rank < n_valid))
    # un-sort only the (few) rejected positions: far cheaper than a full scatter of the mask
    sort_idx.mul_(pixels).add_(torch.arange(pixels, device=device))
    reject = torch.zeros(n * pixels, dtype=torch.bool, device=device)
    reject.index_fill_(0, sort_idx[reject_sorted], True)
    reject = reject.view(n, pixels)
    del sort_idx, reject_sorted
    below = flat < median
    rej_low = (reject & below).reshape(x.shape)
    rej_high = (reject & ~below).reshape(x.shape)
    out = torch.where(reject, float("nan"), flat).reshape(x.shape)
    return out, rej_low, rej_high


def _large_scale_reject(x: torch.Tensor, rej_high: torch.Tensor, settings: StackSettings) -> torch.Tensor:
    """Where high-rejections are dense (a trail), also reject their dilated neighbourhood in that frame."""

    n, channels = x.shape[0], x.shape[1]
    flat = rej_high.reshape(n * channels, 1, *x.shape[2:]).to(x.dtype)
    box = settings.large_scale_box | 1
    density = F.avg_pool2d(flat, box, stride=1, padding=box // 2, count_include_pad=False)
    dense = (density > settings.large_scale_density).to(x.dtype)
    grow = 2 * settings.large_scale_grow + 1
    region = F.max_pool2d(dense, grow, stride=1, padding=grow // 2) > 0
    extra = region.reshape(x.shape) & ~torch.isnan(x) & ~rej_high
    return extra


@dataclass
class CombineOut:
    image: torch.Tensor  # (...) combined
    rej_low: torch.Tensor | None  # (N, ...) bool
    rej_high: torch.Tensor | None
    valid: torch.Tensor  # (N, ...) bool, input samples that were present


def combine_full(x: torch.Tensor, settings: StackSettings, weights: torch.Tensor | None = None) -> CombineOut:
    """Combine an (N, ...) tensor along axis 0, returning per-sample rejection masks."""

    present = ~torch.isnan(x)
    if settings.method == "median":
        return CombineOut(torch.nan_to_num(torch.nanmedian(x, dim=0).values, nan=0.0), None, None, present)
    if settings.method == "none":
        work, rl, rh = x, None, None
    elif settings.method in ("sigma", "winsorized"):
        work, rl, rh = _sigma_clip(x, settings.sigma_low, settings.sigma_high, settings.iterations, settings.method == "winsorized")
    elif settings.method == "percentile":
        work, rl, rh = _percentile_clip(x, settings.percentile_low, settings.percentile_high)
    elif settings.method == "gesd":
        work, rl, rh = _gesd_clip(x, settings.gesd_outliers, settings.gesd_significance, settings.gesd_low_relax)
    else:
        raise ValueError(f"Unknown rejection method {settings.method!r}")
    if settings.large_scale and rh is not None and x.ndim == 4:
        extra = _large_scale_reject(x, rh, settings)
        rh = rh | extra
        work = torch.where(extra, torch.full_like(work, float("nan")), work)
    valid = ~torch.isnan(work)
    if weights is None:
        out = torch.nanmean(work, dim=0)
    else:
        w = weights.to(x.dtype)
        if w.ndim == 1:
            w = w.view(-1, *([1] * (x.ndim - 1)))
        elif w.ndim == x.ndim - 1:
            w = w.unsqueeze(1)
        if work is x:
            work = work.clone()
        work.mul_(w)  # in place: NaN stays NaN, so nansum is the weighted sum over valid samples
        num = torch.nansum(work, dim=0)
        den = torch.where(valid, w, 0.0).sum(dim=0)
        out = num / den.clamp_min(1e-12)
    # pixels rejected everywhere fall back to the plain median so nothing is left as NaN
    if bool(torch.isnan(out).any()):
        fallback = torch.nanmedian(x, dim=0).values
        out = torch.where(torch.isnan(out), fallback, out)
        out = torch.nan_to_num(out, nan=0.0)
    return CombineOut(out, rl, rh, present)


def combine_tensor(x: torch.Tensor, settings: StackSettings, weights: torch.Tensor | None = None) -> tuple[torch.Tensor, int, int]:
    """Combine an (N, ...) tensor along axis 0. Returns (image, n_rejected_low, n_rejected_high)."""

    res = combine_full(x, settings, weights)
    return res.image, int(res.rej_low.sum()) if res.rej_low is not None else 0, int(res.rej_high.sum()) if res.rej_high is not None else 0


class LocalNormMaps:
    """All frames' low-res local-normalisation maps on the device, upsampled per band on demand.

    Equivalent to bilinear ``F.interpolate(..., align_corners=False)`` of each frame's full map followed
    by a row slice, but only the band's rows are ever materialised and all frames share one kernel launch
    per chunk instead of two full-frame interpolations per frame per band.
    """

    def __init__(self, norms: list[FrameNorm], channels: int, device: torch.device, max_chunk_bytes: int = 256 * 2**20) -> None:
        shapes = {tuple(norm.scale_map.shape) for norm in norms if norm.scale_map is not None and norm.offset_map is not None}
        self.enabled = bool(shapes)
        if not self.enabled:
            return
        if len(shapes) != 1:
            raise ValueError(f"Local normalisation maps have mixed shapes: {sorted(shapes)}")
        shape = next(iter(shapes))
        n = len(norms)
        scale = np.ones((n, *shape), dtype=np.float32)
        offset = np.zeros((n, *shape), dtype=np.float32)
        for i, norm in enumerate(norms):
            if norm.scale_map is not None and norm.offset_map is not None:
                scale[i] = norm.scale_map
                offset[i] = norm.offset_map
        self.map_h, self.map_w = int(shape[-2]), int(shape[-1])
        self.channels = channels
        self.scale = torch.from_numpy(scale).to(device).reshape(n * channels, 1, self.map_h, self.map_w)
        self.offset = torch.from_numpy(offset).to(device).reshape(n * channels, 1, self.map_h, self.map_w)
        self.max_chunk_bytes = max_chunk_bytes

    def apply_(self, x: torch.Tensor, r0: int, height: int) -> torch.Tensor:
        """In place: x[i] = x[i] * scale_i + offset_i for band rows r0..r0+rows of a (N, C, rows, W) tensor."""

        if not self.enabled:
            return x
        n, channels, rows, width = x.shape
        # normalised sample coordinates matching F.interpolate(align_corners=False) + border clamp
        ys = (2.0 * torch.arange(r0, r0 + rows, device=x.device, dtype=torch.float32) + 1.0) / height - 1.0
        xs = (2.0 * torch.arange(width, device=x.device, dtype=torch.float32) + 1.0) / width - 1.0
        grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij")[::-1], dim=-1).unsqueeze(0)  # (1, rows, W, 2) as (x, y)
        flat = x.view(n * channels, rows, width)  # view, so the in-place update reaches x
        per_plane = rows * width * 4 * 2  # scale + offset planes in float32
        chunk = max(1, min(flat.shape[0], self.max_chunk_bytes // per_plane))
        for start in range(0, flat.shape[0], chunk):
            stop = min(flat.shape[0], start + chunk)
            g = grid.expand(stop - start, rows, width, 2)
            s = F.grid_sample(self.scale[start:stop], g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0]
            o = F.grid_sample(self.offset[start:stop], g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0]
            flat[start:stop].mul_(s).add_(o)
        return x


def _apply_norm_maps(x: torch.Tensor, norms: list[FrameNorm], r0: int, r1: int, height: int) -> torch.Tensor:
    """Apply per-frame local normalisation maps to a band in place. x: (N, C, rows, W)."""

    return LocalNormMaps(norms, x.shape[1], x.device).apply_(x, r0, height)


def _combine_stack_band(
    x: torch.Tensor,
    store: FrameStore,
    backend: Backend,
    settings: StackSettings,
    r0: int,
    reject_masks: bool,
    offsets: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor | None,
    local_maps: LocalNormMaps | None = None,
    output_start: int = 0,
    output_stop: int | None = None,
) -> StackBandResult:
    n, channels, rows, width = x.shape
    height = store.shape[1]
    if local_maps is None:
        local_maps = LocalNormMaps(store.norms, channels, x.device)
    if local_maps.enabled:
        x = local_maps.apply_(x, r0, height)
    elif settings.normalize:
        x.sub_(offsets).mul_(scales).add_(offsets[0])  # in place: x is the band's own device copy

    band_weights = weights
    if settings.use_weights and any(norm.weight_map is not None for norm in store.norms):
        base_weights = weights if weights is not None else torch.ones(n, device=backend.device)
        gy = torch.zeros(rows, device=backend.device) if height == 1 else 2.0 * torch.arange(r0, r0 + rows, device=backend.device) / (height - 1) - 1.0
        gx = torch.zeros(width, device=backend.device) if width == 1 else torch.linspace(-1.0, 1.0, width, device=backend.device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        sample_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        spatial = []
        for norm in store.norms:
            if norm.weight_map is None:
                spatial.append(torch.ones((1, rows, width), device=backend.device))
            else:
                field = torch.as_tensor(norm.weight_map, dtype=torch.float32, device=backend.device)[None, None]
                spatial.append(F.grid_sample(field, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)[0])
        band_weights = base_weights.view(n, 1, 1, 1) * torch.stack(spatial)

    combined = combine_full(x, settings, band_weights)
    combined.image = combined.image[:, output_start:output_stop]
    combined.valid = combined.valid[:, :, output_start:output_stop]
    if combined.rej_low is not None:
        combined.rej_low = combined.rej_low[:, :, output_start:output_stop]
    if combined.rej_high is not None:
        combined.rej_high = combined.rej_high[:, :, output_start:output_stop]
    rows = combined.image.shape[1]
    image = combined.image.cpu().numpy()
    coverage = combined.valid[:, 0].sum(dim=0).to(torch.int32).cpu().numpy().astype(np.uint16)
    rejection_count = torch.zeros_like(combined.valid[0], dtype=torch.float32)
    rejected_low = rejected_high = 0
    if combined.rej_low is not None:
        rejection_count += combined.rej_low.sum(dim=0).to(torch.float32)
        rejected_low = int(combined.rej_low.sum())
    if combined.rej_high is not None:
        rejection_count += combined.rej_high.sum(dim=0).to(torch.float32)
        rejected_high = int(combined.rej_high.sum())
    present = combined.valid.sum(dim=0).to(torch.float32).clamp_min(1.0)
    rejection = (rejection_count / present).mean(dim=0).cpu().numpy()
    masks = None
    if reject_masks:
        rejected = torch.zeros_like(combined.valid, dtype=torch.bool)
        if combined.rej_low is not None:
            rejected |= combined.rej_low
        if combined.rej_high is not None:
            rejected |= combined.rej_high
        masks = rejected.any(dim=1).to(torch.uint8).cpu().numpy()
    return StackBandResult(image, rejected_low, rejected_high, n * channels * rows * width, coverage, rejection, masks)


class BandReader:
    """Reads the same row range out of every registered ``.npy`` frame with explicit, threaded reads.

    A band is one contiguous run of bytes per channel per frame, so seek + readinto from several
    threads keeps an NVMe busy; strided memmap copies from one thread run at a fraction of that.
    """

    def __init__(self, paths: list[Path], shape: tuple[int, int, int], workers: int = 8, data_offsets: list[int] | None = None) -> None:
        channels, height, width = shape
        self.paths = [str(p) for p in paths]
        self.shape = shape
        self.row_bytes = width * 4
        self.plane_bytes = height * self.row_bytes
        self.workers = max(1, workers)
        self.data_offsets: list[int] = []
        if data_offsets is not None:
            if len(data_offsets) != len(paths) or any(offset < 0 for offset in data_offsets):
                raise ValueError("Registered-frame data offsets do not match the store")
            self.data_offsets = list(data_offsets)
            return
        for path in self.paths:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError(f"Registered frame {path} does not match the store shape")
            self.data_offsets.append(int(array.offset))
            del array

    def __len__(self) -> int:
        return len(self.paths)

    def read_into(self, row_start: int, row_stop: int, out: np.ndarray) -> None:
        """Fill ``out[:, :, :rows, :]`` (N, C, >=rows, W float32, C-contiguous) with the row range."""

        rows = row_stop - row_start
        channels = self.shape[0]
        if out.dtype != np.float32 or out.shape[0] != len(self.paths) or out.shape[1] != channels or out.shape[2] < rows or out.shape[3] != self.shape[2]:
            raise ValueError("Band buffer does not match the store")
        band_bytes = rows * self.row_bytes
        row_offset = row_start * self.row_bytes

        def read_frames(first: int, last: int) -> None:
            for index in range(first, last):
                base = self.data_offsets[index] + row_offset
                with open(self.paths[index], "rb", buffering=0) as handle:
                    for c in range(channels):
                        handle.seek(base + c * self.plane_bytes)
                        view = memoryview(out[index, c, :rows, :]).cast("B")
                        filled = 0
                        while filled < band_bytes:
                            got = handle.readinto(view[filled:])
                            if not got:
                                raise OSError(f"Short read in registered frame {self.paths[index]}")
                            filled += got

        n = len(self.paths)
        workers = min(self.workers, n)
        if workers <= 1:
            read_frames(0, n)
            return
        step = math.ceil(n / workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(read_frames, first, min(n, first + step)) for first in range(0, n, step)]
            for future in futures:
                future.result()

    def read(self, row_start: int, row_stop: int) -> np.ndarray:
        band = np.empty((len(self.paths), self.shape[0], row_stop - row_start, self.shape[2]), dtype=np.float32)
        self.read_into(row_start, row_stop, band)
        return band


def read_store_band(maps: list[np.memmap] | BandReader, shape: tuple[int, int, int], row_start: int, row_stop: int) -> np.ndarray:
    """Copy an absolute row range from every registered frame into one (N, C, rows, W) array."""

    if isinstance(maps, BandReader):
        return maps.read(row_start, row_stop)
    channels, _, width = shape
    band = np.empty((len(maps), channels, row_stop - row_start, width), dtype=np.float32)
    for index, frame in enumerate(maps):
        band[index] = frame[:, row_start:row_stop, :]
    return band


def band_overhead(settings: StackSettings) -> float:
    """Device-memory multiple of one band that the chosen rejection needs for its intermediates."""

    # gesd: torch.sort along the frame axis peaks at ~6x band for N=853 and ~13x for N=7000
    # (values + int64 indices + transposed scratch), so size bands for the large-N case.
    overhead = 14.0 if settings.method == "gesd" else 6.0
    if settings.large_scale:
        overhead += 2.0
    return overhead


def _stack_band_halo(settings: StackSettings) -> int:
    if not settings.large_scale or settings.method in ("none", "median"):
        return 0
    return (settings.large_scale_box | 1) // 2 + settings.large_scale_grow


def _stack_band_rows(store: FrameStore, backend: Backend, settings: StackSettings) -> int:
    channels, height, width = store.shape
    capacity = rows_per_band(len(store), width, channels, backend.budget_bytes, overhead=band_overhead(settings))
    halo = _stack_band_halo(settings)
    if halo and capacity < min(height, 2 * halo + 1):
        raise ValueError("Insufficient device memory for large-scale rejection overlap; disable large-scale rejection or reduce the frame count")
    return max(1, capacity - 2 * halo)


def stack_store_band(
    store: FrameStore,
    backend: Backend,
    settings: StackSettings,
    row_start: int,
    row_stop: int,
    reject_masks: bool = False,
    frame_maps: list[np.memmap] | BandReader | None = None,
    host_data: np.ndarray | None = None,
    cancel_cb: Callable[[], None] | None = None,
) -> StackBandResult:
    """Stack an absolute half-open row range with the same math as ``stack_store``."""

    n = len(store)
    if n == 0:
        raise ValueError("No frames to stack")
    channels, height, width = store.shape
    if row_start < 0 or row_stop <= row_start or row_stop > height:
        raise ValueError("Stack band row range is outside the registered store")
    max_band = _stack_band_rows(store, backend, settings)
    halo = _stack_band_halo(settings)
    data_start = max(0, row_start - halo)
    data_stop = min(height, row_stop + halo)
    maps: list[np.memmap] | BandReader = []
    if host_data is None:
        maps = store.band_reader() if frame_maps is None else frame_maps
        if len(maps) != n:
            raise ValueError("Registered-frame map count does not match the store")
    elif host_data.shape != (n, channels, data_stop - data_start, width) or host_data.dtype != np.float32:
        raise ValueError("Pre-read stack band does not match the requested rows")
    offsets = torch.from_numpy(np.stack([norm_vec(norm.offset, channels) for norm in store.norms])).to(backend.device).view(n, channels, 1, 1)
    scales = torch.from_numpy(np.stack([norm_vec(norm.scale, channels) for norm in store.norms])).to(backend.device).view(n, channels, 1, 1)
    weights = torch.tensor([norm.weight for norm in store.norms], device=backend.device) if settings.use_weights else None
    local_maps = LocalNormMaps(store.norms, channels, backend.device)
    row_count = row_stop - row_start
    image = np.empty((channels, row_count, width), dtype=np.float32)
    coverage = np.empty((row_count, width), dtype=np.uint16)
    rejection = np.empty((row_count, width), dtype=np.float32)
    masks = np.empty((n, row_count, width), dtype=np.uint8) if reject_masks else None
    rejected_low = rejected_high = total_samples = 0
    for absolute_start in range(row_start, row_stop, max_band):
        if cancel_cb is not None:
            cancel_cb()
        absolute_stop = min(row_stop, absolute_start + max_band)
        read_start = max(0, absolute_start - halo)
        read_stop = min(height, absolute_stop + halo)
        rows = read_stop - read_start
        if host_data is None:
            host = torch.empty((n, channels, rows, width), dtype=torch.float32, pin_memory=backend.is_cuda)
            host_np = host.numpy()
            if isinstance(maps, BandReader):
                maps.read_into(read_start, read_stop, host_np)
            else:
                for index, frame in enumerate(maps):
                    host_np[index] = frame[:, read_start:read_stop, :]
        else:
            offset = read_start - data_start
            host = torch.from_numpy(host_data[:, :, offset : offset + rows, :]).clone()  # the band is normalised in place
        tensor = host.to(backend.device, non_blocking=backend.is_cuda)
        result = _combine_stack_band(
            tensor, store, backend, settings, read_start, reject_masks, offsets, scales, weights, local_maps,
            absolute_start - read_start, absolute_stop - read_start,
        )
        local_start = absolute_start - row_start
        local_stop = absolute_stop - row_start
        image[:, local_start:local_stop, :] = result.image
        coverage[local_start:local_stop, :] = result.coverage
        rejection[local_start:local_stop, :] = result.rejection
        if masks is not None and result.rejection_masks is not None:
            masks[:, local_start:local_stop, :] = result.rejection_masks
        rejected_low += result.rejected_low
        rejected_high += result.rejected_high
        total_samples += result.total_samples
    backend.empty_cache()
    return StackBandResult(image, rejected_low, rejected_high, total_samples, coverage, rejection, masks)


def stack_store(store: FrameStore, backend: Backend, settings: StackSettings, progress_cb: ProgressCallback = _noop_progress, label: str = "Stacking", reject_masks: bool = False, cancel_cb: Callable[[], None] | None = None) -> StackResult:
    """Stream the registered frames through the device in row bands and combine them.

    With ``reject_masks`` a (H, W) uint8 mask per frame (1 = rejected in any channel) is written next
    to the registered frames as ``rej_####.npy`` for drizzle to honour.
    """

    n = len(store)
    if n == 0:
        raise ValueError("No frames to stack")
    channels, height, width = store.shape
    reader = store.band_reader()
    mask_maps: list[np.memmap] = []
    if reject_masks:
        store.mask_paths = [p.with_name(p.name.replace("reg_", "rej_")) for p in store.paths]
        mask_maps = [np.lib.format.open_memmap(p, mode="w+", dtype=np.uint8, shape=(height, width)) for p in store.mask_paths]
    band = _stack_band_rows(store, backend, settings)
    halo = _stack_band_halo(settings)
    buffer_rows = min(height, band + 2 * halo)
    bands = math.ceil(height / band)
    out = np.empty((channels, height, width), dtype=np.float32)
    coverage = np.empty((height, width), dtype=np.uint16)
    rejection = np.empty((height, width), dtype=np.float32)

    offsets = torch.from_numpy(np.stack([norm_vec(f.offset, channels) for f in store.norms])).to(backend.device).view(n, channels, 1, 1)
    scales = torch.from_numpy(np.stack([norm_vec(f.scale, channels) for f in store.norms])).to(backend.device).view(n, channels, 1, 1)
    ref_offset = offsets[0]
    weights = torch.tensor([f.weight for f in store.norms], device=backend.device) if settings.use_weights else None
    local_maps = LocalNormMaps(store.norms, channels, backend.device)

    host_buffers = [torch.empty((n, channels, buffer_rows, width), dtype=torch.float32, pin_memory=backend.is_cuda)]
    if backend.is_cuda and bands > 1:
        host_buffers.append(torch.empty((n, channels, buffer_rows, width), dtype=torch.float32, pin_memory=True))
    host_np_buffers = [buffer.numpy() for buffer in host_buffers]

    def stage_host_band(buffer_index: int, r0: int, r1: int) -> None:
        reader.read_into(max(0, r0 - halo), min(height, r1 + halo), host_np_buffers[buffer_index])

    rej_low = rej_high = 0
    prefetch = ThreadPoolExecutor(max_workers=1) if len(host_buffers) > 1 else None
    pending_stage: Future | None = None
    try:
        for bi in range(bands):
            if cancel_cb is not None:
                cancel_cb()
            buffer_index = bi % len(host_buffers)
            r0 = bi * band
            r1 = min(height, r0 + band)
            read_start = max(0, r0 - halo)
            read_stop = min(height, r1 + halo)
            rows = read_stop - read_start
            if pending_stage is not None:
                pending_stage.result()
                pending_stage = None
            else:
                stage_host_band(buffer_index, r0, r1)
            host = host_buffers[buffer_index]
            x = host[:, :, :rows, :].to(backend.device, non_blocking=True)
            if prefetch is not None and bi + 1 < bands:
                next_r0 = (bi + 1) * band
                next_r1 = min(height, next_r0 + band)
                pending_stage = prefetch.submit(stage_host_band, (bi + 1) % len(host_buffers), next_r0, next_r1)
            band_result = _combine_stack_band(
                x, store, backend, settings, read_start, reject_masks, offsets, scales, weights, local_maps,
                r0 - read_start, r1 - read_start,
            )
            out[:, r0:r1, :] = band_result.image
            coverage[r0:r1, :] = band_result.coverage
            rejection[r0:r1, :] = band_result.rejection
            rej_low += band_result.rejected_low
            rej_high += band_result.rejected_high
            if mask_maps and band_result.rejection_masks is not None:
                for i, mm in enumerate(mask_maps):
                    mm[r0:r1, :] = band_result.rejection_masks[i]
            progress_cb((bi + 1) / bands, f"{label} band {bi + 1}/{bands}")
    finally:
        if prefetch is not None:
            prefetch.shutdown(wait=True, cancel_futures=True)
    for mm in mask_maps:
        mm.flush()
    del mask_maps
    backend.empty_cache()
    return StackResult(out, rej_low, rej_high, n * channels * height * width, bands, coverage, rejection)


def estimate_norm(image: torch.Tensor, sample_stride: int = 8) -> tuple[float, float]:
    """Robust (median, MAD-sigma) of valid pixels, sub-sampled for speed."""

    sample = image[..., ::sample_stride, ::sample_stride].flatten()
    sample = sample[~torch.isnan(sample)]
    if sample.numel() == 0:
        return 0.0, 1.0
    med = torch.median(sample)
    mad = torch.median((sample - med).abs()) * 1.4826
    return float(med), float(mad.clamp_min(1e-9))
