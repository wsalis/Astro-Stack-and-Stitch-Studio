"""Batch mode: group light frames by FITS keywords and stack one master per group."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .backend import StatusCallback
from .io import SUPPORTED_SUFFIXES

ProgressCallback = Callable[[float, str], None]

DEFAULT_GROUP_KEYS = ("OBJECT", "FILTER")
# pseudo-keys that are derived rather than read verbatim
DERIVED_KEYS = {
    "NIGHT": "observing night (DATE-OBS shifted by -12h)",
    "FOLDER": "parent folder name",
    "SIZE": "image dimensions",
}


def _noop_status(_: str) -> None:
    pass


def _noop_progress(_: float, __: str) -> None:
    pass


@dataclass(frozen=True)
class GroupKey:
    values: tuple[tuple[str, str], ...]  # ((keyword, value), ...) in group_by order

    @property
    def label(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.values)

    @property
    def slug(self) -> str:
        parts = [_slug(v) for _, v in self.values]
        return "_".join(p for p in parts if p) or "group"

    def format(self, template: str) -> str:
        out = template
        for k, v in self.values:
            out = out.replace("{" + k + "}", _slug(v))
        return _slug(out, keep="_-.")


def _slug(text: str, keep: str = "_-") -> str:
    text = re.sub(r"\s+", "_", str(text).strip())
    text = "".join(ch for ch in text if ch.isalnum() or ch in keep)
    return text.strip("_") or ""


def read_header(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".xisf":
        from xisf import XISF

        meta = XISF(str(path)).get_images_metadata()[0]
        return {k: v[0].get("value") for k, v in meta.get("FITSKeywords", {}).items() if v}
    from astropy.io import fits

    header = fits.getheader(path)
    return {k: v for k, v in header.items() if k and k not in ("COMMENT", "HISTORY")}


def _night(header: dict[str, Any]) -> str:
    value = header.get("DATE-OBS") or header.get("DATE")
    if not value:
        return "UNKNOWN"
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", ""))
    except ValueError:
        return str(value)[:10]
    return (stamp - timedelta(hours=12)).strftime("%Y-%m-%d")


def key_value(path: Path, header: dict[str, Any], keyword: str) -> str:
    kw = keyword.upper()
    if kw == "NIGHT":
        return _night(header)
    if kw == "FOLDER":
        return path.parent.name
    if kw == "SIZE":
        return f"{header.get('NAXIS1', '?')}x{header.get('NAXIS2', '?')}"
    value = header.get(kw)
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).strip() or "UNKNOWN"


@dataclass
class FrameGroup:
    key: GroupKey
    lights: list[Path] = field(default_factory=list)


def group_frames(lights: list[Path], group_by: tuple[str, ...] | list[str] = DEFAULT_GROUP_KEYS, status_cb: StatusCallback = _noop_status) -> list[FrameGroup]:
    """Split lights into groups by header keywords (header-only reads). Order follows first appearance."""

    keys = tuple(k.strip().upper() for k in group_by if k.strip())
    groups: dict[GroupKey, FrameGroup] = {}
    for path in lights:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            header = read_header(path)
        except Exception as exc:  # unreadable file: report and skip
            status_cb(f"Skipping {path.name}: cannot read header ({exc})")
            continue
        key = GroupKey(tuple((k, key_value(path, header, k)) for k in keys))
        groups.setdefault(key, FrameGroup(key)).lights.append(path)
    return list(groups.values())


def describe_groups(groups: list[FrameGroup]) -> list[str]:
    return [f"{g.key.label or '(all)'}: {len(g.lights)} frame(s) -> {g.key.slug}" for g in groups]


@dataclass
class BatchResult:
    outputs: list[tuple[FrameGroup, Path | None, str | None]]  # (group, stack path or None, error)
    results: list[Any] = field(default_factory=list)  # PipelineResult or None, parallel to outputs


def run_batch(settings, groups: list[FrameGroup], output_dir: Path, name_template: str | None = None, status_cb: StatusCallback = _noop_status, progress_cb: ProgressCallback = _noop_progress, min_frames: int = 2, on_pipeline: Callable[[Any], None] | None = None, cancelled: Callable[[], bool] | None = None):
    """Stack each group with a copy of ``settings`` (a PipelineSettings); returns BatchResult."""

    from .pipeline import StackingPipeline

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[FrameGroup, Path | None, str | None]] = []
    pipeline_results: list[Any] = []
    total = len(groups)
    for gi, group in enumerate(groups):
        if cancelled is not None and cancelled():
            status_cb("Batch cancelled")
            break
        name = group.key.format(name_template) if name_template else group.key.slug
        out = output_dir / f"{name}.fit"
        status_cb(f"=== Group {gi + 1}/{total}: {group.key.label or '(all)'} — {len(group.lights)} frame(s) -> {out.name}")
        if len(group.lights) < min_frames:
            status_cb(f"Skipping group: fewer than {min_frames} frames")
            results.append((group, None, f"fewer than {min_frames} frames"))
            pipeline_results.append(None)
            continue
        reference = settings.reference
        if reference is not None and not any(path.resolve() == reference.resolve() for path in group.lights):
            reference = None
        group_settings = replace(settings, lights=list(group.lights), output=out, work_dir=output_dir / f"_gpustacker_work_{name}", reference=reference)
        try:
            pipeline = StackingPipeline(group_settings, status_cb, lambda f, m, gi=gi: progress_cb((gi + f) / total, f"[{gi + 1}/{total}] {m}"))
            if on_pipeline is not None:
                on_pipeline(pipeline)
            result = pipeline.run()
            results.append((group, result.output, None))
            pipeline_results.append(result)
        except Exception as exc:  # keep going with the remaining groups
            status_cb(f"Group failed: {type(exc).__name__}: {exc}")
            results.append((group, None, f"{type(exc).__name__}: {exc}"))
            pipeline_results.append(None)
    done = sum(1 for _, p, _ in results if p)
    status_cb(f"Batch finished: {done}/{total} group(s) stacked into {output_dir}")
    return BatchResult(results, pipeline_results)
