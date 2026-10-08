"""Preflight estimates for stack working and output disk usage."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .io import SUPPORTED_SUFFIXES, load_meta


@dataclass(frozen=True)
class DiskSpaceEstimate:
    working_bytes: int
    output_bytes: int
    free_bytes: int
    volume: Path

    @property
    def expected_bytes(self) -> int:
        return self.working_bytes + self.output_bytes

    @property
    def recommended_bytes(self) -> int:
        return int(self.expected_bytes * 1.1)


def format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def estimate_stack_disk_space(settings, outputs: list[Path], storage_dir: Path, scratch_frames: int | None = None) -> DiskSpaceEstimate:
    """Estimate peak work files and planned image outputs from one input's metadata.

    Input frames are assumed to share dimensions and channel layout. The scratch estimate uses
    every frame because quality filters may not reject any; it uses the largest batch group when
    ``scratch_frames`` is provided.
    """

    if not settings.lights:
        raise ValueError("Cannot estimate disk use without input frames")
    meta = load_meta(settings.lights[0])
    height, width = meta.shape
    pattern = (settings.bayer_pattern or meta.bayer_pattern or "").upper()
    cfa = pattern in ("RGGB", "BGGR", "GRBG", "GBRG")
    debayer = settings.debayer == "on" or (settings.debayer == "auto" and cfa)
    channels = 3 if debayer and meta.channels == 1 else meta.channels
    pixels = height * width
    master_bytes = pixels * channels * 4
    frame_count = scratch_frames if scratch_frames is not None else len(settings.lights)
    rejection_masks = pixels * frame_count if settings.drizzle.enabled or settings.mfdeconv is not None else 0
    working_bytes = frame_count * master_bytes + rejection_masks

    cfa_drizzle = settings.drizzle.cfa and cfa and debayer
    drizzle_channels = 3 if cfa_drizzle else channels
    scale = settings.drizzle.scale
    output_bytes = 0
    for output in outputs:
        stem = output.stem.lower()
        if output.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if stem.endswith("_coverage") or stem.endswith("_rejection"):
            output_bytes += pixels * 4
        elif stem.endswith("_drizzle_weight"):
            output_bytes += pixels * scale * scale * 4
        elif stem.endswith("_drizzle"):
            output_bytes += pixels * scale * scale * drizzle_channels * 4
        else:
            output_bytes += master_bytes

    directory = storage_dir.expanduser()
    while not directory.exists() and directory != directory.parent:
        directory = directory.parent
    usage = shutil.disk_usage(directory)
    volume = Path(directory.anchor) if directory.anchor else directory
    return DiskSpaceEstimate(working_bytes, output_bytes, int(usage.free), volume)