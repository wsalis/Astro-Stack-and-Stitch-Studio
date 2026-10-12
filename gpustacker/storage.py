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
    group_details: tuple[tuple[str, int, int], ...] = ()
    volume_usage: tuple[tuple[Path, int, int], ...] = ()

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


def estimate_stack_disk_space(
    settings,
    outputs: list[Path],
    storage_dir: Path,
    scratch_frames: int | None = None,
    group_inputs: list[tuple[str, list[Path], list[Path]]] | None = None,
) -> DiskSpaceEstimate:
    """Conservatively estimate scratch and outputs for mixed frames or sequential batch groups.

    Registered arrays and per-frame masks use the largest possible reference grid in each group.
    Batch scratch is the largest group's peak because groups run sequentially; planned outputs add
    across groups. Ten percent headroom is exposed separately via ``recommended_bytes``.
    """

    groups = group_inputs or [("all frames", settings.lights, outputs)]
    if not groups or any(not paths for _, paths, _ in groups):
        raise ValueError("Cannot estimate disk use without input frames")
    scale = settings.drizzle.scale

    details: list[tuple[str, int, int]] = []
    output_volume_bytes: dict[Path, int] = {}
    for label, lights, planned_outputs in groups:
        profiles = []
        for light in lights:
            meta = load_meta(light)
            pattern = (settings.bayer_pattern or meta.bayer_pattern or "").upper()
            cfa = pattern in ("RGGB", "BGGR", "GRBG", "GBRG")
            debayer = settings.debayer == "on" or (settings.debayer == "auto" and cfa)
            channels = 3 if debayer and meta.channels == 1 else meta.channels
            height, width = meta.shape
            profiles.append((height, width, channels, cfa and debayer))
        pixels = max(height * width for height, width, _, _ in profiles)
        channels = max(profile[2] for profile in profiles)
        cfa_drizzle = settings.drizzle.cfa and any(profile[3] for profile in profiles)
        drizzle_channels = 3 if cfa_drizzle else channels
        master_bytes = pixels * channels * 4
        frame_count = scratch_frames if group_inputs is None and scratch_frames is not None else len(lights)
        rejection_masks = pixels * frame_count if settings.drizzle.enabled or settings.mfdeconv is not None else 0
        group_working = frame_count * master_bytes + rejection_masks
        if settings.resume:
            two_pass = settings.local_norm.enabled and frame_count >= 3
            checkpoint_masters = 3 if two_pass else 2  # first/final checkpoint plus atomic staging copy
            checkpoint_arrays = checkpoint_masters * master_bytes + pixels * 6
            if two_pass:
                block = max(1, settings.local_norm.block)
                map_pixels = max(
                    ((frame_height + block - 1) // block) * ((frame_width + block - 1) // block)
                    for frame_height, frame_width, _, _ in profiles
                )
                checkpoint_arrays += frame_count * channels * map_pixels * 2 * 4
            group_working += checkpoint_arrays
        group_output = 0
        for output in planned_outputs:
            stem = output.stem.lower()
            if output.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if stem.endswith("_coverage") or stem.endswith("_rejection"):
                product_bytes = pixels * 4
            elif stem.endswith("_drizzle_weight"):
                product_bytes = pixels * scale * scale * 4
            elif stem.endswith("_drizzle"):
                product_bytes = pixels * scale * scale * drizzle_channels * 4
            else:
                product_bytes = master_bytes
            group_output += product_bytes
            output_dir = output.expanduser().parent
            while not output_dir.exists() and output_dir != output_dir.parent:
                output_dir = output_dir.parent
            output_volume = Path(output_dir.anchor) if output_dir.anchor else output_dir
            output_volume_bytes[output_volume] = output_volume_bytes.get(output_volume, 0) + product_bytes
        details.append((label, group_working, group_output))
    working_bytes = max(group[1] for group in details)
    output_bytes = sum(group[2] for group in details)

    directory = storage_dir.expanduser()
    while not directory.exists() and directory != directory.parent:
        directory = directory.parent
    usage = shutil.disk_usage(directory)
    volume = Path(directory.anchor) if directory.anchor else directory
    required_by_volume = dict(output_volume_bytes)
    required_by_volume[volume] = required_by_volume.get(volume, 0) + working_bytes
    volume_usage = []
    free_by_volume = {volume: int(usage.free)}
    for output_volume in output_volume_bytes:
        if output_volume not in free_by_volume:
            output_dir = next(output.expanduser().parent for _, _, paths in groups for output in paths if output.expanduser().parent.anchor == output_volume.anchor)
            while not output_dir.exists() and output_dir != output_dir.parent:
                output_dir = output_dir.parent
            free_by_volume[output_volume] = int(shutil.disk_usage(output_dir).free)
    for required_volume, required_bytes in required_by_volume.items():
        volume_usage.append((required_volume, required_bytes, free_by_volume[required_volume]))
    min_free = min((free for _, _, free in volume_usage), default=int(usage.free))
    return DiskSpaceEstimate(working_bytes, output_bytes, min_free, volume, tuple(details), tuple(volume_usage))