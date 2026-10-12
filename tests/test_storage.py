from __future__ import annotations

import numpy as np
from types import SimpleNamespace
from pathlib import Path

from gpustacker.drizzle import DrizzleSettings
from gpustacker.io import save_fits
from gpustacker.normalization import LocalNormSettings
from gpustacker.pipeline import PipelineSettings
from gpustacker.storage import estimate_stack_disk_space


def test_estimate_stack_disk_space_includes_scratch_masks_and_outputs(tmp_path):
    light = save_fits(tmp_path / "light.fit", np.zeros((1, 2, 3), dtype=np.float32))
    settings = PipelineSettings(
        lights=[light, light],
        output=tmp_path / "master.fit",
        debayer="off",
        drizzle=DrizzleSettings(enabled=True, scale=2),
    )
    outputs = [
        tmp_path / "master.fit",
        tmp_path / "master_coverage.fit",
        tmp_path / "master_rejection.fit",
        tmp_path / "master_drizzle.fit",
        tmp_path / "master_drizzle_weight.fit",
    ]

    estimate = estimate_stack_disk_space(settings, outputs, tmp_path, scratch_frames=2)

    frame_bytes = 2 * 3 * 4
    assert estimate.working_bytes == 2 * frame_bytes + 2 * (2 * 3)
    assert estimate.output_bytes == frame_bytes + 2 * (2 * 3 * 4) + (2 * 3 * 4 * 4) * 2
    assert estimate.expected_bytes == estimate.working_bytes + estimate.output_bytes
    assert estimate.recommended_bytes >= estimate.expected_bytes
    assert estimate.free_bytes > 0
    assert estimate.volume == tmp_path.__class__(tmp_path.anchor) and estimate.volume.is_dir()


def test_estimate_stack_disk_space_uses_largest_mixed_frame_shape(tmp_path):
    small = save_fits(tmp_path / "small.fit", np.zeros((1, 2, 3), dtype=np.float32))
    large = save_fits(tmp_path / "large.fit", np.zeros((3, 4, 5), dtype=np.float32))
    settings = PipelineSettings(lights=[small, large], output=tmp_path / "master.fit", debayer="off")

    estimate = estimate_stack_disk_space(settings, [settings.output], tmp_path)

    expected_master = 4 * 5 * 3 * 4
    assert estimate.working_bytes == 2 * expected_master
    assert estimate.output_bytes == expected_master


def test_estimate_stack_disk_space_includes_resume_checkpoint_and_local_maps(tmp_path):
    light = save_fits(tmp_path / "light.fit", np.zeros((1, 4, 5), dtype=np.float32))
    settings = PipelineSettings(
        lights=[light, light, light],
        output=tmp_path / "master.fit",
        debayer="off",
        resume=True,
        local_norm=LocalNormSettings(enabled=True, block=2),
    )

    estimate = estimate_stack_disk_space(settings, [settings.output], tmp_path)

    frame_bytes = 4 * 5 * 4
    local_map_bytes = 3 * 1 * (2 * 3) * 2 * 4
    checkpoint_bytes = 3 * frame_bytes + (4 * 5 * 6) + local_map_bytes
    assert estimate.working_bytes == 3 * frame_bytes + checkpoint_bytes


def test_estimate_stack_disk_space_sums_outputs_and_peaks_sequential_batch_scratch(tmp_path):
    small = save_fits(tmp_path / "small.fit", np.zeros((1, 2, 3), dtype=np.float32))
    large = save_fits(tmp_path / "large.fit", np.zeros((3, 4, 5), dtype=np.float32))
    settings = PipelineSettings(lights=[small, large], output=tmp_path / "master.fit", debayer="off")
    first_out = tmp_path / "first.fit"
    second_out = tmp_path / "second.fit"

    estimate = estimate_stack_disk_space(settings, [], tmp_path, group_inputs=[
        ("small group", [small, small], [first_out]),
        ("large group", [large, large, large], [second_out]),
    ])

    small_bytes = 2 * 3 * 4
    large_bytes = 4 * 5 * 3 * 4
    assert estimate.working_bytes == 3 * large_bytes
    assert estimate.output_bytes == small_bytes + large_bytes
    assert [group[0] for group in estimate.group_details] == ["small group", "large group"]


def test_estimate_stack_disk_space_checks_scratch_and_output_volumes(tmp_path, monkeypatch):
    import gpustacker.storage as storage

    light = save_fits(tmp_path / "light.fit", np.zeros((1, 2, 3), dtype=np.float32))
    settings = PipelineSettings(lights=[light, light], output=tmp_path / "unused.fit", debayer="off")
    output = tmp_path.__class__("Q:\\stacks\\master.fit")

    def fake_disk_usage(directory):
        free = 100_000 if Path(directory).anchor == output.anchor else 1_000_000
        return SimpleNamespace(free=free)

    monkeypatch.setattr(storage.shutil, "disk_usage", fake_disk_usage)
    estimate = estimate_stack_disk_space(settings, [output], tmp_path)
    volumes = {volume.anchor: (required, free) for volume, required, free in estimate.volume_usage}

    assert set(volumes) == {tmp_path.anchor, output.anchor}
    assert volumes[output.anchor] == (2 * 3 * 4, 100_000)
    assert volumes[tmp_path.anchor][0] == estimate.working_bytes