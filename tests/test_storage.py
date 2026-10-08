from __future__ import annotations

import numpy as np

from gpustacker.drizzle import DrizzleSettings
from gpustacker.io import save_fits
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