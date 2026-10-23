from __future__ import annotations

import json
import csv
import os
import queue
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

import gpustacker.compare as compare_module
import gpustacker.distributed as distributed_module
import gpustacker.gui as gui_module
from gpustacker.gui import planned_output_files, rank_reference_frames, reference_preview_path, reference_preview_rgb
from gpustacker.cli import build_parser, main, settings_from_args
from gpustacker.checkpoint import CheckpointError, fingerprint_inputs
from gpustacker.detection import Stars
from gpustacker.drizzle import DrizzleSettings
from gpustacker.io import load_frame, load_meta
from gpustacker.mfdeconv import MFDeconvSettings
from gpustacker.normalization import LocalNormSettings
from gpustacker.pipeline import FrameInfo, PipelineSettings, StackingPipeline, blend_mfdeconv, preview_frame_filters, select_mfdeconv_indices
from gpustacker.quality import FilterSettings
from gpustacker.registration import Alignment
from gpustacker.stacking import FrameStore, StackResult, StackSettings


def test_mfdeconv_selects_sharpest_eligible_frames_with_weight_tiebreak():
    def info(index, fwhm, weight, rejected_reason=None):
        return FrameInfo(index, None, None, None, fwhm, 1.0, 1.0, weight=weight, rejected_reason=rejected_reason, store_index=index)

    frames = [
        info(0, 4.5, 10.0),
        info(1, 2.8, 0.2),
        info(2, 3.0, 1.0),
        info(3, 2.8, 0.8),
        info(4, 2.5, 5.0, rejected_reason="filter: cloudy"),
    ]

    assert select_mfdeconv_indices(frames, 3) == [3, 1, 2]


def test_mfdeconv_blend_endpoints_and_fraction():
    master = np.array([[[10.0, 20.0]]], dtype=np.float32)
    deconvolved = np.array([[[30.0, 40.0]]], dtype=np.float32)

    np.testing.assert_array_equal(blend_mfdeconv(master, deconvolved, 0.0), master)
    np.testing.assert_array_equal(blend_mfdeconv(master, deconvolved, 1.0), deconvolved)
    np.testing.assert_allclose(blend_mfdeconv(master, deconvolved, 0.25), [[[15.0, 25.0]]])
    with np.testing.assert_raises(ValueError):
        PipelineSettings(lights=[], output=Path("unused.fit"), mfdeconv_blend=1.2)


def test_distributed_pipeline_preflights_worker_and_routes_stack(monkeypatch, tmp_path):
    settings = PipelineSettings(
        lights=[],
        output=tmp_path / "stack.fit",
        distributed_enabled=True,
        distributed_host="worker-host",
        distributed_token="pipeline-worker-token-with-at-least-32-bytes",
    )
    pipeline = StackingPipeline(settings)
    store = FrameStore(tmp_path / "registered", (1, 4, 5))
    store.add("frame.fit", np.ones(store.shape, dtype=np.float32))
    calls = {}
    expected = StackResult(np.zeros(store.shape, dtype=np.float32), 0, 0, 20, 1)

    def fake_probe(host, port, token, **kwargs):
        calls["probe"] = (host, port, token, kwargs)
        return {"stack_task_execution_enabled": True, "stack_task_max_rows": 2}

    def fake_distributed(store_arg, backend, stack_settings, host, port, token, **kwargs):
        calls["stack"] = (store_arg, host, port, token, kwargs)
        return expected

    monkeypatch.setattr(distributed_module, "probe_worker", fake_probe)
    monkeypatch.setattr(distributed_module, "stack_store_distributed", fake_distributed)

    actual = pipeline._stack_store(store, "Stacking", "global", lambda *_: None, False)

    assert actual is expected
    assert calls["probe"][0:3] == ("worker-host", 47861, settings.distributed_token)
    assert calls["probe"][3]["expected_store"] == store.directory
    assert calls["stack"][1:4] == ("worker-host", 47861, settings.distributed_token)
    assert calls["stack"][4]["pass_name"] == "global"
    assert calls["stack"][4]["rows_per_task"] == 2


def test_distributed_pipeline_settings_require_worker_credentials(tmp_path):
    with pytest.raises(ValueError, match="valid distributed worker host"):
        PipelineSettings(lights=[], output=tmp_path / "stack.fit", distributed_enabled=True)


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI is required")
def test_main_gui_remembers_shared_token_only_as_protected_data(monkeypatch, tmp_path):
    class Variable:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    monkeypatch.setattr(gui_module, "SETTINGS_PATH", tmp_path / "gui_settings.json")
    monkeypatch.setattr(gui_module, "_protect_token", lambda token: f"encrypted:{token[::-1]}")
    monkeypatch.setattr(gui_module, "_unprotect_token", lambda value: value.removeprefix("encrypted:")[::-1])
    token = "main-screen-shared-token-with-at-least-32-bytes"
    remember = Variable(True)
    first = gui_module.GPUStackerApp.__new__(gui_module.GPUStackerApp)
    first.distributed_token_var = Variable(token)
    first.distributed_remember_token_var = remember
    first._setting_vars = lambda: {"distributed_remember_token_var": remember}
    first._append_log = lambda _message: None

    first._save_settings()

    saved = gui_module.SETTINGS_PATH.read_text(encoding="utf-8")
    assert token not in saved
    assert json.loads(saved)["protected_distributed_token"] == f"encrypted:{token[::-1]}"

    second = gui_module.GPUStackerApp.__new__(gui_module.GPUStackerApp)
    second.distributed_token_var = Variable("")
    second.distributed_remember_token_var = Variable(False)
    second._setting_vars = lambda: {"distributed_remember_token_var": second.distributed_remember_token_var}
    second._append_log = lambda _message: None
    second._load_settings()

    assert second.distributed_remember_token_var.get() is True
    assert second.distributed_token_var.get() == token


def test_reference_ranking_uses_scale_and_matching_report(tmp_path):
    short = tmp_path / "short.fit"
    long = tmp_path / "long.fit"
    metrics_path = tmp_path / "stack.frames.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file", "stars", "fwhm_px", "noise", "exposure_s", "focal_length_mm", "bright_star_rate", "excluded"))
        writer.writeheader()
        writer.writerow({"file": short.name, "stars": 100, "fwhm_px": 2.0, "noise": 2 * np.sqrt(180), "exposure_s": 180, "focal_length_mm": 360, "bright_star_rate": 50, "excluded": ""})
        writer.writerow({"file": long.name, "stars": 100, "fwhm_px": 2.5, "noise": 2 * np.sqrt(300), "exposure_s": 300, "focal_length_mm": 600, "bright_star_rate": 50, "excluded": ""})

    metrics = rank_reference_frames([short, long], metrics_path)
    assert metrics is not None
    assert metrics[long.name]["rank"] == 1
    assert metrics[short.name]["rank"] == 2
    assert rank_reference_frames([short], metrics_path) is None


def test_reference_preview_defaults_to_top_rank_without_changing_run_reference(tmp_path):
    best = tmp_path / "best.fit"
    other = tmp_path / "other.fit"
    choices = {best.name: best, other.name: other}
    metrics = {best.name: {"rank": 1}, other.name: {"rank": 2}}
    selected_reference = ""

    preview = reference_preview_path("Automatic (sharpest eligible frame)", choices, metrics)

    assert preview == best
    assert selected_reference == ""


def test_reference_dropdown_reopen_skips_filesystem_and_ranking_work(monkeypatch):
    from gpustacker import gui as gui_module

    app = gui_module.GPUStackerApp.__new__(gui_module.GPUStackerApp)
    app._reference_choices_dirty = False
    monkeypatch.setattr(gui_module, "rank_reference_frames", lambda *_args: pytest.fail("CSV should not be re-read"))
    monkeypatch.setattr(gui_module, "rank_reference_infos", lambda *_args: pytest.fail("frames should not be re-ranked"))

    app._refresh_reference_choices()

    assert app._reference_choices_dirty is False
    app._invalidate_analysis_cache()
    assert app._reference_choices_dirty is True


@pytest.mark.parametrize("suffix", [".fit", ".xisf"])
@pytest.mark.parametrize("corrupt", [False, True], ids=["missing", "corrupt"])
def test_reference_preview_worker_reports_missing_and_corrupt_inputs(tmp_path, suffix, corrupt):
    from gpustacker.gui import GPUStackerApp

    path = tmp_path / f"preview{suffix}"
    if corrupt:
        path.write_bytes(b"not a valid astronomical image")
    pipeline = StackingPipeline(PipelineSettings(
        lights=[path],
        output=tmp_path / "preview-output.fit",
        device="cpu",
        cosmetic="off",
        debayer="off",
    ))
    app = GPUStackerApp.__new__(GPUStackerApp)
    app.events = queue.Queue()
    app._cancel_requested = False

    app._preview_worker(path, pipeline)

    kind, message = app.events.get_nowait()
    assert kind == "reference_preview_error"
    assert message


@pytest.mark.parametrize("method", ["vng", "bilinear"])
def test_reference_preview_uses_selected_cfa_debayer_method(tmp_path, method):
    from astropy.io import fits
    from gpustacker.gui import GPUStackerApp

    path = tmp_path / "cfa.fit"
    header = fits.Header()
    header["BAYERPAT"] = "RGGB"
    fits.PrimaryHDU(data=np.arange(32 * 32, dtype=np.float32).reshape(32, 32), header=header).writeto(path)
    pipeline = StackingPipeline(PipelineSettings(
        lights=[path],
        output=tmp_path / "cfa-output.fit",
        device="cpu",
        cosmetic="off",
        debayer="on",
        debayer_method=method,
        bayer_pattern="RGGB",
    ))
    app = GPUStackerApp.__new__(GPUStackerApp)
    app.events = queue.Queue()
    app._cancel_requested = False

    app._preview_worker(path, pipeline)

    kind, payload = app.events.get_nowait()
    assert kind == "reference_preview"
    preview_path, pixels, meta = payload
    assert preview_path == path
    assert pixels.shape == (32, 32, 3)
    assert meta.channels == 1


def test_reference_dropdown_changes_preview_candidate_not_run_reference(tmp_path):
    from gpustacker.gui import GPUStackerApp

    class Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    app = GPUStackerApp.__new__(GPUStackerApp)
    app.reference_path_var = Value("chosen-for-run.fit")
    app.reference_display_var = Value("preview.fit")
    app._reference_choices = {"preview.fit": tmp_path / "preview.fit"}
    app._preview_candidate_path = None
    app._update_reference_stats = lambda: None

    app._select_reference()

    assert app._preview_candidate_path == tmp_path / "preview.fit"
    assert app.reference_path_var.get() == "chosen-for-run.fit"


def test_use_reference_candidate_explicitly_changes_run_reference(tmp_path):
    from gpustacker.gui import GPUStackerApp

    class Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    candidate = tmp_path / "candidate.fit"
    app = GPUStackerApp.__new__(GPUStackerApp)
    app.reference_display_var = Value(candidate.name)
    app.reference_path_var = Value("")
    app.reference_choice_status_var = Value("")
    app.reference_stats_var = Value("")
    app._reference_choices = {candidate.name: candidate}
    app._reference_metrics = None

    app._use_reference_candidate()

    assert app.reference_path_var.get() == str(candidate)


def test_pipeline_uses_explicitly_selected_reference(light_dir, tmp_path):
    lights = sorted(light_dir.glob("*.fit"))[:2]
    chosen = lights[1]
    pipeline = StackingPipeline(PipelineSettings(lights=lights, output=tmp_path / "reference.fit", reference=chosen))
    stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=20)
    infos = [FrameInfo(index, path, load_meta(path), stars, 2.0, 1.0, 10.0) for index, path in enumerate(lights)]

    assert pipeline.choose_reference(infos) == 1


def test_full_field_registration_expands_canvas_and_shifts_reference_coords(light_dir, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import torch

    import gpustacker.pipeline as pipeline_module
    from gpustacker.quality import StarSet

    lights = sorted(light_dir.glob("*.fit"))[:3]
    metas = {path: load_meta(path) for path in lights}
    reference_stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=20)
    shifted_stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=20)
    distant_stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=20)
    infos = [
        FrameInfo(index, path, metas[path], [reference_stars, shifted_stars, distant_stars][index], 2.0, 1.0, 10.0)
        for index, path in enumerate(lights)
    ]
    height, width = metas[lights[0]].shape
    channels = metas[lights[0]].channels
    data = {path: np.full((channels, height, width), index + 1, dtype=np.float32) for index, path in enumerate(lights)}
    settings = PipelineSettings(
        lights=lights,
        output=tmp_path / "full_field.fit",
        full_field=True,
        device="cpu",
        cosmetic="off",
        debayer="off",
        local_norm=LocalNormSettings(enabled=False),
        filters=FilterSettings(enabled=False),
    )
    pipeline = StackingPipeline(settings)
    pipeline.status = lambda _message: None
    monkeypatch.setattr(pipeline, "_detection_pool", lambda: ThreadPoolExecutor(max_workers=2))
    monkeypatch.setattr(pipeline, "_prefetch", lambda paths, _loader: ((path, data[path], metas[path]) for path in paths))
    monkeypatch.setattr(pipeline, "_calibrate_data", lambda image, *_args, **_kwargs: torch.from_numpy(image))
    monkeypatch.setattr(pipeline_module, "select_photometry_stars", lambda *_args: StarSet(np.array([8], np.float32), np.array([8], np.float32), 2))
    monkeypatch.setattr(pipeline_module, "aperture_flux", lambda *_args: np.ones(12, np.float32))

    def solve(source, _reference, _refine):
        if source is reference_stars:
            return Alignment.identity()
        if source is distant_stars:
            return Alignment(np.array([[1, 0, width * 0.8], [0, 1, 0], [0, 0, 1]], dtype=np.float64), 20, 0.0, 1.0, (width * 0.8, 0.0))
        return Alignment(np.array([[1, 0, -4], [0, 1, 2], [0, 0, 1]], dtype=np.float64), 20, 0.0, 1.0, (-4.0, 2.0))

    monkeypatch.setattr(pipeline_module, "_solve_alignment", solve)
    store = FrameStore(tmp_path / "registered", (channels, height, width))
    pipeline.register(infos, 0, store)

    assert store.shape == (channels, height + 2, width + 4)
    assert pipeline.output_origin == (-4, 0)
    assert len(store) == 2
    assert infos[2].rejected_reason == "only 20% overlaps reference (minimum 25%)"
    reference = np.load(store.paths[0])
    shifted = np.load(store.paths[1])
    np.testing.assert_allclose(reference[:, :height, 4 : width + 4], 1.0)
    np.testing.assert_allclose(shifted[:, 2 : height + 2, :width], 2.0)


def test_reference_preview_ppm_is_bounded_and_handles_nonfinite_data():
    image = np.zeros((3, 1600, 900), dtype=np.float32)
    image[:, 10, 10] = np.nan
    image[0, 20, 20] = 1.0

    pixels = reference_preview_rgb(image)
    height, width = pixels.shape[:2]

    assert max(width, height) <= 760
    assert abs(width / height - 900 / 1600) < 0.01
    assert pixels.dtype == np.uint8
    assert pixels.shape == (height, width, 3)


def test_automatic_reference_compares_bright_star_signal_per_second():
    frames = []
    for index, (exposure, rate, fwhm) in enumerate(((180.0, 95.0, 1.8), (300.0, 100.0, 2.1), (300.0, 105.0, 2.5))):
        frames.append(SimpleNamespace(
            index=index,
            rejected_reason=None,
            fwhm=fwhm,
            noise=1.0,
            meta=SimpleNamespace(exposure=exposure),
            stars=SimpleNamespace(total_detected=100, flux=np.full(30, exposure * rate)),
        ))
    pipeline = SimpleNamespace(settings=SimpleNamespace(reference=None))

    assert StackingPipeline.choose_reference(pipeline, frames) == 0


def test_pipeline_reuses_preflight_analysis(light_dir, tmp_path, monkeypatch):
    settings = PipelineSettings(
        lights=sorted(light_dir.glob("*.fit")),
        output=tmp_path / "cached.fit",
        device="cpu",
        cosmetic="off",
        mfdeconv=None,
        local_norm=LocalNormSettings(enabled=False),
        filters=FilterSettings(enabled=False),
        save_maps=False,
    )
    analyzed = StackingPipeline(settings).analyse()
    settings.analysis_cache = {str(info.path.resolve()): info for info in analyzed}
    pipeline = StackingPipeline(settings)
    monkeypatch.setattr(pipeline, "analyse", lambda: pytest.fail("analysis should be reused"))
    messages = []
    pipeline.status = messages.append

    pipeline.run()

    assert any("Reusing analysis" in message for message in messages)


def test_registered_checkpoint_round_trip_and_settings_validation(light_dir, tmp_path):
    light = sorted(light_dir.glob("*.fit"))[0]
    work_dir = tmp_path / "work"
    settings = PipelineSettings(
        lights=[light],
        output=tmp_path / "master.fit",
        work_dir=work_dir,
        resume=True,
        weighting="psfsw",
    )
    pipeline = StackingPipeline(settings)
    pipeline.output_origin = (-2, 3)
    store = FrameStore(work_dir, (1, 4, 5))
    registered = np.arange(20, dtype=np.float32).reshape(1, 4, 5)
    store.add(light.name, registered)
    alignment = Alignment.identity()
    stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=37)
    info = FrameInfo(
        0, light, load_meta(light), stars, 2.25, 1.5, 40.0,
        alignment=alignment, weight=1.2, transparency=0.95, phot_stars=12,
        store_index=0, bright_star_rate=5.5,
    )
    tilt = {"verdict": "test field"}

    pipeline._save_registered_checkpoint(work_dir, store, [info], 0, tilt, fingerprint_inputs([light]))
    pipeline.output_origin = (0, 0)
    restored_store, restored_infos, reference_index, restored_tilt, stage, global_image, master_result = pipeline._restore_registered_checkpoint(work_dir, [light])

    np.testing.assert_array_equal(np.load(restored_store.paths[0]), registered)
    assert restored_infos[0].stars.total_detected == 37
    assert restored_infos[0].alignment is not None
    assert restored_infos[0].bright_star_rate == 5.5
    assert reference_index == 0
    assert restored_tilt == tilt
    assert pipeline.output_origin == (-2, 3)
    assert stage == "registered_before_stack"
    assert global_image is None
    assert master_result is None

    settings.weighting = "noise"
    with pytest.raises(CheckpointError, match="settings do not match"):
        pipeline._restore_registered_checkpoint(work_dir, [light])


@pytest.mark.parametrize("keep_registered", [False, True])
def test_pipeline_resumes_registered_checkpoint_without_reregistering(light_dir, tmp_path, monkeypatch, keep_registered):
    light = sorted(light_dir.glob("*.fit"))[0]
    work_dir = tmp_path / "work"
    settings = PipelineSettings(
        lights=[light],
        output=tmp_path / "resumed.fit",
        work_dir=work_dir,
        resume=True,
        device="cpu",
        cosmetic="off",
        local_norm=LocalNormSettings(enabled=False),
        filters=FilterSettings(enabled=False),
        save_maps=False,
        plate_solve=False,
    )
    pipeline = StackingPipeline(settings)
    store = FrameStore(work_dir, (1, 4, 5))
    store.add(light.name, np.full((1, 4, 5), 42.0, dtype=np.float32))
    stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=20)
    info = FrameInfo(0, light, load_meta(light), stars, 2.0, 1.0, 10.0, alignment=Alignment.identity(), store_index=0)
    pipeline._save_registered_checkpoint(work_dir, store, [info], 0, {}, fingerprint_inputs([light]))
    settings.keep_registered = keep_registered
    messages = []
    pipeline.status = messages.append
    monkeypatch.setattr(pipeline, "analyse", lambda: pytest.fail("resume must not repeat analysis"))
    monkeypatch.setattr(pipeline, "register", lambda *args: pytest.fail("resume must not repeat registration"))

    result = pipeline.run()

    assert result.output.exists()
    assert any("Resuming from registered_before_stack" in message for message in messages)
    if keep_registered:
        manifest = json.loads((work_dir / "registered_store.json").read_text(encoding="utf-8"))
        assert manifest["frame_count"] == 1
        assert manifest["frames"][0]["path"] == str(light.resolve())
        assert (work_dir / "index.json").exists()
    else:
        assert not work_dir.exists()


def test_pipeline_resumes_after_global_stack_without_repeating_first_pass(light_dir, tmp_path, monkeypatch):
    import gpustacker.pipeline as pipeline_module

    lights = sorted(light_dir.glob("*.fit"))[:3]
    work_dir = tmp_path / "global-checkpoint"
    settings = PipelineSettings(
        lights=lights,
        output=tmp_path / "global-resumed.fit",
        work_dir=work_dir,
        resume=True,
        device="cpu",
        cosmetic="off",
        local_norm=LocalNormSettings(enabled=True, block=8),
        filters=FilterSettings(enabled=False),
        save_maps=False,
        plate_solve=False,
    )
    pipeline = StackingPipeline(settings)
    store = FrameStore(work_dir, (1, 4, 5))
    infos = []
    for index, light in enumerate(lights):
        store.add(light.name, np.full((1, 4, 5), 40.0 + index, dtype=np.float32))
        stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=20)
        infos.append(FrameInfo(index, light, load_meta(light), stars, 2.0, 1.0, 10.0, alignment=Alignment.identity(), store_index=index))
    fingerprints = fingerprint_inputs(lights)
    pipeline._save_registered_checkpoint(work_dir, store, infos, 0, {}, fingerprints)
    expected_global = np.full((1, 4, 5), 41.0, dtype=np.float32)
    pipeline._save_global_stack_checkpoint(work_dir, store, infos, 0, {}, fingerprints, expected_global)

    def fit_local_norm(active_store, *_args):
        for norm in active_store.norms:
            norm.scale_map = np.ones((1, 2, 2), dtype=np.float32)
            norm.offset_map = np.zeros((1, 2, 2), dtype=np.float32)

    pipeline._fit_local_norm = fit_local_norm
    original_stack_store = pipeline_module.stack_store
    stack_calls = []

    def count_stack(*args, **kwargs):
        stack_calls.append(kwargs.get("label"))
        return original_stack_store(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "stack_store", count_stack)
    messages = []
    pipeline.status = messages.append

    result = pipeline.run()

    assert result.output.is_file()
    assert stack_calls == ["Stacking (local norm)"]
    assert any("Resuming after global stacking" in message for message in messages)


def test_pipeline_resumes_after_local_norm_without_refitting(light_dir, tmp_path, monkeypatch):
    import gpustacker.pipeline as pipeline_module

    lights = sorted(light_dir.glob("*.fit"))[:3]
    work_dir = tmp_path / "local-norm-checkpoint"
    pipeline = StackingPipeline(PipelineSettings(
        lights=lights,
        output=tmp_path / "local-norm-resumed.fit",
        work_dir=work_dir,
        resume=True,
        device="cpu",
        cosmetic="off",
        local_norm=LocalNormSettings(enabled=True, block=8),
        filters=FilterSettings(enabled=False),
        save_maps=False,
        plate_solve=False,
    ))
    store = FrameStore(work_dir, (1, 4, 5))
    infos = []
    for index, light in enumerate(lights):
        store.add(light.name, np.full((1, 4, 5), 40.0 + index, dtype=np.float32))
        stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=20)
        infos.append(FrameInfo(index, light, load_meta(light), stars, 2.0, 1.0, 10.0, alignment=Alignment.identity(), store_index=index))
    fingerprints = fingerprint_inputs(lights)
    global_image = np.full((1, 4, 5), 41.0, dtype=np.float32)
    pipeline._save_registered_checkpoint(work_dir, store, infos, 0, {}, fingerprints)
    pipeline._save_global_stack_checkpoint(work_dir, store, infos, 0, {}, fingerprints, global_image)
    for norm in store.norms:
        norm.scale_map = np.ones((1, 2, 2), dtype=np.float32)
        norm.offset_map = np.zeros((1, 2, 2), dtype=np.float32)
    pipeline._save_local_norm_checkpoint(work_dir, store, infos, 0, {}, fingerprints, global_image)
    pipeline._fit_local_norm = lambda *_args: (_ for _ in ()).throw(AssertionError("maps should not be refit"))
    stack_calls = []
    original_stack_store = pipeline_module.stack_store

    def count_stack(*args, **kwargs):
        stack_calls.append(kwargs.get("label"))
        return original_stack_store(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "stack_store", count_stack)

    result = pipeline.run()

    assert result.output.is_file()
    assert stack_calls == ["Stacking (local norm)"]


def test_pipeline_resumes_after_master_without_restacking(light_dir, tmp_path, monkeypatch):
    import gpustacker.pipeline as pipeline_module

    light = sorted(light_dir.glob("*.fit"))[0]
    work_dir = tmp_path / "master-checkpoint"
    settings = PipelineSettings(
        lights=[light],
        output=tmp_path / "master-resumed.fit",
        work_dir=work_dir,
        resume=True,
        device="cpu",
        cosmetic="off",
        local_norm=LocalNormSettings(enabled=False),
        filters=FilterSettings(enabled=False),
        save_maps=False,
        plate_solve=False,
        autocrop=0.5,
        stack=StackSettings(method="none", normalize=False),
    )
    pipeline = StackingPipeline(settings)
    store = FrameStore(work_dir, (1, 4, 5))
    store.add(light.name, np.ones((1, 4, 5), dtype=np.float32))
    stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=20)
    info = FrameInfo(0, light, load_meta(light), stars, 2.0, 1.0, 10.0, alignment=Alignment.identity(), store_index=0)
    fingerprints = fingerprint_inputs([light])
    pipeline._save_registered_checkpoint(work_dir, store, [info], 0, {}, fingerprints)
    expected = np.full((1, 2, 3), 55.0, dtype=np.float32)
    stack = StackResult(
        image=expected,
        rejected_low=1,
        rejected_high=2,
        total_samples=20,
        bands=1,
        coverage=np.ones((2, 3), dtype=np.uint16),
        rejection=np.full((2, 3), 0.15, dtype=np.float32),
        crop_box=(1, 3, 1, 4),
    )
    pipeline._save_master_checkpoint(work_dir, store, [info], 0, {}, fingerprints, stack)
    monkeypatch.setattr(pipeline_module, "stack_store", lambda *_args, **_kwargs: pytest.fail("master checkpoint should skip stacking"))
    messages = []
    pipeline.status = messages.append

    result = pipeline.run()

    actual, _ = load_frame(result.output)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(result.stack.coverage, stack.coverage)
    np.testing.assert_array_equal(result.stack.rejection, stack.rejection)
    assert result.stack.rejected_low == 1
    assert result.stack.rejected_high == 2
    assert any("Resuming after master stacking" in message for message in messages)


def test_cancelled_stack_preserves_checkpoint_for_resume(light_dir, tmp_path, monkeypatch):
    import gpustacker.pipeline as pipeline_module

    light = sorted(light_dir.glob("*.fit"))[0]
    work_dir = tmp_path / "cancel-resume"
    settings = PipelineSettings(
        lights=[light],
        output=tmp_path / "cancel-resume.fit",
        work_dir=work_dir,
        resume=True,
        device="cpu",
        cosmetic="off",
        local_norm=LocalNormSettings(enabled=False),
        filters=FilterSettings(enabled=False),
        save_maps=False,
        plate_solve=False,
        stack=StackSettings(method="none", normalize=False),
    )
    first_pipeline = StackingPipeline(settings)
    meta = load_meta(light)
    shape = (meta.channels, *meta.shape)
    store = FrameStore(work_dir, shape)
    store.add(light.name, np.full(shape, 42.0, dtype=np.float32))
    stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=20)
    info = FrameInfo(0, light, meta, stars, 2.0, 1.0, 10.0, alignment=Alignment.identity(), store_index=0)
    first_pipeline._save_registered_checkpoint(work_dir, store, [info], 0, {}, fingerprint_inputs([light]))

    def cancel_at_stack(*_args, **kwargs):
        first_pipeline.cancel()
        kwargs["cancel_cb"]()

    monkeypatch.setattr(pipeline_module, "stack_store", cancel_at_stack)
    with pytest.raises(RuntimeError, match="Cancelled"):
        first_pipeline.run()
    assert (work_dir / "checkpoint.json").is_file()
    assert (work_dir / "reg_0000.npy").is_file()

    monkeypatch.undo()
    resumed = StackingPipeline(settings).run()

    actual, _ = load_frame(resumed.output)
    np.testing.assert_array_equal(actual, np.full(shape, 42.0, dtype=np.float32))
    assert not work_dir.exists()


@pytest.mark.parametrize("corruption", ["truncated_metadata", "missing_map"])
def test_master_checkpoint_rejects_corrupt_or_missing_payload(light_dir, tmp_path, corruption):
    light = sorted(light_dir.glob("*.fit"))[0]
    work_dir = tmp_path / "corrupt-master-checkpoint"
    pipeline = StackingPipeline(PipelineSettings(
        lights=[light],
        output=tmp_path / "corrupt-master.fit",
        work_dir=work_dir,
        resume=True,
        device="cpu",
        cosmetic="off",
        local_norm=LocalNormSettings(enabled=False),
        plate_solve=False,
    ))
    store = FrameStore(work_dir, (1, 4, 5))
    store.add(light.name, np.ones((1, 4, 5), dtype=np.float32))
    stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=20)
    info = FrameInfo(0, light, load_meta(light), stars, 2.0, 1.0, 10.0, alignment=Alignment.identity(), store_index=0)
    fingerprints = fingerprint_inputs([light])
    pipeline._save_registered_checkpoint(work_dir, store, [info], 0, {}, fingerprints)
    stack = StackResult(
        image=np.ones((1, 4, 5), dtype=np.float32),
        rejected_low=0,
        rejected_high=0,
        total_samples=20,
        bands=1,
        coverage=np.ones((4, 5), dtype=np.uint16),
        rejection=np.zeros((4, 5), dtype=np.float32),
    )
    pipeline._save_master_checkpoint(work_dir, store, [info], 0, {}, fingerprints, stack)
    if corruption == "truncated_metadata":
        (work_dir / "stack_result.json").write_text('{"crop_box":', encoding="utf-8")
    else:
        (work_dir / "master_rejection.npy").unlink()

    with pytest.raises(CheckpointError, match="Registered checkpoint is incomplete or corrupt"):
        pipeline._restore_registered_checkpoint(work_dir, [light])


def test_report_and_csv_counts_reconcile_with_excluded_frames(light_dir, tmp_path):
    lights = sorted(light_dir.glob("*.fit"))[:4]
    pipeline = StackingPipeline(PipelineSettings(lights=lights, output=tmp_path / "counts.fit", device="cpu"))
    meta = load_meta(lights[0])
    stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=12)
    infos = [
        FrameInfo(0, lights[0], meta, stars, 2.0, 1.0, 20.0, alignment=Alignment.identity(), weight=1.0, transparency=1.0, phot_stars=9, store_index=0, bright_star_rate=4.0),
        FrameInfo(1, lights[1], load_meta(lights[1]), stars, 2.5, 1.2, 22.0, rejected_reason="filter: quality", bright_star_rate=3.0),
        FrameInfo(2, lights[2], load_meta(lights[2]), stars, 2.8, 1.3, 24.0, rejected_reason="alignment failed: synthetic fixture"),
        FrameInfo(3, lights[3], load_meta(lights[3]), stars, 3.0, 1.4, 26.0, rejected_reason="only 4 stars"),
    ]
    channels = meta.channels
    stack = StackResult(
        image=np.zeros((channels, 2, 3), dtype=np.float32),
        rejected_low=1,
        rejected_high=2,
        total_samples=channels * 6,
        bands=1,
        coverage=np.ones((2, 3), dtype=np.uint16),
        rejection=np.zeros((2, 3), dtype=np.float32),
    )

    csv_path = pipeline._write_csv(infos)
    report_path = pipeline._write_report(infos, 0, stack, None, {}, 1.0, 1.0)
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    counts = report["frame_counts"]

    assert counts == {"input": 4, "eligible": 1, "stacked": 1, "excluded": 3}
    assert report["stack"]["frames"] == counts["stacked"]
    assert sum(row["stacked"] == "True" for row in rows) == counts["stacked"]
    assert [row["excluded"] for row in rows] == [frame["excluded"] or "" for frame in report["frames"]]
    assert [frame["reason"] for frame in report["excluded"]["frames"]] == [
        "filter: quality", "alignment failed: synthetic fixture", "only 4 stars",
    ]
    assert all(int(row["report_schema_version"]) == report["schema_version"] == 2 for row in rows)
    assert report["metric_definitions"]["rejected_low_high"].endswith("not per-frame counts.")


def test_pipeline_records_map_failure_and_retains_master(light_dir, tmp_path, monkeypatch):
    import gpustacker.pipeline as pipeline_module

    lights = sorted(light_dir.glob("*.fit"))[:2]
    output = tmp_path / "map-failure.fit"
    settings = PipelineSettings(
        lights=lights,
        output=output,
        device="cpu",
        cosmetic="off",
        local_norm=LocalNormSettings(enabled=False),
        filters=FilterSettings(enabled=False),
        save_maps=True,
        plate_solve=False,
        stack=StackSettings(method="none", normalize=False),
    )
    original_save_fits = pipeline_module.save_fits

    def fail_coverage(path, *args, **kwargs):
        if Path(path).stem.endswith("_coverage"):
            raise OSError("simulated coverage write failure")
        return original_save_fits(path, *args, **kwargs)

    monkeypatch.setattr(pipeline_module, "save_fits", fail_coverage)

    result = StackingPipeline(settings).run()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert output.is_file()
    assert (tmp_path / "map-failure_rejection.fit").is_file()
    assert report["products"]["master"]["status"] == "complete"
    assert report["products"]["coverage_map"]["status"] == "failed"
    assert report["products"]["rejection_map"]["status"] == "complete"


def test_pipeline_records_alternate_weighting_failure_and_retains_master(light_dir, tmp_path, monkeypatch):
    import gpustacker.pipeline as pipeline_module

    lights = sorted(light_dir.glob("*.fit"))[:2]
    output = tmp_path / "variant-failure.fit"
    settings = PipelineSettings(
        lights=lights,
        output=output,
        device="cpu",
        cosmetic="off",
        local_norm=LocalNormSettings(enabled=False),
        filters=FilterSettings(enabled=False),
        save_maps=False,
        plate_solve=False,
        weightings=["psfsw", "noise"],
        stack=StackSettings(method="none", normalize=False),
    )
    original_stack_store = pipeline_module.stack_store

    def fail_noise_variant(*args, **kwargs):
        if kwargs.get("label") == "Stacking noise":
            raise OSError("simulated variant stack failure")
        return original_stack_store(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "stack_store", fail_noise_variant)

    result = StackingPipeline(settings).run()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert result.output.is_file()
    assert result.weighting_outputs == {"psfsw": result.output}
    assert report["products"]["weighting_noise"]["status"] == "failed"
    assert "simulated variant stack failure" in report["products"]["weighting_noise"]["error"]


def test_pipeline_stacks_from_retained_registered_store(light_dir, tmp_path, monkeypatch):
    light = sorted(light_dir.glob("*.fit"))[0]
    work_dir = tmp_path / "registered"
    meta = load_meta(light)
    shape = (meta.channels, *meta.shape)
    registered = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 100.0
    producer = StackingPipeline(PipelineSettings(lights=[light], output=tmp_path / "source.fit"))
    store = FrameStore(work_dir, shape)
    store.add(light.name, registered)
    store.norms[0].weight_map = np.ones((4, 4), dtype=np.float32)
    stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=37)
    info = FrameInfo(0, light, meta, stars, 2.0, 1.0, 10.0, alignment=Alignment.identity(), weight=1.0, store_index=0)
    producer._save_registered_store_manifest(work_dir, store, [info], 0, {})

    def unexpected_source_read(*_args, **_kwargs):
        pytest.fail("registered-store stacking must use manifest metadata, not reopen the originals")

    monkeypatch.setattr("gpustacker.pipeline.load_meta", unexpected_source_read)
    settings = PipelineSettings(
        lights=[],
        output=tmp_path / "from_registered.fit",
        registered_dir=work_dir,
        device="cpu",
        local_norm=LocalNormSettings(enabled=False),
        filters=FilterSettings(enabled=False),
        save_maps=False,
        plate_solve=False,
        stack=StackSettings(method="none", normalize=False),
        mfdeconv=MFDeconvSettings(iterations=1),
    )
    pipeline = StackingPipeline(settings)
    monkeypatch.setattr(pipeline, "analyse", lambda: pytest.fail("registered-store stacking must not analyze"))
    monkeypatch.setattr(pipeline, "register", lambda *args: pytest.fail("registered-store stacking must not register"))
    mf_output = tmp_path / "from_registered_mfdeconv.fit"
    monkeypatch.setattr(pipeline, "_run_mfdeconv", lambda *_args: (mf_output, None))

    result = pipeline.run()

    assert result.output.exists()
    assert result.mfdeconv_output == mf_output
    assert result.frames[0].stars.total_detected == 37
    np.testing.assert_allclose(result.stack.image, registered)
    assert (work_dir / "index.json").exists()
    assert (work_dir / "reg_0000.npy").exists()


def test_retained_registered_store_round_trips_and_cleans_external_drive(light_dir, tmp_path):
    from gpustacker.stacking import cleanup_registered_store

    light = sorted(light_dir.glob("*.fit"))[0]
    work_dir = tmp_path / "registered"
    external_dir = tmp_path / f".gpustacker_registered_{'a' * 32}"
    external_dir.mkdir()
    note = external_dir / "keep.txt"
    note.write_text("unowned", encoding="utf-8")
    meta = load_meta(light)
    shape = (meta.channels, *meta.shape)
    store = FrameStore(work_dir, shape)
    store.storage_dirs = [external_dir]
    store._remaining_bytes = [10**9]
    frame = np.ones(shape, dtype=np.float32)
    store.add(light.name, frame)
    stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=10)
    info = FrameInfo(0, light, meta, stars, 2.0, 1.0, 10.0, alignment=Alignment.identity(), store_index=0)
    StackingPipeline(PipelineSettings(lights=[light], output=tmp_path / "source.fit"))._save_registered_store_manifest(work_dir, store, [info], 0, {})

    restored, _, _, _, _ = StackingPipeline(PipelineSettings(
        lights=[], output=tmp_path / "restored.fit", registered_dir=work_dir,
    ))._restore_registered_store(work_dir)
    assert restored.paths == store.paths
    assert json.loads((work_dir / "registered_store.json").read_text(encoding="utf-8"))["storage_dirs"] == [str(external_dir.resolve())]

    removed_count, _ = cleanup_registered_store(work_dir)

    assert removed_count >= 3
    assert not store.paths[0].exists()
    assert note.read_text(encoding="utf-8") == "unowned"


def test_retained_registered_store_matches_ordinary_stack(light_dir, tmp_path):
    lights = sorted(light_dir.glob("*.fit"))[:2]
    work_dir = tmp_path / "retained-equivalence"
    ordinary_settings = PipelineSettings(
        lights=lights,
        output=tmp_path / "ordinary.fit",
        work_dir=work_dir,
        keep_registered=True,
        device="cpu",
        cosmetic="off",
        local_norm=LocalNormSettings(enabled=False),
        filters=FilterSettings(enabled=False),
        save_maps=False,
        plate_solve=False,
        stack=StackSettings(method="none", normalize=False),
    )
    ordinary = StackingPipeline(ordinary_settings).run()
    retained = StackingPipeline(PipelineSettings(
        lights=[],
        output=tmp_path / "retained.fit",
        registered_dir=work_dir,
        device="cpu",
        cosmetic="off",
        local_norm=LocalNormSettings(enabled=False),
        filters=FilterSettings(enabled=False),
        save_maps=False,
        plate_solve=False,
        stack=StackSettings(method="none", normalize=False),
    )).run()

    np.testing.assert_allclose(retained.stack.image, ordinary.stack.image, rtol=0, atol=0)


def test_pipeline_rejects_inconsistent_registered_store_without_mutation(light_dir, tmp_path):
    light = sorted(light_dir.glob("*.fit"))[0]
    work_dir = tmp_path / "registered"
    meta = load_meta(light)
    shape = (meta.channels, *meta.shape)
    store = FrameStore(work_dir, shape)
    store.add(light.name, np.ones(shape, dtype=np.float32))
    stars = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=10)
    info = FrameInfo(0, light, meta, stars, 2.0, 1.0, 10.0, alignment=Alignment.identity(), store_index=0)
    StackingPipeline(PipelineSettings(lights=[light], output=tmp_path / "source.fit"))._save_registered_store_manifest(work_dir, store, [info], 0, {})
    payload_path = store.paths[0]
    original_payload = payload_path.read_bytes()
    manifest_path = work_dir / "registered_store.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_count"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    settings = PipelineSettings(lights=[], output=tmp_path / "rejected.fit", registered_dir=work_dir, device="cpu")
    with pytest.raises(CheckpointError, match="does not match its index"):
        StackingPipeline(settings).run()

    assert payload_path.read_bytes() == original_payload
    assert not settings.output.exists()


def test_mfdeconv_blend_cli_option():
    args = build_parser().parse_args(["stack", "light.fit", "-o", "out.fit", "--mf-blend", "0.25"])
    assert args.mf_blend == 0.25


def test_full_field_cli_option(light_dir, tmp_path):
    light = sorted(light_dir.glob("*.fit"))[0]
    args = build_parser().parse_args(["stack", str(light), "-o", str(tmp_path / "out.fit"), "--full-field"])
    assert settings_from_args(args).full_field


def test_resume_cli_option(light_dir, tmp_path):
    light = sorted(light_dir.glob("*.fit"))[0]
    args = build_parser().parse_args(["stack", str(light), "-o", str(tmp_path / "out.fit"), "--resume"])
    assert settings_from_args(args).resume


def test_registered_store_cli_option(tmp_path):
    store_dir = tmp_path / "registered"
    args = build_parser().parse_args(["stack", "-o", str(tmp_path / "out.fit"), "--registered-dir", str(store_dir)])

    assert settings_from_args(args).registered_dir == store_dir
    assert settings_from_args(args).lights == []


def test_registered_drive_cli_option(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setattr("gpustacker.cli._collect_lights", lambda _inputs: [tmp_path / "light.fit"])
    args = build_parser().parse_args([
        "stack", "light.fit", "-o", "out.fit", "--registered-drive", str(first), "--registered-drive", str(second),
    ])

    assert settings_from_args(args).registered_storage_dirs == [first, second]


def test_drizzle_kernel_and_min_weight_cli_options():
    args = build_parser().parse_args([
        "stack", "light.fit", "-o", "out.fit", "--drizzle", "2",
        "--drizzle-kernel", "gaussian", "--drizzle-min-weight", "0.1",
    ])
    assert args.drizzle_kernel == "gaussian"
    assert args.drizzle_min_weight == 0.1


def test_planned_output_files_match_enabled_products(tmp_path):
    output = tmp_path / "master.fit"
    settings = PipelineSettings(
        lights=[],
        output=output,
        weightings=["psfsw", "noise", "psfsw+field"],
        drizzle=DrizzleSettings(enabled=True),
        mfdeconv=MFDeconvSettings(),
    )

    assert [path.name for path in planned_output_files(output, settings)] == [
        "master_weight-psfsw.fit",
        "master_weight-psfsw_coverage.fit",
        "master_weight-psfsw_rejection.fit",
        "master_weight-noise.fit",
        "master_weight-psfsw-field.fit",
        "master_weight-psfsw_drizzle.fit",
        "master_weight-psfsw_drizzle_weight.fit",
        "master_weight-psfsw_mfdeconv.fit",
        "master_weight-psfsw.frames.csv",
        "master_weight-psfsw.report.json",
    ]


def test_no_filters_cli_option():
    args = build_parser().parse_args(["stack", "light.fit", "-o", "out.fit", "--no-filters"])

    assert args.no_filters


def test_disabled_filters_do_not_exclude_frames(tmp_path):
    settings = PipelineSettings(
        lights=[],
        output=tmp_path / "unused.fit",
        filters=FilterSettings(enabled=False),
        device="cpu",
    )
    messages = []
    pipeline = StackingPipeline(settings, messages.append)
    infos = [
        FrameInfo(index, Path(f"frame-{index}.fit"), None, None, fwhm=5.0, noise=1.0, background=100.0, transparency=value)
        for index, value in enumerate((1.0, 0.1, 1.0))
    ]

    messages.clear()
    pipeline.apply_prefilters(infos)
    pipeline.apply_transparency_filter(infos, None)

    assert all(info.rejected_reason is None for info in infos)
    assert [info.transparency for info in infos] == [1.0, 0.1, 1.0]
    assert messages == []


def test_newtonian_and_additional_weighting_cli_options():
    args = build_parser().parse_args([
        "stack", "light.fit", "-o", "out.fit", "--weighting", "psfsw+field", "--also-weighting", "noise", "--also-weighting", "none",
    ])
    assert args.weighting == "psfsw+field"
    assert args.also_weighting == ["noise", "none"]


def test_compare_many_uses_one_reference_for_every_candidate(monkeypatch):
    candidates = [Path("one.fit"), Path("two.fit"), Path("three.fit")]
    reference = Path("reference.fit")
    calls = []

    def fake_compare(candidate, ref, status):
        calls.append((candidate, ref))
        return candidate

    monkeypatch.setattr(compare_module, "compare_stacks", fake_compare)
    results = compare_module.compare_many(candidates, reference)
    assert results == candidates
    assert calls == [(candidate, reference) for candidate in candidates]


def test_compare_cli_accepts_additional_masters():
    args = build_parser().parse_args([
        "compare", "one.fit", "reference.fit", "--also", "two.fit", "--also", "three.fit",
    ])
    assert args.gpu == Path("one.fit")
    assert args.reference == Path("reference.fit")
    assert args.also == [Path("two.fit"), Path("three.fit")]


def test_multi_compare_report_contains_each_candidate(tmp_path):
    def result(name):
        return compare_module.CompareResult(
            gpu=name,
            reference="reference.fit",
            alignment_rms_px=0.1,
            common_fraction=0.95,
            stars_gpu=100,
            stars_ref=105,
            stars_matched=98,
            fwhm_gpu=2.8,
            fwhm_ref=2.9,
            faint_peak_snr_gpu=7.0,
            faint_peak_snr_ref=6.8,
            peak_snr_ratio=1.02,
            peak_snr_ratio_faint=1.01,
            star_to_nebula_flux_ratio=0.99,
        )

    path = tmp_path / "compare.json"
    compare_module.write_report([result("one.fit"), result("two.fit")], path)
    data = json.loads(path.read_text())
    assert [item["gpu"] for item in data["comparisons"]] == ["one.fit", "two.fit"]


def test_compare_regional_medians_and_rows():
    xs = np.array([5, 6, 7, 8, 9, 55, 56, 57, 58, 59], dtype=np.float32)
    ys = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9], dtype=np.float32)
    values = np.arange(1, 11, dtype=np.float32)
    medians, counts = compare_module._regional_medians(xs, ys, values, (100, 100))

    assert medians == {"top left": 3.0, "top centre-right": 8.0}
    assert counts == {"top left": 5, "top centre-right": 5}

    result = compare_module.CompareResult(
        gpu="master.fit",
        reference="reference.fit",
        alignment_rms_px=0.1,
        common_fraction=1.0,
        stars_gpu=20,
        stars_ref=20,
        stars_matched=20,
        fwhm_gpu=2.5,
        fwhm_ref=2.6,
        faint_peak_snr_gpu=7.0,
        faint_peak_snr_ref=7.0,
        peak_snr_ratio=1.0,
        peak_snr_ratio_faint=1.0,
        star_to_nebula_flux_ratio=1.0,
        regional_fwhm_gpu={"top left": 2.4},
        regional_fwhm_ref={"top left": 2.8},
        regional_star_snr_gpu={"top left": 10.0},
        regional_star_snr_ref={"top left": 8.0},
        regional_matched_stars={"top left": 6},
    )
    from gpustacker.compare_gui import _rows

    labels = [row[0] for row in _rows(result)]
    assert "FWHM top left [px]" in labels
    assert "Matched-star SNR top left (n=6)" in labels


def _filter_test_frame(index, exposure=60.0, stars=100, background=100.0, fwhm=2.0, flux_ratio=1.0):
    from gpustacker.io import FrameMeta

    meta = FrameMeta(Path(f"filter_{index}.fit"), (4, 4), 1, exposure=exposure)
    detected = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=stars)
    return FrameInfo(index, meta.path, meta, detected, fwhm=fwhm, noise=1.0, background=background, flux_ratio=flux_ratio)


def test_cli_filter_preset_allows_individual_overrides(light_dir, tmp_path):
    args = build_parser().parse_args([
        "stack", str(light_dir), "-o", str(tmp_path / "unused.fit"),
        "--filter-preset", "selective", "--filter-fwhm", "1.6", "--filter-psfsw", "0.15",
    ])
    settings = settings_from_args(args)
    assert settings.filters.stars_min_ratio == 0.8
    assert settings.filters.fwhm_max_ratio == 1.6
    assert settings.filters.psfsw_min_ratio == 0.15
    defaults = settings_from_args(build_parser().parse_args(["stack", str(light_dir), "-o", str(tmp_path / "default.fit")]))
    assert defaults.filters == FilterSettings()


def test_gui_filter_preset_keeps_custom_fields_and_advanced_cutoff_editable():
    import tkinter as tk

    app = gui_module.GPUStackerApp.__new__(gui_module.GPUStackerApp)
    root = tk.Tcl()
    app._applying_filter_preset = False
    app.filter_preset_var = tk.StringVar(master=root, value="Selective")
    app._filter_preset_vars = {name: tk.DoubleVar(master=root, value=0) for name in ("transparency_min", "stars_min_ratio", "background_max_ratio", "fwhm_max_ratio")}
    app.f_psfsw = tk.DoubleVar(master=root, value=0.15)
    app._apply_filter_preset()
    assert app._filter_preset_vars["stars_min_ratio"].get() == 0.8
    assert app.f_psfsw.get() == 0.15
    app._filter_preset_vars["fwhm_max_ratio"].set(1.6)
    app._filter_fields_edited()
    assert app.filter_preset_var.get() == "Selective (modified)"
    app._filter_preset_vars["fwhm_max_ratio"].set(1.2)
    app._filter_fields_edited()
    assert app.filter_preset_var.get() == "Selective"


def test_gui_filter_preview_counts_exposure_pending_metrics_and_hard_failures(monkeypatch):
    import tkinter as tk
    from dataclasses import replace

    monkeypatch.setattr(gui_module.GPUStackerApp, "_load_settings", lambda self: None)
    root = tk.Tk()
    root.withdraw()
    try:
        app = gui_module.GPUStackerApp(root)
        app.filter_preset_var.set("Balanced")
        app._apply_filter_preset()
        app.f_psfsw.set(0.05)
        app.filters_enabled_var.set(True)
        infos = [_filter_test_frame(index) for index in range(6)]
        infos[4].stars.total_detected = 20
        app._analysis_cache = {str(info.path.resolve()): info for info in infos}
        app._analysis_cache_signature = "preview-test"
        app._show_filter_preview()
        root.update_idletasks()
        assert "Keep 5/6 frames; reject 1" in app.filter_summary_var.get()
        assert "83%" in app.filter_summary_var.get()
        assert "pending registration" in app.filter_pending_var.get()
        assert app._filter_tree.item("0", "values")[6:8] == ("Pending", "Pending")
        app._filter_measurements = {str(info.path.resolve()): replace(info, transparency=1.0, psfsw=1.0) for info in infos}
        app._filter_measurements_signature = "preview-test"
        app._filter_measurements[str(infos[0].path.resolve())].transparency = 0.2
        app._filter_measurements[str(infos[0].path.resolve())].psfsw = 0.01
        app._filter_measurements[str(infos[1].path.resolve())].rejected_reason = "registration failed"
        with monkeypatch.context() as changes:
            changes.setattr(Path, "resolve", lambda *_args, **_kwargs: pytest.fail("Live filter edits must not resolve paths"))
            app._refresh_filter_preview()
        assert "Keep 3/6 frames; reject 3" in app.filter_summary_var.get()
        assert "50%" in app.filter_summary_var.get()
        assert "transparency" in app._filter_preview_infos[0].rejected_reason
        assert "PSFSW" in app._filter_preview_infos[0].rejected_reason
        assert all(info.rejected_reason is None for info in infos)
        app._sort_filter_preview("psfsw")
        assert app._filter_tree.get_children()[0] == "0"
        app.filters_enabled_var.set(False)
        assert "Keep 5/6 frames; reject 1" in app.filter_summary_var.get()
        app.f_stars.set("")
        assert "valid non-negative" in app.filter_summary_var.get()
        assert not app._filter_tree.get_children()
        app.f_stars.set(0.6)
        app._filter_measurements_signature = "different-inputs"
        app._refresh_filter_preview()
        assert "Keep 6/6 frames; reject 0" in app.filter_summary_var.get()
        app._filter_preview_window.geometry("850x440")
        root.update_idletasks()
        assert len(app._filter_tree.get_children()) == 6
        assert app._filter_tree.winfo_height() > 100
    finally:
        root.destroy()


def test_psfsw_cutoff_cannot_be_silently_ignored_for_retained_store(tmp_path):
    settings = PipelineSettings(lights=[], output=tmp_path / "unused.fit", device="cpu", registered_dir=tmp_path / "retained", filters=FilterSettings(psfsw_min_ratio=0.1))
    with pytest.raises(ValueError, match="PSFSW filtering requires the original light frames"):
        StackingPipeline(settings).run()


def test_filter_preview_is_non_mutating_and_uses_fixed_baselines(tmp_path):
    infos = [_filter_test_frame(index) for index in range(6)]
    infos[4].stars.total_detected = 20
    infos[4].background = 1000
    infos[4].fwhm = 5
    infos[5].rejected_reason = "too few stars"
    settings = FilterSettings()
    preview, baseline = preview_frame_filters(infos, settings)
    assert baseline["candidate_frames"] == 5
    assert all(info.rejected_reason is None for info in infos[:5])
    assert all(token in preview[4].rejected_reason for token in ("stars", "sky", "FWHM"))
    assert preview[5].rejected_reason == "too few stars"
    relaxed, other_baseline = preview_frame_filters(preview, FilterSettings(stars_min_ratio=0, background_max_ratio=0, fwhm_max_ratio=0))
    assert relaxed[4].rejected_reason is None
    assert other_baseline == baseline
    pipeline = StackingPipeline(PipelineSettings(lights=[], output=tmp_path / "unused.fit", device="cpu"))
    pipeline.apply_prefilters(infos)
    assert [info.rejected_reason for info in infos] == [info.rejected_reason for info in preview]


@pytest.mark.parametrize("enabled,threshold,dropped", [(True, 0.05, [3, 7]), (False, 0.05, []), (True, 0.0, [])])
def test_optional_psfsw_filter_is_exposure_aware_and_preserves_store_indices(tmp_path, enabled, threshold, dropped):
    settings = PipelineSettings(lights=[], output=tmp_path / "unused.fit", device="cpu", filters=FilterSettings(enabled=enabled, psfsw_min_ratio=threshold))
    pipeline = StackingPipeline(settings)
    store = FrameStore(tmp_path / "registered", (1, 4, 4))
    infos = []
    for index, (exposure, signal) in enumerate([(60, 1), (60, 1), (60, 1), (60, 0.1), (20, 1 / 3), (20, 1 / 3), (20, 1 / 3), (20, 0.01), (20, np.nan), (30, 0.01)]):
        info = _filter_test_frame(index, exposure=exposure, flux_ratio=signal)
        info.store_index = index
        store.add(info.path.name, np.ones(store.shape, dtype=np.float32))
        infos.append(info)
    pipeline.measure_psfsw(infos)
    pipeline.apply_psfsw_filter(infos, store)
    assert [info.index for info in infos if info.rejected_reason] == dropped
    accepted = [info for info in infos if info.rejected_reason is None]
    assert [info.store_index for info in accepted] == list(range(len(accepted)))
    assert [info.path.name for info in accepted] == store.names
    assert all(info.weight == 1.0 for info in accepted)
    assert np.isnan(infos[8].psfsw) and np.isnan(infos[9].psfsw)
    assert pipeline._serialize_frame_info(infos[0])["psfsw"] == pytest.approx(1.0)
    assert pipeline._frame_rows(infos)[0]["psfsw"] == pytest.approx(1.0)


def test_prefilters_use_per_exposure_medians_for_stars_and_sky(tmp_path):
    from gpustacker.io import FrameMeta

    pipeline = StackingPipeline(PipelineSettings(lights=[], output=tmp_path / "unused.fit", device="cpu"), lambda _: None)

    def info(index: int, exposure: float, stars: int, background: float) -> FrameInfo:
        meta = FrameMeta(Path(f"f{index}.fit"), (10, 10), 1, exposure=exposure)
        detected = Stars(*(np.empty(0, dtype=np.float32) for _ in range(6)), total_detected=stars)
        return FrameInfo(index, meta.path, meta, detected, fwhm=2.0, noise=1.0, background=background)

    # four 60 s subs and four 20 s subs; the short subs have a third of the stars and sky
    infos = [info(i, 60.0, 1500 + i, 1500.0) for i in range(4)] + [info(4 + i, 20.0, 500 + i, 500.0) for i in range(4)]
    infos.append(info(8, 20.0, 150, 500.0))  # genuinely poor short sub

    pipeline.apply_prefilters(infos)

    assert all(i.rejected_reason is None for i in infos[:8])
    assert infos[8].rejected_reason and "stars 150" in infos[8].rejected_reason
    per_exposure = pipeline.filter_baselines["prefilters"]["per_exposure"]
    assert set(per_exposure) == {"60s", "20s"}
    assert per_exposure["20s"]["frames"] == 5


def test_mixed_exposures_are_kept_and_weighted_by_signal(tmp_path, scene):
    """20 s subs beside 60 s subs: the transparency filter must not drop them, and their weights must reflect signal."""

    rng = np.random.default_rng(7)
    base, _ = scene
    folder = tmp_path / "mixed"
    folder.mkdir()
    shifts = [(0, 0), (3.2, -1.7), (-2.4, 2.9), (5.1, 4.4), (-4.6, -3.3), (1.3, 6.0), (2.2, -4.1), (-1.5, 1.1)]
    exposures = [60.0] * 4 + [20.0] * 4
    from astropy.io import fits
    from conftest import shifted_noisy

    for i, ((dx, dy), exposure) in enumerate(zip(shifts, exposures)):
        frame = shifted_noisy(base * (exposure / 60.0), dx, dy, rng)
        hdr = fits.Header()
        hdr["EXPTIME"] = exposure
        hdr["EGAIN"] = 1.0
        hdr["FILTER"] = "L"
        fits.PrimaryHDU(data=frame, header=hdr).writeto(folder / f"light_{i:02d}.fit")
    lights = sorted(folder.glob("*.fit"))
    out = tmp_path / "mixed.fit"
    log: list[str] = []
    result = StackingPipeline(PipelineSettings(lights=lights, output=out, device="cpu", local_norm=LocalNormSettings(enabled=False)), log.append).run()

    report = json.loads(result.report_path.read_text())
    assert report["frame_counts"]["stacked"] == 8, report["excluded"]
    assert report["timings_s"]["register"] > 0 and "stack" in report["timings_s"]
    rows = {row["file"]: row for row in report["frames"]}
    long_rows = [rows[f"light_{i:02d}.fit"] for i in range(4)]
    short_rows = [rows[f"light_{i:02d}.fit"] for i in range(4, 8)]
    for row in long_rows + short_rows:
        assert 0.85 < row["transparency"] < 1.15, row
    assert report["reference"].endswith(("light_00.fit", "light_01.fit", "light_02.fit", "light_03.fit"))
    assert all(0.8 < row["flux_ratio"] < 1.25 for row in long_rows), long_rows
    assert all(0.25 < row["flux_ratio"] < 0.42 for row in short_rows), short_rows
    assert max(row["weight"] for row in short_rows) < min(row["weight"] for row in long_rows)
    stacked, meta = load_frame(out)
    assert meta.header["EXPTOTAL"] == 320.0
    assert any(line.startswith("Stage timings:") for line in log)


def test_pipeline_end_to_end(light_dir, tmp_path, backend, scene):
    lights = sorted(light_dir.glob("*.fit"))
    out = tmp_path / "stack.fit"
    settings = PipelineSettings(lights=lights, output=out, stack=StackSettings(method="winsorized"), device="cpu" if not backend.is_cuda else "auto", mfdeconv=MFDeconvSettings(iterations=5), mfdeconv_frames=4)
    log: list[str] = []
    result = StackingPipeline(settings, log.append).run()

    assert out.exists() and result.mfdeconv_output and result.mfdeconv_output.exists()
    assert result.weighting_outputs == {"psfsw": out}
    stacked, meta = load_frame(out)
    assert stacked.shape == (1, 256, 320)
    assert meta.header["NFRAMES"] == 6
    assert meta.header["EXPTOTAL"] == 360.0
    # the streak in frame 2 must be rejected: compare against the clean scene
    clean = scene[0]
    inner = (slice(40, -40), slice(40, -40))
    err = np.abs(stacked[0][inner] - clean[inner])
    assert np.median(err) < 6.0
    assert np.median(err[60:64]) < 25.0  # rows carrying the streak (after ~2.9px shift) stay sane
    assert result.stack.rejected_high > 0
    report = json.loads(result.report_path.read_text())
    assert len(report["frames"]) == 6 and report["mfdeconv"]["iterations"] >= 1
    assert report["settings"]["weighting"] == "psfsw" and report["stack"]["effective_frames"] > 3
    assert report["schema_version"] == 2
    assert set(report["filter_baselines"]) == {"prefilters", "transparency"}
    assert report["filter_baselines"]["prefilters"]["candidate_frames"] == 6
    counts = report["frame_counts"]
    assert counts == {"input": 6, "eligible": 6, "stacked": 6, "excluded": 0}
    assert report["stack"]["frames"] == counts["stacked"]
    assert report["per_frame_rejection_attribution"]["available"] is False
    assert sum(bool(frame["stacked"]) for frame in report["frames"]) == counts["stacked"]
    assert all(f["transparency"] is not None for f in report["frames"] if f["excluded"] is None)
    assert result.csv_path.exists() and result.csv_path.read_text().count("\n") == 7
    assert (out.parent / "stack_coverage.fit").exists() and (out.parent / "stack_rejection.fit").exists()
    cov, _ = load_frame(out.parent / "stack_coverage.fit")
    assert cov.max() == 6 and cov.min() < 6  # dithered borders are shallower
    assert meta.header["LOCNORM"] is True
    assert not (out.parent / "_gpustacker_work").exists()


def test_pipeline_selected_weightings_write_only_selected_masters(light_dir, tmp_path, backend):
    lights = sorted(light_dir.glob("*.fit"))
    output = tmp_path / "variants.fit"
    settings = PipelineSettings(
        lights=lights,
        output=output,
        local_norm=LocalNormSettings(enabled=False),
        weighting="psfsw",
        weightings=["psfsw", "noise", "psfsw+field"],
        device="cpu" if not backend.is_cuda else "auto",
    )
    result = StackingPipeline(settings, lambda _: None).run()

    assert set(result.weighting_outputs) == {"psfsw", "noise", "psfsw+field"}
    assert result.weighting_outputs["psfsw"] == output.with_name("variants_weight-psfsw.fit")
    for mode, path in result.weighting_outputs.items():
        assert path.exists()
        _, meta = load_frame(path)
        assert meta.header["WEIGHTMD"] == mode
    report = json.loads(result.report_path.read_text())
    assert set(report["weighting_outputs"]) == {"psfsw", "noise", "psfsw+field"}


def test_optional_drizzle_failure_keeps_master_and_records_failure(light_dir, tmp_path, monkeypatch):
    lights = sorted(light_dir.glob("*.fit"))
    output = tmp_path / "drizzle-failure.fit"
    settings = PipelineSettings(lights=lights, output=output, device="cpu", drizzle=DrizzleSettings(enabled=True))

    def fail_drizzle(*_args, **_kwargs):
        raise OSError("simulated drizzle failure")

    monkeypatch.setattr(StackingPipeline, "_run_drizzle", fail_drizzle)
    result = StackingPipeline(settings, lambda _: None).run()

    assert output.is_file()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["products"]["master"]["status"] == "complete"
    assert report["products"]["drizzle"]["status"] == "failed"
    assert report["products"]["drizzle"]["error"] == "OSError: simulated drizzle failure"
    assert "Rerun" in report["products"]["drizzle"]["retry"]


def test_drizzle_weight_failure_keeps_and_reports_drizzle_image(light_dir, tmp_path, monkeypatch):
    import gpustacker.pipeline as pipeline_module

    lights = sorted(light_dir.glob("*.fit"))[:2]
    output = tmp_path / "drizzle-weight-failure.fit"
    settings = PipelineSettings(
        lights=lights,
        output=output,
        device="cpu",
        cosmetic="off",
        local_norm=LocalNormSettings(enabled=False),
        filters=FilterSettings(enabled=False),
        save_maps=False,
        plate_solve=False,
        drizzle=DrizzleSettings(enabled=True),
        stack=StackSettings(method="none", normalize=False),
    )
    original_save_fits = pipeline_module.save_fits

    def fail_weight_map(path, *args, **kwargs):
        if Path(path).stem.endswith("_drizzle_weight"):
            raise OSError("simulated drizzle-weight write failure")
        return original_save_fits(path, *args, **kwargs)

    monkeypatch.setattr(pipeline_module, "save_fits", fail_weight_map)

    result = StackingPipeline(settings).run()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    drizzle_path = output.with_name("drizzle-weight-failure_drizzle.fit")

    assert drizzle_path.is_file()
    assert report["products"]["drizzle"]["status"] == "complete"
    assert report["products"]["drizzle_weight"]["status"] == "failed"


@pytest.mark.parametrize(("refit", "expected_fits"), [(False, 1), (True, 2)])
def test_pipeline_variant_local_norm_policy(light_dir, tmp_path, backend, refit, expected_fits):
    lights = sorted(light_dir.glob("*.fit"))
    settings = PipelineSettings(
        lights=lights,
        output=tmp_path / f"variants-{refit}.fit",
        local_norm=LocalNormSettings(enabled=True, block=32),
        weighting="psfsw",
        weightings=["psfsw", "noise"],
        refit_local_norm_per_weighting=refit,
        device="cpu" if not backend.is_cuda else "auto",
    )
    pipeline = StackingPipeline(settings, lambda _: None)
    original_fit_local_norm = pipeline._fit_local_norm
    fit_count = 0

    def count_fit_local_norm(*args, **kwargs):
        nonlocal fit_count
        fit_count += 1
        return original_fit_local_norm(*args, **kwargs)

    pipeline._fit_local_norm = count_fit_local_norm
    pipeline.run()

    assert fit_count == expected_fits


def test_pipeline_autocrop_full_depth(light_dir, tmp_path):
    lights = sorted(light_dir.glob("*.fit"))
    out = tmp_path / "crop.fit"
    settings = PipelineSettings(lights=lights, output=out, device="cpu", autocrop=1.0, local_norm=LocalNormSettings(enabled=False), save_maps=True)
    result = StackingPipeline(settings).run()
    stacked, meta = load_frame(out)
    assert result.stack.crop_box is not None
    y0, y1, x0, x1 = result.stack.crop_box
    assert stacked.shape == (1, y1 - y0, x1 - x0) and stacked.shape[1] < 256 and stacked.shape[2] < 320
    assert meta.header["CROPX0"] == x0 and meta.header["CROPY0"] == y0
    cov, _ = load_frame(out.parent / "crop_coverage.fit")
    assert cov.min() == 6  # every pixel of the cropped output is full depth


def test_filters_exclude_cloudy_frame(light_dir, tmp_path, scene):
    from astropy.io import fits
    from conftest import shifted_noisy

    # add a frame with 40% transparency (dim stars, smoother sky): noise weighting would love it
    base, _ = scene
    cloudy = shifted_noisy(base * 0.4 + 60.0, 1.0, 1.0, np.random.default_rng(9), read_noise=2.0)
    fits.PrimaryHDU(data=cloudy).writeto(light_dir / "light_99_cloudy.fit")
    lights = sorted(light_dir.glob("*.fit"))
    out = tmp_path / "f.fit"
    log: list[str] = []
    result = StackingPipeline(PipelineSettings(lights=lights, output=out, device="cpu", local_norm=LocalNormSettings(enabled=False), mfdeconv=None), log.append).run()
    cloudy_info = [f for f in result.frames if "cloudy" in f.path.name][0]
    assert cloudy_info.rejected_reason and "filter" in cloudy_info.rejected_reason
    _, meta = load_frame(out)
    assert meta.header["NFRAMES"] == 6
    assert any("Excluded" in line for line in log)


def test_cli_runs(light_dir, tmp_path, capsys):
    out = tmp_path / "cli.fit"
    code = main(["stack", str(light_dir), "-o", str(out), "--cpu", "--method", "sigma", "--weighting", "none"])
    assert code == 0 and out.exists()
    assert "Finished" in capsys.readouterr().out


def test_varying_transparency_keeps_star_nebula_balance(tmp_path, scene):
    """Frames at 65-100% transparency must stack with stars and nebula on the same (reference) flux scale.

    Noise-ratio normalisation + block-median local norm pinned the nebula to the reference while stars
    came out ~sqrt(T) low (star/nebula balance 0.87 against PixInsight on real data).
    """

    from astropy.io import fits
    from conftest import shifted_noisy
    from gpustacker.detection import detect_stars

    base, xy = scene
    folder = tmp_path / "lights"
    folder.mkdir()
    rng = np.random.default_rng(5)
    sky = 50.0
    shifts = [(0, 0), (2.1, -1.3), (-1.7, 2.4), (3.3, 1.9), (-2.8, -2.2), (1.1, 3.6)]
    transps = [1.0, 0.65, 0.7, 0.8, 0.9, 0.75]
    for i, ((dx, dy), t) in enumerate(zip(shifts, transps)):
        frame = shifted_noisy((base - sky) * t + sky, dx, dy, rng, read_noise=3.0)  # haze dims signal, not sky
        fits.PrimaryHDU(data=frame, header=fits.Header({"EXPTIME": 60.0, "EGAIN": 1.0})).writeto(folder / f"light_{i:02d}.fit")
    out = tmp_path / "t.fit"
    settings = PipelineSettings(lights=sorted(folder.glob("*.fit")), output=out, device="cpu", mfdeconv=None, local_norm=LocalNormSettings(enabled=True, block=32), filters=FilterSettings(transparency_min=0.0))
    result = StackingPipeline(settings).run()
    assert all(f.flux_scaled for f in result.frames if f.rejected_reason is None)
    stacked, _ = load_frame(out)
    img = stacked[0]
    # nebula: mean over the bright blob (centre 0.6W, 0.4H in make_field) excluding stars
    h, w = base.shape
    stars = detect_stars(img, 5.0)
    mask = np.zeros_like(img, bool)
    for x, y in zip(stars.x, stars.y):
        mask[max(0, int(y) - 7) : int(y) + 8, max(0, int(x) - 7) : int(x) + 8] = True
    blob = np.zeros_like(mask)
    blob[int(0.4 * h) - 25 : int(0.4 * h) + 25, int(0.6 * w) - 35 : int(0.6 * w) + 35] = True
    sel = blob & ~mask
    neb_ratio = (img - sky)[sel].mean() / (base - sky)[sel].mean()
    # star flux vs truth: aperture sums at the known positions (r=6 so resampled wings are included)
    def ap(a, x, y):
        xi, yi = int(round(x)), int(round(y))
        return (a[yi - 6 : yi + 7, xi - 6 : xi + 7] - sky).sum()
    star_ratio = np.median([ap(img, x, y) / ap(base, x, y) for x, y in xy if 30 < x < w - 30 and 30 < y < h - 30])
    # a reasonably clear frame must be the reference (haze would otherwise win on low noise)
    assert 0.8 < neb_ratio < 1.1 and 0.8 < star_ratio < 1.1, (star_ratio, neb_ratio)
    assert abs(star_ratio / neb_ratio - 1.0) < 0.03, (star_ratio, neb_ratio)
