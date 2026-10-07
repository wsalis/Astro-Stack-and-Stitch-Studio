from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import gpustacker.compare as compare_module
from gpustacker.gui import planned_output_files
from gpustacker.cli import build_parser, main
from gpustacker.drizzle import DrizzleSettings
from gpustacker.io import load_frame
from gpustacker.mfdeconv import MFDeconvSettings
from gpustacker.normalization import LocalNormSettings
from gpustacker.pipeline import FrameInfo, PipelineSettings, StackingPipeline, blend_mfdeconv, select_mfdeconv_indices
from gpustacker.quality import FilterSettings
from gpustacker.stacking import StackSettings


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


def test_mfdeconv_blend_cli_option():
    args = build_parser().parse_args(["stack", "light.fit", "-o", "out.fit", "--mf-blend", "0.25"])
    assert args.mf_blend == 0.25


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
