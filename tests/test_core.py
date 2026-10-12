from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gpustacker.debayer import debayer_bilinear
from gpustacker.io import load_frame, load_meta, normalize_layout, save_fits
from gpustacker.stacking import FrameNorm, FrameStore, StackSettings, cleanup_registered_store, combine_tensor, stack_store, stack_store_band


def test_stack_store_per_channel_normalisation(tmp_path):
    """Frames with a different sky colour must be matched to the reference channel by channel."""

    from gpustacker.backend import select_backend

    backend = select_backend("cpu")
    base = np.full((3, 16, 16), 1000.0, np.float32) + np.random.default_rng(1).normal(0, 2, (3, 16, 16)).astype(np.float32)
    tint = np.array([80.0, 0.0, -60.0], np.float32)[:, None, None]  # different night: redder sky, bluer deficit
    store = FrameStore(tmp_path, (3, 16, 16))
    store.add("ref", base, FrameNorm(offset=np.array([1000.0, 1000.0, 1000.0]), scale=np.ones(3)))
    store.add("other", base + tint, FrameNorm(offset=np.array([1080.0, 1000.0, 940.0]), scale=np.ones(3)))
    out = stack_store(store, backend, StackSettings(method="none", normalize=True)).image
    assert np.allclose(out, base, atol=1e-3)  # colour cast of the second frame fully removed


def test_stack_store_bands_match_full_stack_with_local_norm_weights_and_masks(tmp_path):
    from gpustacker.backend import select_backend

    rng = np.random.default_rng(14)
    shape = (1, 11, 13)
    base = rng.normal(100.0, 2.0, size=shape).astype(np.float32)
    store = FrameStore(tmp_path / "bands", shape)
    for index in range(5):
        frame = base + np.float32(index * 0.2) + rng.normal(0.0, 0.3, size=shape).astype(np.float32)
        if index == 4:
            frame[0, 5, 6] += 20.0
        scale_map = np.full((1, 3, 4), 1.0 + index * 0.01, dtype=np.float32)
        offset_map = np.full((1, 3, 4), index * 0.05, dtype=np.float32)
        weight_map = np.full((4, 5), 1.0 + index * 0.1, dtype=np.float32)
        store.add(
            f"frame-{index}",
            frame,
            FrameNorm(offset=100.0, scale=1.0, weight=1.0 + index * 0.1, scale_map=scale_map, offset_map=offset_map, weight_map=weight_map),
        )
    settings = StackSettings(method="winsorized", iterations=2, use_weights=True, normalize=True)
    backend = select_backend("cpu")

    complete = stack_store(store, backend, settings, reject_masks=True)
    bands = [
        stack_store_band(store, backend, settings, row_start, row_stop, reject_masks=True)
        for row_start, row_stop in ((0, 4), (4, 9), (9, 11))
    ]

    np.testing.assert_allclose(np.concatenate([band.image for band in bands], axis=1), complete.image, rtol=0, atol=2e-5)
    np.testing.assert_array_equal(np.concatenate([band.coverage for band in bands], axis=0), complete.coverage)
    np.testing.assert_allclose(np.concatenate([band.rejection for band in bands], axis=0), complete.rejection, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(
        np.concatenate([band.rejection_masks for band in bands], axis=1),
        np.stack([np.load(path) for path in store.mask_paths]),
    )
    assert sum(band.rejected_low for band in bands) == complete.rejected_low
    assert sum(band.rejected_high for band in bands) == complete.rejected_high
    assert sum(band.total_samples for band in bands) == complete.total_samples


def test_frame_store_index_roundtrip_and_validation(tmp_path):
    directory = tmp_path / "registered"
    expected = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    store = FrameStore(directory, expected.shape)
    store.add("discarded.fit", np.zeros_like(expected))
    store.add("light.fit", expected, FrameNorm(offset=12.5, scale=np.array([1.25]), weight=0.8))
    store.drop(0)
    store.save_index()

    index_path = directory / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["frames"][0]["path"] == "reg_0001.npy"
    index["frames"][0]["path"] = str(directory / "legacy" / "reg_0001.npy")
    index_path.write_text(json.dumps(index), encoding="utf-8")

    restored = FrameStore.load_index(directory)
    assert restored.shape == expected.shape
    assert restored.names == ["light.fit"]
    assert restored.norms[0].offset == 12.5
    assert restored.norms[0].weight == 0.8
    assert restored.norms[0].scale.tolist() == [1.25]
    np.testing.assert_array_equal(np.load(restored.paths[0]), expected)

    outside = tmp_path / "outside.npy"
    np.save(outside, expected)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["frames"][0]["path"] = str(outside)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="outside its store"):
        FrameStore.load_index(directory)


def test_benchmark_store_read_measures_every_frame(tmp_path, capsys):
    from gpustacker.cli import main

    directory = tmp_path / "registered"
    store = FrameStore(directory, (1, 4, 5))
    store.add("first.fit", np.ones((1, 4, 5), dtype=np.float32))
    store.add("second.fit", np.full((1, 4, 5), 2.0, dtype=np.float32))
    store.save_index()

    assert main(["store-benchmark", str(directory), "--rows", "2", "--start-row", "1"]) == 0
    output = capsys.readouterr().out
    assert "Frames: 2; shape: 1x4x5" in output
    assert "Read: 2 rows from each frame at row 1" in output
    assert "Checksum: 3" in output


def test_frame_store_weight_map_roundtrip_and_path_validation(tmp_path):
    directory = tmp_path / "registered"
    store = FrameStore(directory, (1, 4, 5))
    store.add("light.fit", np.ones((1, 4, 5), dtype=np.float32))
    expected_map = np.arange(6, dtype=np.float32).reshape(2, 3)
    store.norms[0].weight_map = expected_map
    store.norms[0].scale_map = (expected_map[None] + 1.0).astype(np.float32)
    store.norms[0].offset_map = expected_map[None].astype(np.float32)
    store.save_index()

    restored = FrameStore.load_index(directory)
    np.testing.assert_array_equal(restored.norms[0].weight_map, expected_map)
    np.testing.assert_array_equal(restored.norms[0].scale_map, store.norms[0].scale_map)
    np.testing.assert_array_equal(restored.norms[0].offset_map, store.norms[0].offset_map)

    index_path = directory / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    outside = tmp_path / "outside.npy"
    np.save(outside, expected_map)
    index["frames"][0]["weight_map"] = "../outside.npy"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="Weight map is outside its store"):
        FrameStore.load_index(directory)

def test_cleanup_registered_store_removes_only_manifest_owned_files(tmp_path):
    directory = tmp_path / "registered"
    store = FrameStore(directory, (1, 4, 5))
    store.add("light.fit", np.ones((1, 4, 5), dtype=np.float32))
    store.norms[0].weight_map = np.ones((2, 3), dtype=np.float32)
    store.norms[0].scale_map = np.ones((1, 2, 2), dtype=np.float32)
    store.norms[0].offset_map = np.zeros((1, 2, 2), dtype=np.float32)
    store.save_index()
    np.save(directory / "rej_0000.npy", np.ones((4, 5), dtype=np.uint8))
    manifest = {
        "schema": 1,
        "stage": "registered_before_stack",
        "shape": list(store.shape),
        "frame_count": len(store),
        "index": "index.json",
    }
    (directory / "registered_store.json").write_text(json.dumps(manifest), encoding="utf-8")
    unrelated = directory / "notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    owned = [path for path in directory.iterdir() if path.name != unrelated.name]
    expected_bytes = sum(path.stat().st_size for path in owned)

    removed_count, removed_bytes = cleanup_registered_store(directory)

    assert removed_count == len(owned)
    assert removed_bytes == expected_bytes
    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert list(directory.iterdir()) == [unrelated]

def test_cleanup_registered_store_can_retry_after_partial_deletion(tmp_path, monkeypatch):
    directory = tmp_path / "registered"
    store = FrameStore(directory, (1, 4, 5))
    store.add("light.fit", np.ones((1, 4, 5), dtype=np.float32))
    store.save_index()
    (directory / "registered_store.json").write_text(json.dumps({
        "schema": 1,
        "stage": "registered_before_stack",
        "shape": list(store.shape),
        "frame_count": len(store),
        "index": "index.json",
    }), encoding="utf-8")
    blocked = store.paths[0]
    original_unlink = Path.unlink
    failed_once = False

    def fail_one_unlink(path, *args, **kwargs):
        nonlocal failed_once
        if path == blocked and not failed_once:
            failed_once = True
            raise PermissionError("temporarily locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_one_unlink)
    with pytest.raises(OSError, match="temporarily locked"):
        cleanup_registered_store(directory)

    assert blocked.exists()
    assert (directory / "cleanup_pending.json").is_file()
    monkeypatch.setattr(Path, "unlink", original_unlink)

    cleanup_registered_store(directory)

    assert list(directory.iterdir()) == []

def test_cleanup_registered_store_rejects_invalid_manifest_without_deleting(tmp_path):
    directory = tmp_path / "registered"
    directory.mkdir()
    (directory / "registered_store.json").write_text('{"schema": 2}', encoding="utf-8")
    unrelated = directory / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        cleanup_registered_store(directory)

    assert unrelated.is_file()


def test_cleanup_registered_store_tolerates_missing_owned_frame(tmp_path):
    directory = tmp_path / "registered"
    store = FrameStore(directory, (1, 4, 5))
    store.add("light.fit", np.ones((1, 4, 5), dtype=np.float32))
    store.save_index()
    (directory / "registered_store.json").write_text(json.dumps({
        "schema": 1,
        "stage": "registered_before_stack",
        "shape": list(store.shape),
        "frame_count": len(store),
        "index": "index.json",
    }), encoding="utf-8")
    store.paths[0].unlink()

    cleanup_registered_store(directory)

    assert list(directory.iterdir()) == []


def test_cleanup_registered_store_rejects_path_traversal_without_mutation(tmp_path):
    directory = tmp_path / "registered"
    store = FrameStore(directory, (1, 4, 5))
    store.add("light.fit", np.ones((1, 4, 5), dtype=np.float32))
    store.save_index()
    outside = tmp_path / "outside.npy"
    np.save(outside, np.ones((1, 4, 5), dtype=np.float32))
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["frames"][0]["path"] = "../outside.npy"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    (directory / "registered_store.json").write_text(json.dumps({
        "schema": 1,
        "stage": "registered_before_stack",
        "shape": list(store.shape),
        "frame_count": len(store),
        "index": "index.json",
    }), encoding="utf-8")
    outside_bytes = outside.read_bytes()

    with pytest.raises(ValueError, match="outside its store"):
        cleanup_registered_store(directory)

    assert outside.read_bytes() == outside_bytes
    assert index_path.exists()


def test_gui_registered_store_cleanup_requires_confirmation(tmp_path, monkeypatch):
    from gpustacker import gui as gui_module

    directory = tmp_path / "registered"
    store = FrameStore(directory, (1, 4, 5))
    store.add("light.fit", np.ones((1, 4, 5), dtype=np.float32))
    store.save_index()
    (directory / "registered_store.json").write_text(json.dumps({
        "schema": 1,
        "stage": "registered_before_stack",
        "shape": list(store.shape),
        "frame_count": len(store),
        "index": "index.json",
    }), encoding="utf-8")

    class Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    app = gui_module.GPUStackerApp.__new__(gui_module.GPUStackerApp)
    app._run_active = False
    app.registered_dir_var = Value(str(directory))
    app._run_active = True
    monkeypatch.setattr(gui_module.messagebox, "showerror", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gui_module.messagebox, "askyesno", lambda *_args, **_kwargs: pytest.fail("active task must block cleanup"))
    app._cleanup_registered_store()
    assert (directory / "reg_0000.npy").exists()

    app._run_active = False
    monkeypatch.setattr(gui_module.messagebox, "askyesno", lambda *_args, **_kwargs: False)
    app._cleanup_registered_store()
    assert (directory / "reg_0000.npy").exists()
    assert app.registered_dir_var.get() == str(directory)

    monkeypatch.setattr(gui_module.messagebox, "askyesno", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(gui_module.messagebox, "showinfo", lambda *_args, **_kwargs: None)
    app._cleanup_registered_store()
    assert app.registered_dir_var.get() == ""
    assert list(directory.iterdir()) == []

def test_stack_store_checks_cancel_callback_before_each_band(tmp_path):
    from gpustacker.backend import select_backend

    store = FrameStore(tmp_path / "cancel", (1, 8, 8))
    store.add("light.fit", np.ones((1, 8, 8), dtype=np.float32))
    callback_count = 0

    def cancel() -> None:
        nonlocal callback_count
        callback_count += 1
        raise RuntimeError("Cancelled")

    with pytest.raises(RuntimeError, match="Cancelled"):
        stack_store(store, select_backend("cpu"), StackSettings(method="none"), cancel_cb=cancel)

    assert callback_count == 1


def test_normalize_layout_variants():
    assert normalize_layout(np.zeros((10, 12))).shape == (1, 10, 12)
    assert normalize_layout(np.zeros((3, 10, 12))).shape == (3, 10, 12)
    assert normalize_layout(np.zeros((10, 12, 3))).shape == (3, 10, 12)


def test_fits_roundtrip(tmp_path):
    data = np.random.default_rng(0).random((3, 8, 9)).astype(np.float32)
    path = save_fits(tmp_path / "x.fit", data, [("NFRAMES", 3, "n")])
    loaded, meta = load_frame(path)
    assert loaded.shape == (3, 8, 9)
    assert np.allclose(loaded, data)
    assert meta.header["NFRAMES"] == 3


def test_save_fits_failure_preserves_existing_product(tmp_path, monkeypatch):
    from astropy.io import fits

    path = save_fits(tmp_path / "stable.fit", np.ones((1, 4, 5), dtype=np.float32))
    original = path.read_bytes()

    def fail_write(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(fits.PrimaryHDU, "writeto", fail_write)
    with pytest.raises(OSError, match="simulated write failure"):
        save_fits(path, np.zeros((1, 4, 5), dtype=np.float32))

    assert path.read_bytes() == original
    assert list(tmp_path.glob("stable.*.fit")) == []


def test_frame_csv_failure_preserves_existing_product(tmp_path, monkeypatch):
    import csv

    from gpustacker.diagnostics import write_frames_csv

    path = tmp_path / "frames.csv"
    path.write_text("previous report\n", encoding="utf-8")

    def fail_header(*_args, **_kwargs):
        raise OSError("simulated CSV failure")

    monkeypatch.setattr(csv.DictWriter, "writeheader", fail_header)
    with pytest.raises(OSError, match="simulated CSV failure"):
        write_frames_csv(path, [{"frame": "light.fit"}])

    assert path.read_text(encoding="utf-8") == "previous report\n"
    assert list(tmp_path.glob("frames.csv.*.tmp")) == []


def test_load_meta_reads_fits_header_without_pixels(tmp_path, monkeypatch):
    path = save_fits(tmp_path / "metadata.fit", np.zeros((3, 12, 20), dtype=np.float32))

    def fail_pixel_load(*_args, **_kwargs):
        raise AssertionError("load_meta should not decode image pixels")

    monkeypatch.setattr("gpustacker.io.load_frame", fail_pixel_load)
    meta = load_meta(path)

    assert meta.shape == (12, 20)
    assert meta.channels == 3


def test_debayer_flat_field_is_flat():
    cfa = torch.full((1, 16, 16), 100.0)
    rgb = debayer_bilinear(cfa, "RGGB")
    assert rgb.shape == (3, 16, 16)
    assert torch.allclose(rgb, torch.full_like(rgb, 100.0), atol=1e-4)


def test_debayer_recovers_pure_red():
    cfa = torch.zeros((1, 16, 16))
    cfa[0, 0::2, 0::2] = 200.0  # R sites only for RGGB
    rgb = debayer_bilinear(cfa, "RGGB")
    assert torch.allclose(rgb[0, 2:-2, 2:-2], torch.full((12, 12), 200.0), atol=1e-3)
    assert torch.all(rgb[1:] == 0)


def test_vng_flat_and_colour_field():
    from gpustacker.debayer import debayer_vng

    assert torch.allclose(debayer_vng(torch.full((1, 16, 16), 100.0), "RGGB"), torch.full((3, 16, 16), 100.0), atol=1e-3)
    # uniform colour (R=300, G=100, B=50) must come back exactly at every pixel
    cfa = torch.full((1, 32, 32), 100.0)
    cfa[0, 0::2, 0::2] = 300.0
    cfa[0, 1::2, 1::2] = 50.0
    rgb = debayer_vng(cfa, "RGGB")[:, 4:-4, 4:-4]
    for c, v in enumerate((300.0, 100.0, 50.0)):
        assert torch.allclose(rgb[c], torch.full_like(rgb[c], v), atol=1e-3)


def test_vng_keeps_star_sharper_than_bilinear():
    from gpustacker.debayer import debayer_vng

    yy, xx = np.mgrid[0:48, 0:48].astype(np.float32)
    star = 2000.0 * np.exp(-((xx - 24.3) ** 2 + (yy - 23.6) ** 2) / (2 * 0.9**2)) + 100.0
    cfa = torch.from_numpy(star)[None]  # grey star: every colour equals the scene
    truth_peak = float(star.max())
    vng = debayer_vng(cfa, "RGGB").mean(dim=0)
    bil = debayer_bilinear(cfa, "RGGB").mean(dim=0)
    assert float(vng.max()) > float(bil.max())
    assert abs(float(vng.max()) - truth_peak) < abs(float(bil.max()) - truth_peak)


def test_sigma_clip_rejects_outlier():
    rng = np.random.default_rng(1)
    stack = torch.from_numpy(rng.normal(100, 2, (12, 1, 6, 6)).astype(np.float32))
    stack[3, 0, 2, 2] = 5000.0
    out, low, high = combine_tensor(stack, StackSettings(method="sigma", sigma_low=3, sigma_high=3))
    assert high >= 1 and abs(float(out[0, 2, 2]) - 100) < 5


def test_winsorized_and_median_ignore_nan():
    stack = torch.full((5, 1, 4, 4), 10.0)
    stack[0] = float("nan")
    stack[1, 0, 0, 0] = 90.0
    for method in ("winsorized", "median", "none", "percentile"):
        out, _, _ = combine_tensor(stack, StackSettings(method=method))
        assert torch.isfinite(out).all()
        if method != "none":
            assert abs(float(out[0, 0, 0]) - 10.0) < 1e-3, method
