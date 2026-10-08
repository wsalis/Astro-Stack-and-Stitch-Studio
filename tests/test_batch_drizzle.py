from __future__ import annotations

import numpy as np
import pytest
import torch
from astropy.io import fits

from gpustacker.batch import DEFAULT_GROUP_KEYS, GroupKey, describe_groups, group_frames, run_batch
from gpustacker.drizzle import DrizzleSettings, drizzle_frame, new_accumulator
from gpustacker.normalization import LocalNormSettings
from gpustacker.pipeline import PipelineSettings
from gpustacker.registration import Alignment
from gpustacker.stacking import FrameNorm


# ----------------------------------------------------------------------------- drizzle


def _shift_alignment(dx: float, dy: float) -> Alignment:
    return Alignment(np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float), 0, 0.0, 1.0, (dx, dy))


@pytest.mark.parametrize("kernel", ["square", "point", "gaussian"])
def test_drizzle_flat_field_and_coverage(kernel):
    acc = new_accumulator(1, 32, 32, 2, torch.device("cpu"))
    frame = torch.full((1, 32, 32), 100.0)
    for dx, dy in [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]:
        drizzle_frame(acc, frame, _shift_alignment(dx, dy), FrameNorm(offset=100.0, scale=1.0, weight=1.0), 100.0, DrizzleSettings(scale=2, kernel=kernel, pixfrac=0.8))
    img = acc.image(0.05)
    inner = img[0, 8:-8, 8:-8]
    assert img.shape == (1, 64, 64)
    assert abs(inner.mean() - 100) < 1e-3 and inner.std() < 1e-3
    assert acc.weight_map()[8:-8, 8:-8].min() > 0


@pytest.mark.parametrize("kernel", ["square", "point", "gaussian"])
def test_drizzle_recovers_subpixel_detail(kernel):
    # a point source dithered by half pixels: drizzle at 2x should be sharper than the 1x mean
    rng = np.random.default_rng(0)
    dim = 48
    yy, xx = np.mgrid[0:dim, 0:dim].astype(np.float32)
    acc = new_accumulator(1, dim, dim, 2, torch.device("cpu"))
    plain = np.zeros((dim, dim), np.float32)
    shifts = [(sx / 4, sy / 4) for sx in range(4) for sy in range(4)]
    for dx, dy in shifts:
        # frame content is the scene sampled at (x+dx, y+dy): the source sees the star shifted by -d
        star = 1000 * np.exp(-((xx + dx - 24) ** 2 + (yy + dy - 24) ** 2) / (2 * 0.6**2)).astype(np.float32)
        frame = torch.from_numpy(star + 10)[None]
        drizzle_frame(acc, frame, _shift_alignment(dx, dy), FrameNorm(offset=10.0, scale=1.0), 10.0, DrizzleSettings(scale=2, kernel=kernel, pixfrac=0.6))
        plain += star + 10
    drz = acc.image(0.05)[0]
    plain /= len(shifts)
    # peak over total flux is a sharpness proxy (independent of pixel scale after accounting for 4x pixels)
    sharp_drz = drz.max() / drz.sum() * 4
    sharp_plain = plain.max() / plain.sum()
    assert sharp_drz > sharp_plain * 1.2
    assert abs(drz.sum() / 4 - plain.sum()) / plain.sum() < 0.05  # flux conserved


def test_drizzle_honours_reject_mask_and_cfa():
    acc = new_accumulator(3, 16, 16, 2, torch.device("cpu"))
    cfa = torch.full((1, 16, 16), 50.0)
    cfa[0, 0::2, 0::2] = 300.0  # R sites (RGGB)
    cfa[0, 1::2, 1::2] = 100.0  # B sites
    mask = torch.zeros((16, 16), dtype=torch.bool)
    mask[6:10, 6:10] = True
    drizzle_frame(acc, cfa, Alignment.identity(), FrameNorm(offset=0.0, scale=1.0), 0.0, DrizzleSettings(scale=2, pixfrac=1.0), reject_mask=mask, cfa_pattern="RGGB")
    img = acc.image(0.0)
    w = acc.den
    assert w[:, 12:20, 12:20].sum() == 0  # rejected block contributes nothing (2x scale)
    r = img[0][w[0] > 0]
    g = img[1][w[1] > 0]
    b = img[2][w[2] > 0]
    assert np.allclose(r, 300.0) and np.allclose(g, 50.0) and np.allclose(b, 100.0)


@pytest.mark.parametrize("cfa", [False, True])
def test_drizzle_applies_local_norm_maps(cfa):
    # local-norm maps are (C, bh, bw); the sampler must return one value per drizzled pixel
    dim = 16
    acc = new_accumulator(3, dim, dim, 2, torch.device("cpu"))
    scale_map = np.ones((3, 2, 2), np.float32)
    offset_map = np.zeros((3, 2, 2), np.float32)
    offset_map[:, :, 0] = -20.0  # left half of the field
    offset_map[:, :, 1] = +20.0  # right half
    norm = FrameNorm(offset=0.0, scale=1.0, weight=1.0, scale_map=scale_map, offset_map=offset_map)
    frame = torch.full((1 if cfa else 3, dim, dim), 100.0)
    drizzle_frame(acc, frame, Alignment.identity(), norm, 0.0, DrizzleSettings(scale=2, pixfrac=1.0), cfa_pattern="RGGB" if cfa else None)
    img = acc.image(0.0)
    covered = acc.den.numpy() > 0
    left = img[:, :, : dim // 2][covered[:, :, : dim // 2]]
    right = img[:, :, -dim // 2 :][covered[:, :, -dim // 2 :]]
    assert left.size and right.size
    assert left.mean() < 95 and right.mean() > 105


# ----------------------------------------------------------------------------- grouping


def _write(path, **hdr):
    h = fits.Header()
    for k, v in hdr.items():
        h[k] = v
    fits.PrimaryHDU(data=np.zeros((8, 8), np.float32), header=h).writeto(path)


def test_group_frames_by_object_and_filter(tmp_path):
    for i, (obj, filt) in enumerate([("NGC 7000 Panel 1", "Ha"), ("NGC 7000 Panel 1", "Ha"), ("NGC 7000 Panel 2", "Ha"), ("NGC 7000 Panel 1", "OIII"), ("M31", None)]):
        kw = {"OBJECT": obj, "DATE-OBS": "2026-09-05T23:10:00"}
        if filt:
            kw["FILTER"] = filt
        _write(tmp_path / f"f{i}.fit", **kw)
    groups = group_frames(sorted(tmp_path.glob("*.fit")), DEFAULT_GROUP_KEYS)
    labels = {g.key.label: len(g.lights) for g in groups}
    assert labels == {"OBJECT=NGC 7000 Panel 1, FILTER=Ha": 2, "OBJECT=NGC 7000 Panel 2, FILTER=Ha": 1, "OBJECT=NGC 7000 Panel 1, FILTER=OIII": 1, "OBJECT=M31, FILTER=UNKNOWN": 1}
    slugs = {g.key.slug for g in groups}
    assert "NGC_7000_Panel_1_Ha" in slugs
    assert GroupKey((("OBJECT", "NGC 7000 Panel 1"), ("FILTER", "Ha"))).format("{FILTER}-{OBJECT}") == "Ha-NGC_7000_Panel_1"
    by_night = group_frames(sorted(tmp_path.glob("*.fit")), ("NIGHT",))
    assert len(by_night) == 1 and by_night[0].key.values[0] == ("NIGHT", "2026-09-05")
    assert len(describe_groups(groups)) == 4


def test_run_batch_two_groups(light_dir, tmp_path):
    # tag half the frames as a second panel; each group gets its own master
    files = sorted(light_dir.glob("*.fit"))
    for i, p in enumerate(files):
        with fits.open(p, mode="update") as hdul:
            hdul[0].header["OBJECT"] = f"Panel {1 if i < 3 else 2}"
            hdul[0].header["FILTER"] = "L"
    groups = group_frames(files, ("OBJECT", "FILTER"))
    assert [len(g.lights) for g in groups] == [3, 3]
    settings = PipelineSettings(lights=files, output=tmp_path / "x.fit", reference=files[0], device="cpu", local_norm=LocalNormSettings(enabled=False), save_maps=False)
    log: list[str] = []
    result = run_batch(settings, groups, tmp_path / "masters", "{OBJECT}_{FILTER}", log.append)
    outs = [p for _, p, _ in result.outputs]
    assert all(p is not None and p.exists() for p in outs)
    assert {p.name for p in outs} == {"Panel_1_L.fit", "Panel_2_L.fit"}
    assert result.results[0].frames[result.results[0].reference_index].path == files[0]
    assert result.results[1].frames[result.results[1].reference_index].path in groups[1].lights
    assert any("Batch finished: 2/2" in line for line in log)
    assert not list((tmp_path / "masters").glob("_gpustacker_work*"))
