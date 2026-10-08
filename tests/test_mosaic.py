"""Mosaic assembly on a synthetic two-tile sky with known gain/offset/gradient differences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from gpustacker.mosaic import MosaicBuilder, MosaicSettings, _gaussian_blur, correct_gradient, wcs_from_header

SCALE = 2.0 / 3600.0  # deg / px


def sky_wcs(crpix: tuple[float, float]) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [37.9, 61.7]
    w.wcs.crpix = list(crpix)
    w.wcs.cd = [[-SCALE, 0.0], [0.0, SCALE]]
    return w


def make_sky(height: int = 420, width: int = 1000, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    sky = 300.0 + 120.0 * np.exp(-(((xx - 520) / 180.0) ** 2 + ((yy - 200) / 90.0) ** 2))
    for x, y, a in zip(rng.uniform(10, width - 10, 500), rng.uniform(10, height - 10, 500), rng.uniform(200, 8000, 500)):
        sky += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.4**2))
    return sky.astype(np.float32)


def write_tile(path: Path, data: np.ndarray, crpix: tuple[float, float], with_wcs: bool = True) -> None:
    hdr = fits.Header()
    hdr["OBJECT"] = "synthetic"
    hdr["XPIXSZ"] = 3.76
    hdr["FOCALLEN"] = 388.0
    if with_wcs:
        hdr.update(sky_wcs(crpix).to_header())
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


@pytest.fixture
def two_tiles(tmp_path: Path) -> tuple[Path, Path, np.ndarray, WCS]:
    rng = np.random.default_rng(7)
    sky = make_sky()
    h, w = sky.shape
    yy, xx = np.mgrid[0:h, 0:600].astype(np.float32)
    # tile A: columns 0..600 with a mild gradient; tile B: columns 400..1000, gain 1.25, offset +40, steeper gradient
    a = sky[:, :600] + 0.03 * xx + 0.02 * yy + rng.normal(0, 3.0, (h, 600)).astype(np.float32)
    b = 1.25 * sky[:, 400:] + 40.0 - 0.05 * xx + 0.04 * yy + rng.normal(0, 3.0, (h, 600)).astype(np.float32)
    pa, pb = tmp_path / "tile_a.fit", tmp_path / "tile_b.fit"
    crpix = (500.0, 200.0)  # FITS 1-based reference pixel of the full sky
    write_tile(pa, a, crpix)
    write_tile(pb, b, (crpix[0] - 400.0, crpix[1]))
    return pa, pb, sky, sky_wcs(crpix)


def test_gaussian_blur_preserves_constant(backend):
    import torch

    x = torch.full((2, 64, 80), 5.0, device=backend.device)
    for sigma in (1.5, 4.0, 30.0):
        y = _gaussian_blur(x, sigma)
        assert y.shape == x.shape
        assert torch.allclose(y, x, atol=1e-3)


def test_correct_gradient_flattens_plane():
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:300, 0:400].astype(np.float32)
    img = (200.0 + 0.1 * xx - 0.05 * yy + rng.normal(0, 2.0, (300, 400))).astype(np.float32)[None]
    img[0, 100:130, 200:260] += 400.0  # bright nebula patch must not bend the fit
    out, pp = correct_gradient(img, degree=1, block=32)
    assert pp[0] == pytest.approx(0.1 * 400 + 0.05 * 300, rel=0.1)
    rows = np.median(out[0, :, :150], axis=1)
    cols = np.median(out[0, :80], axis=0)
    assert np.ptp(rows) < 2.0 and np.ptp(cols) < 2.0
    assert abs(float(np.median(out[0])) - float(np.median(img[0]))) < 5.0


def test_wcs_from_header_rejects_missing():
    assert wcs_from_header({"OBJECT": "x"}, (10, 10)) is None
    h = {k: v for k, v in sky_wcs((5.0, 5.0)).to_header().items()}
    h["CRVAL1"] = str(h["CRVAL1"])  # XISF-style string keyword
    wcs = wcs_from_header(h, (10, 10))
    assert wcs is not None and wcs.has_celestial


def test_mosaic_uses_coverage_map_for_zero_valued_pixels(tmp_path):
    shape = (120, 140)
    image = np.zeros(shape, dtype=np.float32)
    paths = [tmp_path / "tile_a.fit", tmp_path / "tile_b.fit"]
    for path in paths:
        write_tile(path, image, (70.0, 60.0))
        fits.PrimaryHDU(data=np.ones(shape, dtype=np.uint8)).writeto(
            path.with_name(f"{path.stem}_coverage.fit")
        )

    builder = MosaicBuilder(MosaicSettings(paths, tmp_path / "mosaic.fit", plate_solve=False))
    tiles = builder.load_tiles()

    assert all(np.isfinite(tile.image).all() for tile in tiles)


def test_mosaic_two_tiles_seamless(two_tiles, backend, tmp_path):
    pa, pb, sky, sky_w = two_tiles
    out = tmp_path / "mosaic.fit"
    log: list[str] = []
    settings = MosaicSettings(tiles=[pa, pb], output=out, gradient_degree=1, gradient_block=32, feather=60.0, plate_solve=False, auto_crop=False, device="cpu" if not backend.is_cuda else "auto")
    builder = MosaicBuilder(settings, log.append)
    result = builder.run()
    assert out.is_file() and result.coverage_output is not None and result.coverage_output.is_file()
    assert not any(f.code == "localized_overlap_residual" for f in result.analysis.findings)
    mosaic = fits.getdata(out).astype(np.float32)
    cov = fits.getdata(result.coverage_output)
    hdr = fits.getheader(out)
    assert hdr["MOSTILES"] == 2 and hdr["CTYPE1"].startswith("RA---TAN")
    assert result.pixel_scale == pytest.approx(2.0, rel=1e-3)

    # canvas pixel of sky pixel (0, 0): integer offset since scale/rotation match the sky grid
    ra, dec = sky_w.all_pix2world(0, 0, 0)
    ox, oy = (float(v) for v in builder.out_wcs.all_world2pix(ra, dec, 0))
    assert abs(ox - round(ox)) < 1e-3 and abs(oy - round(oy)) < 1e-3
    ox, oy = int(round(ox)), int(round(oy))
    h, w = sky.shape
    region = mosaic[oy : oy + h, ox : ox + w]
    cov_region = cov[oy : oy + h, ox : ox + w]
    assert region.shape == sky.shape
    assert cov_region[20:-20, 20:-20].min() >= 1 and cov_region.max() == 2

    # Compare against the truth in tile A's photometric system (gain 1); gradient removal keeps the
    # sky level, so allow one global offset. Stars excluded via a per-pixel threshold on the truth.
    core = (slice(30, h - 30), slice(30, w - 30))
    diff = (region - sky)[core]
    faint = sky[core] < 460.0
    offset = float(np.median(diff[faint]))
    resid = diff[faint] - offset
    assert np.std(resid) < 6.0  # ~ sqrt(2) * 3 ADU noise plus interpolation
    # no seam: background level must change smoothly across A-only, overlap and B-only columns
    # (the broad synthetic nebula leaves a gentle bow after per-tile gradient removal; a seam is a jump)
    cols = np.array([np.nanmedian(np.where(faint, diff, np.nan)[:, c0 : c0 + 20]) for c0 in range(0, w - 60 - 20, 20)])
    cols = cols[np.isfinite(cols)]
    assert len(cols) > 20 and np.max(np.abs(np.diff(cols))) < 1.5, cols
    # a star in the B-only part keeps its position and brightness (within interpolation)
    ys, xs = np.unravel_index(np.argmax(np.where(np.arange(w)[None] > 650, sky, 0)), sky.shape)
    peak = region[ys - 1 : ys + 2, xs - 1 : xs + 2].max()
    assert peak > 0.8 * sky[ys, xs]
    assert any("Photometry tile_b" in line for line in log)
    b_tile = result.tiles[1]
    assert b_tile.gain[0] == pytest.approx(1 / 1.25, rel=0.05)


def test_mosaic_analyzer_finds_localized_overlap_residual(two_tiles, backend, tmp_path):
    import json

    pa, pb, _, _ = two_tiles
    data = fits.getdata(pb).copy()
    header = fits.getheader(pb)
    data[160:256, 140:236] += 300.0
    fits.PrimaryHDU(data=data, header=header).writeto(pb, overwrite=True)
    settings = MosaicSettings(
        tiles=[pa, pb], output=tmp_path / "mosaic.fit", gradient_degree=1,
        gradient_block=32, feather=60.0, plate_solve=False, auto_crop=False,
        device="cpu" if not backend.is_cuda else "auto",
    )
    builder = MosaicBuilder(settings)

    result = builder.run()

    assert result.analysis_output is not None and result.analysis_output.is_file()
    report = json.loads(result.analysis_output.read_text(encoding="utf-8"))
    assert any(f["code"] == "localized_overlap_residual" for f in report["findings"])
    assert "Blend=Seam" in next(f["suggestion"] for f in report["findings"] if f["code"] == "localized_overlap_residual")
    tile_wcs = wcs_from_header(dict(header), data.shape)
    world = tile_wcs.all_pix2world(188.0, 208.0, 0)
    expected = builder.out_wcs.all_world2pix(*world, 0)
    hotspots = report["overlaps"][0]["hotspots"]
    assert any(np.hypot(h["x"] - expected[0], h["y"] - expected[1]) < 60 for h in hotspots)


def test_mosaic_autocrop_removes_uncovered_border(two_tiles, backend, tmp_path):
    pa, pb, sky, sky_w = two_tiles
    out = tmp_path / "cropped.fit"
    builder = MosaicBuilder(MosaicSettings(
        tiles=[pa, pb], output=out, gradient_degree=0, plate_solve=False,
        auto_crop=True, save_coverage=True, device="cpu" if not backend.is_cuda else "auto",
    ))
    result = builder.run()
    image = fits.getdata(out)
    coverage = fits.getdata(result.coverage_output)
    header = fits.getheader(out)
    assert header["MOSCROP"] is True
    assert image.shape[-2:] == coverage.shape == result.shape
    assert coverage.min() >= 1
    assert header["CROPX0"] > 0 and header["CROPY0"] > 0

    cropped_wcs = wcs_from_header(dict(header), result.shape)
    canvas_wcs = cropped_wcs.deepcopy()
    canvas_wcs.wcs.crpix += np.array([header["CROPX0"], header["CROPY0"]])
    ra, dec = sky_w.all_pix2world(250.0, 120.0, 0)
    canvas_xy = canvas_wcs.all_world2pix(ra, dec, 0)
    cropped_xy = cropped_wcs.all_world2pix(ra, dec, 0)
    np.testing.assert_allclose(cropped_xy, np.asarray(canvas_xy) - [header["CROPX0"], header["CROPY0"]], atol=1e-5)


def test_mosaic_requires_wcs(tmp_path):
    sky = make_sky(120, 300)
    pa, pb = tmp_path / "a.fit", tmp_path / "b.fit"
    write_tile(pa, sky[:, :200], (100.0, 60.0))
    write_tile(pb, sky[:, 100:], (0.0, 60.0), with_wcs=False)
    with pytest.raises(RuntimeError, match="no WCS"):
        MosaicBuilder(MosaicSettings([pa, pb], tmp_path / "m.fit", plate_solve=False, device="cpu")).run()
    with pytest.raises(ValueError):
        MosaicBuilder(MosaicSettings([pa], tmp_path / "m.fit", device="cpu"))
