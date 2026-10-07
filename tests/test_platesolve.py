import numpy as np
import pytest
from astropy.io import fits

from gpustacker.io import save_fits
from gpustacker.platesolve import find_astap, solve_fits, solve_hints


def test_save_fits_drops_inherited_wcs(tmp_path):
    base = {"CTYPE1": "RA---TAN-SIP", "CRPIX1": 10.0, "CD1_1": 1e-4, "A_ORDER": 2, "A_0_2": 1e-6, "PLTSOLVD": True, "RA": 10.0, "FOCALLEN": 366}
    out = save_fits(tmp_path / "m.fit", np.zeros((8, 8), np.float32), base_header=base)
    h = fits.getheader(out)
    assert not any(k in h for k in ("CTYPE1", "CRPIX1", "CD1_1", "A_ORDER", "A_0_2", "PLTSOLVD"))
    assert h["RA"] == 10.0 and h["FOCALLEN"] == 366


def test_solve_hints_numeric_and_sexagesimal():
    h = {"RA": 37.6, "DEC": 62.17, "XPIXSZ": 3.76, "FOCALLEN": 366, "NAXIS2": 2708, "DRIZSCL": 2}
    hints = solve_hints(h)
    assert hints["ra"] == pytest.approx(37.6) and hints["dec"] == pytest.approx(62.17)
    assert hints["scale"] == pytest.approx(206.265 * 3.76 / 366 / 2)
    assert hints["fov"] == pytest.approx(2708 * hints["scale"] / 3600)
    hints = solve_hints({"OBJCTRA": "02 30 00", "OBJCTDEC": "-05 30 00"})
    assert hints["ra"] == pytest.approx(37.5) and hints["dec"] == pytest.approx(-5.5)
    assert solve_hints({}) == {}


def test_find_astap_explicit_missing(tmp_path):
    assert find_astap(tmp_path / "nope.exe") is None


def test_solve_keeps_data(tmp_path):
    exe = find_astap()
    if exe is None:
        pytest.skip("ASTAP not installed")
    rng = np.random.default_rng(0)
    data = rng.normal(100, 5, (256, 256)).astype(np.float32)
    out = save_fits(tmp_path / "noise.fit", data)
    assert solve_fits(out, exe, timeout=60) is None
    np.testing.assert_array_equal(fits.getdata(out), data)
    assert "PLTSOLVD" not in fits.getheader(out)
