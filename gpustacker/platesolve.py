"""Plate solving of finished masters with ASTAP (https://www.hnsky.org/astap.htm, MPL 2.0)."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import WCS_KEY

ASTAP_ERRORS = {
    16: "error reading image file",
    32: "no star database found (install D50/D80 from hnsky.org)",
    33: "error reading star database",
}


@dataclass
class SolveResult:
    ra: float  # deg
    dec: float  # deg
    scale: float  # arcsec / px
    rotation: float  # deg, east of north
    flipped: bool
    seconds: float


def find_astap(explicit: Path | str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for name in ("astap_cli", "astap"):
        found = shutil.which(name)
        if found:
            return Path(found)
    roots = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LOCALAPPDATA")]
    candidates = [Path(r) / "astap" / exe for r in roots if r for exe in ("astap_cli.exe", "astap.exe")]
    candidates += [Path("/Applications/ASTAP.app/Contents/MacOS/astap"), Path("/opt/astap/astap_cli"), Path("/opt/astap/astap")]
    return next((c for c in candidates if c.is_file()), None)


def _sexagesimal(value: Any, hours: bool) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(":", " ").replace("h", " ").replace("m", " ").replace("s", " ").replace("d", " ")
    parts = text.split()
    if not parts:
        return None
    try:
        nums = [float(p) for p in parts[:3]]
    except ValueError:
        return None
    sign = -1.0 if parts[0].startswith("-") else 1.0
    deg = abs(nums[0]) + (nums[1] / 60 if len(nums) > 1 else 0) + (nums[2] / 3600 if len(nums) > 2 else 0)
    return sign * deg * (15.0 if hours and len(nums) > 1 else 1.0)


def solve_hints(header: Any) -> dict[str, float]:
    """RA/Dec [deg] and field height [deg] from the mount/optics keys a master inherits from its subs."""

    hints: dict[str, float] = {}
    ra = _header_num(header, "RA")
    ra = ra if ra is not None else _sexagesimal(header.get("OBJCTRA"), hours=True)
    dec = _header_num(header, "DEC")
    dec = dec if dec is not None else _sexagesimal(header.get("OBJCTDEC"), hours=False)
    if ra is not None and dec is not None and -90 <= dec <= 90:
        hints["ra"] = ra % 360.0
        hints["dec"] = dec
    pix = _header_num(header, "XPIXSZ")
    focal = _header_num(header, "FOCALLEN")
    height = _header_num(header, "NAXIS2")
    if pix and focal and height:
        scale = 206.265 * pix / focal / (_header_num(header, "DRIZSCL") or 1.0)
        hints["scale"] = scale
        hints["fov"] = height * scale / 3600.0
    return hints


def _header_num(header: Any, key: str) -> float | None:
    value = header.get(key)
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_wcs(path: Path):
    from astropy.io import fits

    raw = path.read_bytes().decode("ascii", errors="replace")
    return fits.Header.fromstring(raw, sep="\n" if "\n" in raw else "")


def _run_astap(exe: Path, image: Path, base: Path, hints: dict[str, float], radius: float, timeout: float) -> tuple[int, str]:
    cmd = [str(exe), "-f", str(image), "-o", str(base), "-wcs", "-sip", "-z", "0", "-r", f"{radius:g}"]
    cmd += ["-fov", f"{hints['fov']:.4f}" if "fov" in hints else "0"]
    if "ra" in hints:
        cmd += ["-ra", f"{hints['ra'] / 15.0:.6f}", "-spd", f"{hints['dec'] + 90.0:.6f}"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=flags)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def solve_fits(path: Path | str, exe: Path, timeout: float = 180.0) -> SolveResult | None:
    """Solve a FITS file with ASTAP and write the WCS (+SIP) into its header. None = no solution."""

    from astropy.io import fits

    path = Path(path)
    t0 = time.perf_counter()
    hints = solve_hints(fits.getheader(path))
    # Hinted search first; if the mount position is off, widen to the whole sky with the known field size.
    radii = [10.0, 180.0] if "ra" in hints else [180.0]
    with tempfile.TemporaryDirectory(prefix="gpustacker_astap_") as tmp:
        base = Path(tmp) / "solve"
        wcs_path = base.with_suffix(".wcs")
        for radius in radii:
            code, log = _run_astap(exe, path, base, hints, radius, timeout)
            if code == 0 and wcs_path.is_file():
                break
            if code in ASTAP_ERRORS:
                raise RuntimeError(f"ASTAP: {ASTAP_ERRORS[code]}")
            if code not in (1, 2):
                raise RuntimeError(f"ASTAP exit code {code}: {log.splitlines()[-1] if log else ''}")
        else:
            return None
        solved = _read_wcs(wcs_path)

    with fits.open(path, mode="update") as hdul:
        header = hdul[0].header
        for key in [k for k in header if WCS_KEY.match(k)]:
            del header[key]
        for card in solved.cards:
            if WCS_KEY.match(card.keyword) or card.keyword in ("EQUINOX", "RADESYS"):
                header[card.keyword] = (card.value, card.comment)
        header["PLTSOLVD"] = (True, "Plate solved by ASTAP")

    cd11, cd12, cd21, cd22 = (float(solved.get(k, 0.0)) for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"))
    det = cd11 * cd22 - cd12 * cd21
    return SolveResult(
        ra=float(solved["CRVAL1"]),
        dec=float(solved["CRVAL2"]),
        scale=math.sqrt(abs(det)) * 3600.0,
        rotation=math.degrees(math.atan2(cd12, cd22)) % 360.0,
        flipped=det > 0,
        seconds=time.perf_counter() - t0,
    )
