"""FITS / XISF loading and saving with the header facts the pipeline needs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SUPPORTED_SUFFIXES = (".fit", ".fits", ".fts", ".xisf")

# A sub's WCS no longer matches after registration, crop or drizzle, so it is never copied to outputs.
WCS_KEY = re.compile(
    r"^(WCSAXES|WCSNAME|CTYPE\d|CUNIT\d|CRPIX\d|CRVAL\d|CDELT\d|CROTA\d|CD\d_\d|PC\d_\d|PV\d_\d+|LONPOLE|LATPOLE|"
    r"(A|B|AP|BP)_(ORDER|DMAX|\d+_\d+)|PLTSOLVD|IMAGEW|IMAGEH)$"
)


@dataclass
class FrameMeta:
    path: Path
    shape: tuple[int, int]
    channels: int
    exposure: float | None = None
    gain: float | None = None
    read_noise: float | None = None
    filter_name: str | None = None
    bayer_pattern: str | None = None
    fwhm: float | None = None
    date_obs: str | None = None
    header: dict[str, Any] = field(default_factory=dict)

    @property
    def is_cfa(self) -> bool:
        return self.channels == 1 and bool(self.bayer_pattern)


def is_gpustacker_output(path: Path | str) -> bool:
    """True for files this program wrote (stacks, maps, deconvolutions); header-only read."""

    p = Path(path)
    if p.suffix.lower() == ".xisf":
        return False
    try:
        from astropy.io import fits

        header = fits.getheader(p)
    except Exception:
        return False
    return header.get("SOFTWARE") == "GPUStacker" or "GPUSTACK" in header or "MAPTYPE" in header


def discover_frames(folder: Path | str, recursive: bool = False, skip_outputs: bool = True) -> list[Path]:
    root = Path(folder)
    pattern = "**/*" if recursive else "*"
    files = sorted(p for p in root.glob(pattern) if p.suffix.lower() in SUPPORTED_SUFFIXES and p.is_file())
    if skip_outputs:
        files = [p for p in files if not is_gpustacker_output(p)]
    return files


def _header_float(header: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = header.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _header_str(header: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = header.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def normalize_layout(data: np.ndarray) -> np.ndarray:
    """Return a (C, H, W) float32 array from any common 2-D / 3-D layout."""

    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[-1] in (1, 3) and arr.shape[0] not in (1, 3):
            arr = np.moveaxis(arr, -1, 0)
        elif arr.shape[-1] in (1, 3) and arr.shape[0] in (1, 3) and arr.shape[0] > arr.shape[-1]:
            arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Unsupported image dimensionality: {arr.shape}")
    if arr.shape[0] not in (1, 3):
        raise ValueError(f"Expected 1 or 3 channels, got shape {arr.shape}")
    out = np.ascontiguousarray(arr, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _read_fits(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        hdu = next((h for h in hdul if h.data is not None and getattr(h.data, "ndim", 0) >= 2), None)
        if hdu is None:
            raise ValueError(f"No image data in {path}")
        data = np.asarray(hdu.data)
        header = {k: v for k, v in hdu.header.items() if k and k not in ("COMMENT", "HISTORY")}
    return data, header


def _read_xisf(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    from xisf import XISF

    reader = XISF(str(path))
    data = reader.read_image(0)
    meta = reader.get_images_metadata()[0]
    header: dict[str, Any] = {}
    for key, entries in meta.get("FITSKeywords", {}).items():
        if entries:
            header[key] = entries[0].get("value")
    return data, header


def load_frame(path: Path | str) -> tuple[np.ndarray, FrameMeta]:
    """Load an image as (C, H, W) float32 plus parsed metadata."""

    p = Path(path)
    if p.suffix.lower() == ".xisf":
        raw, header = _read_xisf(p)
    else:
        raw, header = _read_fits(p)
    data = normalize_layout(raw)
    channels, height, width = data.shape
    meta = FrameMeta(
        path=p,
        shape=(height, width),
        channels=channels,
        exposure=_header_float(header, "EXPTIME", "EXPOSURE"),
        gain=_header_float(header, "EGAIN", "GAIN"),
        read_noise=_header_float(header, "RDNOISE", "READNOIS", "RON"),
        filter_name=_header_str(header, "FILTER"),
        bayer_pattern=_header_str(header, "BAYERPAT", "COLORTYP"),
        fwhm=_header_float(header, "FWHM"),
        date_obs=_header_str(header, "DATE-OBS"),
        header=header,
    )
    if meta.bayer_pattern and meta.bayer_pattern.upper() not in ("RGGB", "BGGR", "GRBG", "GBRG"):
        meta.bayer_pattern = None
    return data, meta


def load_meta(path: Path | str) -> FrameMeta:
    """Read frame metadata and geometry without decoding the image pixels."""

    p = Path(path)
    if p.suffix.lower() == ".xisf":
        from xisf import XISF

        image = XISF(str(p)).get_images_metadata()[0]
        width, height, channels = image["geometry"]
        header = {
            key: entries[0].get("value")
            for key, entries in image.get("FITSKeywords", {}).items()
            if entries
        }
    else:
        from astropy.io import fits

        with fits.open(p, memmap=True, do_not_scale_image_data=True) as hdul:
            hdu = next((h for h in hdul if int(h.header.get("NAXIS", 0)) >= 2), None)
            if hdu is None:
                raise ValueError(f"No image data in {p}")
            header = {k: v for k, v in hdu.header.items() if k and k not in ("COMMENT", "HISTORY")}
        axes = [int(header.get(f"NAXIS{i}", 1)) for i in range(1, int(header["NAXIS"]) + 1)]
        raw_shape = tuple(reversed(axes))
        if len(raw_shape) == 2:
            channels, height, width = 1, raw_shape[0], raw_shape[1]
        elif len(raw_shape) == 3:
            if raw_shape[-1] in (1, 3) and raw_shape[0] not in (1, 3):
                height, width, channels = raw_shape
            elif raw_shape[-1] in (1, 3) and raw_shape[0] in (1, 3) and raw_shape[0] > raw_shape[-1]:
                height, width, channels = raw_shape
            else:
                channels, height, width = raw_shape
        else:
            raise ValueError(f"Unsupported image dimensionality: {raw_shape}")

    meta = FrameMeta(
        path=p,
        shape=(int(height), int(width)),
        channels=int(channels),
        exposure=_header_float(header, "EXPTIME", "EXPOSURE"),
        gain=_header_float(header, "EGAIN", "GAIN"),
        read_noise=_header_float(header, "RDNOISE", "READNOIS", "RON"),
        filter_name=_header_str(header, "FILTER"),
        bayer_pattern=_header_str(header, "BAYERPAT", "COLORTYP"),
        fwhm=_header_float(header, "FWHM"),
        date_obs=_header_str(header, "DATE-OBS"),
        header=header,
    )
    if meta.bayer_pattern and meta.bayer_pattern.upper() not in ("RGGB", "BGGR", "GRBG", "GBRG"):
        meta.bayer_pattern = None
    return meta


def save_fits(path: Path | str, data: np.ndarray, header_items: Iterable[tuple[str, Any, str]] = (), base_header: dict[str, Any] | None = None) -> Path:
    """Write a (C, H, W) or (H, W) float32 image to FITS."""

    from astropy.io import fits

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    header = fits.Header()
    for key, value in (base_header or {}).items():
        if key in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "BZERO", "BSCALE", "BAYERPAT") or WCS_KEY.match(key):
            continue
        try:
            header[key] = value
        except (ValueError, TypeError):
            continue
    for key, value, comment in header_items:
        header[key] = (value, comment)
    header["SOFTWARE"] = ("GPUStacker", "gpustacker.io")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=arr, header=header).writeto(out, overwrite=True)
    return out
