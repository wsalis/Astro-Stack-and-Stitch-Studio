"""Master-frame calibration applied on the torch device."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .backend import Backend, to_tensor
from .io import load_frame

ADU_FULL_SCALE = 65535.0
AUTO_PEDESTAL_ADU = 200.0  # keeps dark-sky frames above zero after dark subtraction (PI "output pedestal")


def _is_unit_range(t: torch.Tensor) -> bool:
    return bool(t.max() <= 1.0 + 1e-6)


@dataclass
class Masters:
    bias: torch.Tensor | None = None
    dark: torch.Tensor | None = None
    flat: torch.Tensor | None = None  # already normalised to median 1.0
    pedestal: float = 0.0
    # PixInsight writes masters normalised to [0, 1]; raw lights are 16-bit ADU
    bias_unit: bool = False
    dark_unit: bool = False

    @property
    def active(self) -> bool:
        return any(m is not None for m in (self.bias, self.dark, self.flat))


def _match_channels(master: torch.Tensor, channels: int) -> torch.Tensor:
    if master.shape[0] == channels:
        return master
    if master.shape[0] == 1:
        return master.expand(channels, -1, -1)
    if channels == 1:
        return master.mean(dim=0, keepdim=True)
    raise ValueError(f"Master has {master.shape[0]} channels; light has {channels}")


def load_masters(backend: Backend, bias: Path | str | None = None, dark: Path | str | None = None, flat: Path | str | None = None, pedestal: float = 0.0) -> Masters:
    """Load master FITS/XISF files; flats are bias/dark-corrected and normalised here."""

    masters = Masters(pedestal=pedestal)
    if bias:
        masters.bias = to_tensor(load_frame(bias)[0], backend)
        masters.bias_unit = _is_unit_range(masters.bias)
    if dark:
        masters.dark = to_tensor(load_frame(dark)[0], backend)
        masters.dark_unit = _is_unit_range(masters.dark)
    if flat:
        flat_t = to_tensor(load_frame(flat)[0], backend)
        if masters.bias is not None:
            bias_t = masters.bias
            if masters.bias_unit != _is_unit_range(flat_t):
                bias_t = bias_t * ADU_FULL_SCALE if masters.bias_unit else bias_t / ADU_FULL_SCALE
            flat_t = flat_t - _match_channels(bias_t, flat_t.shape[0])
        flat_t = flat_t.clamp_min(1e-6)
        med = flat_t.flatten(1).median(dim=1).values.view(-1, 1, 1)
        masters.flat = flat_t / med
    return masters


def calibrate(light: torch.Tensor, masters: Masters) -> torch.Tensor:
    """(light - dark [or bias]) / flat + pedestal, clipped at zero."""

    if not masters.active and masters.pedestal == 0.0:
        return light
    channels = light.shape[0]
    light_unit = _is_unit_range(light)

    def in_light_units(master: torch.Tensor, master_unit: bool) -> torch.Tensor:
        if master_unit == light_unit:
            return master
        return master * ADU_FULL_SCALE if master_unit else master / ADU_FULL_SCALE

    out = light
    if masters.dark is not None:
        out = out - _match_channels(in_light_units(masters.dark, masters.dark_unit), channels)
    elif masters.bias is not None:
        out = out - _match_channels(in_light_units(masters.bias, masters.bias_unit), channels)
    if masters.flat is not None:
        out = out / _match_channels(masters.flat, channels)
    pedestal = masters.pedestal
    if not pedestal and (masters.dark is not None or masters.bias is not None):
        pedestal = AUTO_PEDESTAL_ADU / ADU_FULL_SCALE if light_unit else AUTO_PEDESTAL_ADU
    if pedestal:
        out = out + pedestal
    return out.clamp_min(0.0)
