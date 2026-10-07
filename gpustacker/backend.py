"""Torch device selection, VRAM budgeting, and array helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

StatusCallback = Callable[[str], None]


def _noop(_: str) -> None:
    pass


@dataclass(frozen=True)
class Backend:
    device: torch.device
    name: str
    total_bytes: int
    budget_bytes: int

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def free_bytes(self) -> int:
        if not self.is_cuda:
            return self.budget_bytes
        free, _ = torch.cuda.mem_get_info(self.device)
        return int(free)

    def empty_cache(self) -> None:
        if self.is_cuda:
            torch.cuda.empty_cache()

    def describe(self) -> str:
        gib = self.total_bytes / 2**30
        return f"{self.name} ({gib:.1f} GiB, budget {self.budget_bytes / 2**30:.1f} GiB)"


def select_backend(prefer: str = "auto", vram_fraction: float = 0.75, status_cb: StatusCallback = _noop) -> Backend:
    """Pick CUDA when available (unless forced to CPU) and derive a working memory budget."""

    forced_cpu = prefer == "cpu" or os.environ.get("GPUSTACKER_FORCE_CPU") == "1"
    if not forced_cpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        total = int(props.total_memory)
        free, _ = torch.cuda.mem_get_info(device)
        budget = int(min(free, total) * vram_fraction)
        torch.backends.cudnn.benchmark = True
        backend = Backend(device, props.name, total, budget)
        status_cb(f"Backend: CUDA {backend.describe()} | torch {torch.__version__}")
        return backend

    import psutil

    total = int(psutil.virtual_memory().total)
    budget = int(psutil.virtual_memory().available * 0.5)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    backend = Backend(torch.device("cpu"), "CPU", total, budget)
    status_cb(f"Backend: {backend.describe()} | torch {torch.__version__}")
    return backend


def to_tensor(array: np.ndarray, backend: Backend, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    arr = np.ascontiguousarray(array)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="))
    return torch.from_numpy(arr).to(device=backend.device, dtype=dtype, non_blocking=backend.is_cuda)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy()


def rows_per_band(n_frames: int, width: int, channels: int, budget_bytes: int, overhead: float = 6.0) -> int:
    """Rows of a frame stack that fit in the budget, allowing for intermediates."""

    per_row = n_frames * width * channels * 4 * overhead
    return int(max(16, min(4096, budget_bytes // max(1, per_row))))
