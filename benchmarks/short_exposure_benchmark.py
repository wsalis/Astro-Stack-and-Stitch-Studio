"""Opt-in many-file short-exposure benchmark; emits a reproducibility record as JSON."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from astropy.io import fits

from gpustacker import __version__
from gpustacker.normalization import LocalNormSettings
from gpustacker.pipeline import PipelineSettings, StackingPipeline
from gpustacker.quality import FilterSettings
from gpustacker.storage import estimate_stack_disk_space
from gpustacker.stacking import StackSettings


def make_short_exposure(height: int, width: int, seed: int) -> np.ndarray:
    y_grid, x_grid = np.mgrid[:height, :width].astype(np.float32)
    scene = np.full((height, width), 350.0, dtype=np.float32)
    x_positions = np.linspace(10, width - 10, 7)
    y_positions = np.linspace(10, height - 10, 6)
    for index, (y, x) in enumerate((y, x) for y in y_positions for x in x_positions):
        amplitude = 1800.0 + (index % 9) * 450.0
        scene += amplitude * np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2.0 * 1.35**2))
    rng = np.random.default_rng(seed)
    return rng.poisson(np.maximum(scene, 0)).astype(np.float32)


def directory_bytes(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def run_benchmark(args: argparse.Namespace) -> dict:
    parent = args.directory.resolve() if args.directory else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gpustacker-short-exposure-", dir=parent) as temporary:
        run_dir = Path(temporary)
        lights_dir = run_dir / "lights"
        lights_dir.mkdir()
        lights = []
        for index in range(args.frames):
            path = lights_dir / f"Light_L_{index:05d}_10s.fit"
            header = fits.Header()
            header["EXPTIME"] = 10.0
            header["EGAIN"] = 1.0
            header["RDNOISE"] = 4.0
            header["FILTER"] = "L"
            fits.PrimaryHDU(data=make_short_exposure(args.height, args.width, index), header=header).writeto(path)
            lights.append(path)

        before_free = shutil.disk_usage(run_dir).free
        settings = PipelineSettings(
            lights=lights,
            output=run_dir / "benchmark-master.fit",
            work_dir=run_dir / "work",
            cosmetic="off",
            debayer="off",
            filters=FilterSettings(enabled=False),
            local_norm=LocalNormSettings(enabled=False),
            stack=StackSettings(method="none", normalize=False),
            save_maps=False,
            plate_solve=False,
            device=args.device,
            workers=1,
        )
        estimate = estimate_stack_disk_space(settings, [settings.output], settings.work_dir)
        pipeline = StackingPipeline(settings)
        baseline_bytes = directory_bytes(run_dir)
        peak_bytes = [baseline_bytes]
        stop_sampling = threading.Event()

        def sample_directory() -> None:
            while not stop_sampling.is_set():
                peak_bytes[0] = max(peak_bytes[0], directory_bytes(run_dir))
                if stop_sampling.wait(0.2):
                    return

        sampler = threading.Thread(target=sample_directory, name="gpustacker-benchmark-disk-sampler", daemon=True)
        sampler.start()
        started = time.perf_counter()
        try:
            result = pipeline.run()
            elapsed = time.perf_counter() - started
        finally:
            stop_sampling.set()
            sampler.join()
        peak_bytes[0] = max(peak_bytes[0], directory_bytes(run_dir))
        report = json.loads(result.report_path.read_text(encoding="utf-8"))
        input_bytes = sum(path.stat().st_size for path in lights)
        output_bytes = result.output.stat().st_size
        record = {
            "benchmark": "many_short_exposures",
            "gpustacker_version": __version__,
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "storage_path": str(run_dir),
            "storage_volume": str(Path(run_dir.anchor)),
            "free_bytes_before_run": before_free,
            "free_bytes_after_run": shutil.disk_usage(run_dir).free,
            "peak_storage_sample_interval_seconds": 0.2,
            "estimated_working_bytes": estimate.working_bytes,
            "estimated_output_bytes": estimate.output_bytes,
            "sampled_peak_run_directory_bytes": peak_bytes[0],
            "sampled_peak_growth_during_run_bytes": max(0, peak_bytes[0] - baseline_bytes),
            "settings": {
                "frames": args.frames,
                "shape": [args.height, args.width],
                "exposure_seconds": 10,
                "device_requested": args.device,
                "workers": 1,
                "rejection": "none",
                "local_normalization": False,
            },
            "results": {
                "wall_seconds": round(elapsed, 3),
                "pipeline_seconds": round(result.seconds, 3),
                "frames_per_second": round(args.frames / elapsed, 3),
                "input_bytes": input_bytes,
                "master_bytes": output_bytes,
                "frames_stacked": report["stack"]["frames"],
                "effective_frames": report["stack"]["effective_frames"],
                "rejection_fraction": report["stack"]["rejection_fraction"],
                "background_noise": report["stack"]["background_noise"],
            },
        }
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--directory", type=Path, help="Parent directory on the storage volume to benchmark")
    parser.add_argument("--json", type=Path, help="Also write the result record to this JSON file")
    args = parser.parse_args()
    if args.frames < 2 or args.width < 64 or args.height < 64:
        parser.error("--frames must be at least 2 and dimensions at least 64 pixels")
    record = run_benchmark(args)
    rendered = json.dumps(record, indent=2)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()