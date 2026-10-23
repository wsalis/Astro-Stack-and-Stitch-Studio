# Astro Stack and Stitch Studio

GPU-accelerated deep-sky stacking and mosaic application. The Python package and command-line
entry points remain named `GPUStacker` (`gpustacker`). It includes a dark tkinter GUI and CLI.
**ImageMM multi-frame PSF-aware deconvolution** is an experimental optional output; it never
changes the regular master.

## Version 1.3.0: filtering, storage, and run workflow

This release adds editable frame-filter presets and preview, including an optional post-registration
PSFSW cutoff; multi-drive registered-frame storage; full-field stacking; and persistent run logs.
The CLI accepts preset overrides and repeated `--registered-drive` paths. Distributed stacking,
registration, and storage handling also receive reliability improvements, with expanded regression
coverage. The full test suite passes (242 tests).

## Version 1.2.0: large-stack performance

This release substantially accelerates high-frame-count stacking. On a benchmark of 1,044
3840x2160 Seestar FITS lights (853 stacked), total runtime fell from 22m24s to 8m13s, about
2.7x faster. Rejection remained 0.24%, measured noise 5.59 ADU, and effective frame count
591.3, matching the prior run. Results depend on hardware, storage, and input data.

The gains come from tiled VNG debayering and Lanczos warping, overlapping alignment solves with
GPU registration, shared-memory star-detection input, faster registered-store reads, reusing
local-normalization statistics captured during registration, and more efficient GESD rejection
and critical-value calculation. The full test suite passes (198 tests).

Pipeline: load FITS/XISF -> master calibration -> cosmetic hot/cold repair -> debayer (CFA) ->
star detection (multi-process) -> frame filters (star count, sky level, FWHM) -> similarity
registration (astroalign) refined with a cubic polynomial over all matched stars -> clamped
Lanczos-3 GPU warp -> aperture photometry on a shared star set (transparency) ->
PSF-signal weighting -> row-banded GPU stacking with rejection -> second pass with local
normalisation -> optional autocrop to the full-depth region -> coverage/rejection maps,
per-frame CSV, JSON report -> optional ImageMM deconvolution of the best frames.

The GUI also supports ranked reference-frame review and preview, per-volume disk-space preflight,
resuming matching interrupted runs at saved stage boundaries, and assembling plate-solved master
tiles into a mosaic. Retained registered-frame stores can be stacked without repeating calibration
or registration. Experimental multi-system stacking can offload stack bands to a trusted-LAN worker;
see its limitations and security notes below.

## Install

```powershell
cd W:\GPUStacker
py -3 -m pip install -r requirements.txt
```

GPUStacker supports Python 3.10 and newer. On Windows, `py -3` selects the latest installed
Python 3 interpreter. Use the latest Python release with a wheel available for the pinned
PyTorch 2.14.1 + CUDA 13.0 build; PyTorch support can lag a new Python release. A compatible
NVIDIA driver is required for GPU use; CPU fallback is automatic when CUDA is unavailable.

## Run

```powershell
# GUI
py -3 run_gpustacker.py gui

# CLI: folder, files, or globs
py -3 run_gpustacker.py stack "G:\Target\Lights" -o "G:\Target\stack.fit" --dark master_dark.fit --flat master_flat.fit
py -3 run_gpustacker.py stack "G:\Target\Lights" -o out.fit --mfdeconv --mf-frames 12 --mf-iters 30
py -3 run_gpustacker.py stack --help
py -3 run_gpustacker.py compare master.fit reference.xisf --also master_weight-noise.fit --also master_weight-psfsw-field.fit
```

Key options: `--method {winsorized,gesd,sigma,percentile,median,none}`, `--sigma-low/high`,
`--gesd-outliers/--gesd-significance/--gesd-low-relax`, `--large-scale` (grow trail rejections),
`--cosmetic {auto,on,off}` with `--hot-sigma 3 --cold-sigma 0`,
`--weighting {psfsw,psfsw+fwhm,psfsw+field,noise,fwhm,both,none}` (psfsw = (transparency/noise)^2),
`--also-weighting MODE` (repeat to write only the selected additional weighting masters),
`--filter-transparency 0.6 --filter-stars 0.6 --filter-background 2.0 --filter-fwhm 1.4` (x median; 0 = off; `--no-filters` disables all),
`--interp {lanczos3,bicubic,bilinear}`, `--no-refine`, `--no-local-norm`, `--local-norm-block 128`,
`--autocrop 1.0` (keep only pixels covered by every frame; 0 = off), `--no-maps`,
`--debayer {auto,on,off}` / `--bayer RGGB`, `--reference`, `--keep-registered`,
`--registered-drive PATH` (repeat to distribute registered frames across drives), `--cpu`,
`--vram-fraction`, `--workers`.

### Filter presets and preview

**Lenient**, **Balanced**, and **Selective** supply editable quality-filter thresholds. Balanced
preserves the previous defaults; these are provisional starting points, not fixed rejection quotas.
Presets do not alter stacking weights, pixel rejection, or the advanced PSFSW cutoff.

| Preset | Transparency min | Stars min | Sky max | FWHM max |
| --- | ---: | ---: | ---: | ---: |
| Lenient | 0.4 | 0.35 | 3.0 | 1.8 |
| Balanced | 0.6 | 0.6 | 2.0 | 1.4 |
| Selective | 0.8 | 0.8 | 1.5 | 1.2 |

**Preview filters...** analyzes one stacking group, then updates decisions from cached analysis as
you edit thresholds. The sortable table shows measured values and reasons, with unique rejected
frame counts and retained known exposure. Transparency and PSFSW are pending until registration;
completed-run measurements are available afterward. Changing early filters can change late-stage
baselines on the next run, so previews using previous registration measurements remain estimates.
Batch runs evaluate presets independently for each group; cross-group preview is not supported.

CLI presets accept individual overrides, for example:

```powershell
py -3 run_gpustacker.py stack "G:\Target\Lights" -o master.fit `
  --filter-preset selective --filter-fwhm 1.6
```

The optional **PSFSW min (advanced)** / `--filter-psfsw` cutoff defaults to zero (off). It compares
unclipped `(flux_ratio / noise)^2` with its exposure-group median after registration, independent of
the chosen stacking weights. Groups need at least three valid measurements; unknown measurements
are not rejected. Scores are recorded in the `psfsw` report column separately from stack `weight`.
Retained registered stores reuse saved exclusions and weights and cannot apply a new cutoff.
Disabling quality filters does not bypass unreadable-frame or registration failures.

### Reference review and storage preflight

In the GUI, **Analyze & rank** lists reference candidates. **Preview** displays a candidate without
changing the run reference; use **Use as reference** to explicitly select it. The automatic top-ranked
candidate is previewed by default.

Before a GUI stack starts, output confirmation estimates scratch and output storage on each relevant
volume. Batch runs show group estimates. Treat estimates as planning guidance: actual peak use depends
on the data and options, and representative runs are still needed to calibrate them.

Registered frames can be distributed across multiple drives by adding one frame-storage folder per
drive in the GUI, or repeating `--registered-drive` on the CLI:

```powershell
py -3 run_gpustacker.py stack "D:\Lights" -o "E:\Stacks\master.fit" `
  --registered-drive "G:\GPUStacker frames" --registered-drive "P:\GPUStacker frames"
```

Frames are stored whole in per-run folders and mapped by the registered-store manifest. Use
`--keep-registered` to retain the store after stacking. Multi-drive stores are not supported with
experimental network workers in this first version.

### Resume interrupted runs

Enable **Resume after failure** in the GUI or pass `--resume` on the CLI. A matching rerun validates
its inputs and settings, then reuses completed checkpoints after registration, global stacking,
local-normalization fitting (when enabled), or primary-master writing. Resume is stage-boundary based;
it does not restart from the middle of an active stack band. Keep the work directory until recovery
is complete.

```powershell
py -3 run_gpustacker.py stack "G:\Target\Lights" -o "G:\Target\stack.fit" --resume
```

### Mosaic finished panels

Mosaic assembles two or more plate-solved FITS/XISF masters into a shared WCS canvas, refines
registration from overlap stars, optionally matches gains/backgrounds across overlaps, then blends
the tiles. Open it from the GUI or run:

```powershell
py -3 run_gpustacker.py mosaic "G:\Target\panel-1.fit" "G:\Target\panel-2.fit" -o "G:\Target\mosaic.fit" --gradient auto
```

`--gradient` accepts `auto`, `0`, `1`, `2`, or `3` (default `2`). Auto chooses a per-tile polynomial
degree from 0 through 3 using a conservative fit-selection heuristic and records each choice in the
report and FITS header. It cannot reliably distinguish an instrumental gradient from real extended
nebula emission; compare with `--gradient 0` when broad structure or color changes. The GUI also
offers **feather** and **seam** blending, photometric matching, an optional residual plane, and a
coverage map. See `py -3 run_gpustacker.py mosaic --help` for all options.

To stack an existing registered-frame store, first retain it with **Keep registered frames** enabled,
then run:

```powershell
py -3 run_gpustacker.py stack -o resumed-master.fit --registered-dir "D:\Astro\_gpustacker_work"
```

With `--resume`, GPUStacker checkpoints after registration, after the global first pass and
local-normalization fitting when enabled, and after writing the primary master. If interrupted while
writing optional products, a matching rerun can reuse the completed master and continue those products.

Registered-store stacking reuses the stored normalization, spatial weight maps, frame weights, and
frame metadata. Stack rejection, local normalization, and MFDeconv are supported; registration,
frame filtering, alternate weighting modes, drizzle, and resume are not supported in this mode. The
source store is left untouched unless you explicitly use the GUI's confirmed **Clean up…** action.
Cleanup validates the store ownership list and can be retried after a partial deletion.

To preflight a registered store from a remote Windows worker, use its UNC path rather than a drive
letter:

```powershell
py -3 run_gpustacker.py store-benchmark "\\HOST\Share\_gpustacker_work" --rows 8
```

The bounded probe validates the index and reads the selected rows from every registered frame. Run it
from each worker; it measures shared-store access, not GPU utilization or full-stack wall time.

### Experimental multi-system stacking

The GUI can dispatch registered-frame row bands to an authenticated worker. The desktop continues to
analyze, register, fit local-normalization maps, and write final products; both computers can contribute
to each stack pass. This is experimental and remains opt-in. Use the same GPUStacker version and a
registered-store folder reachable from both computers, preferably through a UNC path. Generate a random
token of at least 32 bytes and use the same value on both computers:

```powershell
$env:GPUSTACKER_WORKER_TOKEN = "<same random token on both hosts>"

# On the worker, start Multi-system setup, select the shared registered-store folder,
# choose a trusted-LAN bind, enable experimental stack-band execution, then start service.
# The command-line worker-agent remains capability-only.
py -3 run_gpustacker.py worker-agent --shared-store "\\HOST\Share\_gpustacker_work"

# To allow a coordinator on the LAN to connect, bind explicitly and restrict
# inbound firewall access to the coordinator. The protocol does not encrypt traffic.
py -3 run_gpustacker.py worker-agent --shared-store "\\HOST\Share\_gpustacker_work" --bind 0.0.0.0

# On the coordinator, check the worker and shared-store path in Multi-system setup.
py -3 run_gpustacker.py worker-probe WORKER-HOST --shared-store "\\HOST\Share\_gpustacker_work" --require-cuda
```

In the main stacking window, enable **Use remote worker for stack bands**, then enter the worker host,
port, and token. For a fresh run, the shared store is created as `_gpustacker_work` beside the output;
the worker service must point at that same folder before the stack begins. For an existing registered
store, select that folder in both windows. Each band is assigned dynamically to the desktop or worker.
The worker is kept two tasks ahead: it reads the next band from the shared store while its GPU computes
the current one, so expect worker RAM use of roughly two bands' worth of frame rows.
If the worker hits CUDA out-of-memory, it retries the assigned range as smaller sub-bands down to one
row (except with large-scale trail rejection). If the worker rejects a task, disconnects, or returns an
invalid artifact, that band is recomputed on the desktop. A remote task response can wait up to six
hours before local fallback; connection establishment still times out after five seconds.

The experimental run control currently excludes batch mode, resume, and multiple weighting variants.
The retained-store workflow's existing limitations still apply. The worker's execution opt-in is off
by default; its GUI capability report indicates whether it is enabled. The CLI `worker-agent` remains
probe-only.

The coordinator verifies the HMAC response, protocol and exact GPUStacker version, store manifest/index
fingerprint, a deterministic sample of up to 32 frame reads, and optional CUDA availability. The worker
checks the complete file listing and temporary write access; it does not read every frame's array header.
Use this only on a trusted, firewall-restricted network; HMAC
authenticates the peer but does not provide confidentiality or protect an exposed endpoint from denial
of service. `--bind 0.0.0.0` should not be used on an untrusted network.

The **Multi-system…** GUI saves its local host, path, port, bind, and probe preferences in
`~/.gpustacker/distributed_settings.json` on each computer. **Remember token securely** is enabled by
default on Windows; the token is stored encrypted for that Windows user with DPAPI, not as readable
text. Uncheck it to require entering the token again next time. The coordinator's store-check response
timeout is adjustable from 1 to 120 minutes (10 minutes by default); the initial network connection
still times out quickly if the worker is unreachable.

The main stacking screen has its own **Remember token securely** option under Experimental multi-system
stacking. It is enabled by default on Windows and saves the shared token in the GUI settings file only as
DPAPI-protected data. Uncheck it to remove the saved token on the next settings save.

The worker protocol uses HMAC authentication but does not encrypt traffic. Bind only on a trusted,
firewall-restricted network. Before relying on distributed runs, measure representative shared-SSD
throughput, GPU utilization, end-to-end wall time, and single-versus-multi-worker output tolerances.
Worker loss, storage disconnect, and coordinator restart recovery are not yet certified; keep the original
registered inputs until the run completes and inspect the output before deleting them.

For a reproducible many-file short-exposure workload, run:

```powershell
py -3 benchmarks/short_exposure_benchmark.py --frames 300 --device auto --directory "D:\Astro\Benchmarks" --json "D:\Astro\Benchmarks\run.json"
```

The harness synthesizes short FITS exposures on the selected volume and records software/hardware,
storage location, settings, wall time, frames per second, the preflight estimate, and sampled peak
run-directory growth. Sampling can miss very brief peaks, so use representative long runs when
calibrating the estimate. For image-quality comparison on representative big-rig data, use the
existing compare command against an immutable reference master; do not treat the synthetic stress
workload as an image-quality benchmark.

Outputs next to the stack: `<name>_coverage.fit` (frames per pixel), `<name>_rejection.fit`
(rejected fraction per pixel), `<name>.frames.csv` (per-frame stats, weights, transparency,
alignment, exclusions), `<name>.report.json`, and `<name>_mfdeconv.fit` when enabled.
Mosaic builds also write `<name>_mosaic_report.json` with overlap residual hotspots, alignment warnings, and settings to compare when investigating a seam.
When multiple weightings are selected, all masters, including the primary, use
`<name>_weight-<method>.fit`; a single selected weighting keeps `<name>.fit`. Compare accepts
multiple candidate masters in the GUI or repeatable CLI `--also` arguments against one reference.
Compare also reports 4x4 regional FWHM and matched-star SNR where each cell has enough stars.
The log reports meridian-flip detection, excluded frames with reasons, weight range,
autocrop trim per side, stack background noise and effective frame count (`NEFF`).

Experimental ImageMM options: `--mf-preset {gentle,normal,strong}` (explicit `--mf-*` flags override
the preset), `--mf-frames`, `--mf-blend` (0 = unchanged master, 1 = full deconvolution), `--mf-iters`,
`--mf-kappa` (step clip), `--mf-relax` (damping), `--mf-huber` (negative = factor of residual RMS),
`--mf-color {lrgb,luma,perchannel}` (`lrgb` is the default), `--mf-psf-size` (0 = auto-k),
`--mf-no-star-mask`, `--mf-no-variance`, `--mf-tile`, `--mf-dering`, `--mf-stop-tol`, and
`--mf-save-psfs DIR`.

## How the GPU is used

- Calibration, debayer, warping, rejection and combination all run as torch tensors.
- Registered frames are written to float32 `.npy` memmaps; stacking streams row bands
  sized from the VRAM budget, so stacks larger than VRAM work.
- MFDeconv tiles the image (with overlap) when the frame set does not fit in VRAM.

## Tests

```powershell
py -3 -m pytest -q
```

## License

GPUStacker is licensed under the MIT License. See [LICENSE](LICENSE).

## Credits

The deconvolution follows Sukurdeep (2025, AJ 170, doi:10.3847/1538-3881/adfb72) and
the implementation notes of Marek (2026, [ImageMM practical notes](https://doi.org/10.5281/zenodo.19168050)). Full
attribution in [CREDITS.md](CREDITS.md).
