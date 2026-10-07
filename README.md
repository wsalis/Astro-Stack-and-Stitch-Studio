# GPUStacker

GPU-accelerated deep-sky stacker in Python (PyTorch backend) with an optional
**ImageMM multi-frame PSF-aware deconvolution** output. Dark tkinter GUI and a CLI.

Pipeline: load FITS/XISF -> master calibration -> cosmetic hot/cold repair -> debayer (CFA) ->
star detection (multi-process) -> frame filters (star count, sky level, FWHM) -> similarity
registration (astroalign) refined with a cubic polynomial over all matched stars -> clamped
Lanczos-3 GPU warp -> aperture photometry on a shared star set (transparency) ->
PSF-signal weighting -> row-banded GPU stacking with rejection -> second pass with local
normalisation -> optional autocrop to the full-depth region -> coverage/rejection maps,
per-frame CSV, JSON report -> optional ImageMM deconvolution of the best frames.

## Install

```powershell
cd W:\GPUStacker
py -3 -m pip install -r requirements.txt
```

PyTorch with CUDA must be installed for GPU use (CPU fallback is automatic).

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
`--debayer {auto,on,off}` / `--bayer RGGB`, `--reference`, `--keep-registered`, `--cpu`,
`--vram-fraction`, `--workers`.

Outputs next to the stack: `<name>_coverage.fit` (frames per pixel), `<name>_rejection.fit`
(rejected fraction per pixel), `<name>.frames.csv` (per-frame stats, weights, transparency,
alignment, exclusions), `<name>.report.json`, and `<name>_mfdeconv.fit` when enabled.
When multiple weightings are selected, all masters, including the primary, use
`<name>_weight-<method>.fit`; a single selected weighting keeps `<name>.fit`. Compare accepts
multiple candidate masters in the GUI or repeatable CLI `--also` arguments against one reference.
Compare also reports 4x4 regional FWHM and matched-star SNR where each cell has enough stars.
The log reports meridian-flip detection, excluded frames with reasons, weight range,
autocrop trim per side, stack background noise and effective frame count (`NEFF`).

ImageMM knobs: `--mf-iters`, `--mf-kappa` (step clip), `--mf-relax` (damping),
`--mf-huber` (negative = factor of residual RMS), `--mf-color {perchannel,luma}`,
`--mf-psf-size` (0 = auto-k), `--mf-no-star-mask`, `--mf-no-variance`, `--mf-tile`,
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

## Credits

The deconvolution follows Sukurdeep (2025, AJ 170, doi:10.3847/1538-3881/adfb72) and
the implementation notes of Marek (2026, [ImageMM practical notes](https://doi.org/10.5281/zenodo.19168050)). Full
attribution in [CREDITS.md](CREDITS.md).
