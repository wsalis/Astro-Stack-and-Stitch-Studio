# Credits and Attribution

GPUStacker is an independent implementation. The multi-frame deconvolution
module (`gpustacker/mfdeconv.py`) follows a published method and a practitioner
write-up; both deserve explicit credit.

## ImageMM multi-frame deconvolution

**Method (primary reference):**

> Y. Sukurdeep (2025). *ImageMM: Joint Multi-frame Image Restoration and
> Super-resolution.* The Astronomical Journal, 170.
> https://doi.org/10.3847/1538-3881/adfb72

**Practical implementation notes that guided this module:**

> F. Marek (2026). *ImageMM: Multi-Frame PSF-Aware Deconvolution — Practical
> Notes & Implementation Diagram*, Rev 2.1 (March 22, 2026).
> https://doi.org/10.5281/zenodo.19168050

The forward model, robust weighted least-squares objective, Huber weights,
soft star keep-mask, per-pixel variance map, majorize-minimize multiplicative
update with step clipping (kappa) and relaxation (alpha), and auto-k PSF sizing
implemented here are as described in those documents. The early-stop rule
departs from the published median(|u - 1|) criterion: on sky-pedestal data that
ratio is tiny from the first iteration, so GPUStacker stops on the median
per-pixel change relative to the sky noise instead.

**Reference software:** Seti Astro Suite Pro by Franklin Marek
(https://github.com/setiastro/setiastrosuitepro, GPL-3.0). GPUStacker's code
was written from the published equations and does not copy SASpro source.
If you ever port code from SASpro directly, this project must be distributed
under GPL-3.0-compatible terms.

## Libraries

- PyTorch — GPU tensor backend
- Astropy — FITS I/O
- xisf — XISF I/O
- astroalign / sep-pjw — star-pattern registration transform estimation
- SciPy, NumPy — CPU numerics

## How to cite GPUStacker's deconvolution output

If you publish results produced with the `--mfdeconv` path, please cite
Sukurdeep (2025) for the method and Marek (2026) for the implementation notes.
