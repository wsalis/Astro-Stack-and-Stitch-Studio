"""User documentation shown by the GUI Help window. Sections are (title, body) pairs; keep in sync with the options."""

from __future__ import annotations

HELP_SECTIONS: list[tuple[str, str]] = [
    (
        "Overview",
        """GPUStacker turns a folder of light frames into a calibrated, registered, rejection-stacked master.
Everything heavy runs on the GPU as PyTorch tensors (CPU fallback is automatic).

Pipeline, in order:

  1. Load FITS/XISF lights (16-bit integer data is read as ADU floats).
  2. Master calibration: bias/dark subtraction, flat division. Normalised [0..1] masters are
     rescaled to light units automatically; a 200 ADU pedestal is added after subtraction so no
     pixel goes negative.
  3. Cosmetic hot/cold pixel repair (on the raw CFA for OSC cameras).
  4. Debayer (VNG or bilinear) when the frame carries a Bayer pattern.
  5. Star detection on every frame (multi-process), FWHM and sky-noise measurement.
  6. Frame filters: too few stars, bright sky, poor FWHM, low transparency.
  7. Reference choice: a clear, sharp frame with a normal star count.
  8. Registration: astroalign star-pattern match -> similarity transform, refined with a cubic
     polynomial over all matched stars. Meridian flips (180 deg) are detected and handled.
  9. Lanczos-3 GPU warp onto the reference grid; registered frames are stored as float32 memmaps.
 10. Aperture photometry on a shared star set gives each frame's transparency.
 11. Normalisation: per-channel sky offset and 1/transparency flux scale (PixInsight-style LN).
 12. Weights (default PSF-signal weight = (transparency / noise)^2).
 13. Stack pass 1 with the chosen rejection; pass 2 with local normalisation maps fitted against
     the pass-1 result (removes gradients and vignetting mismatch frame-by-frame).
 14. Optional autocrop to the region covered by the requested fraction of frames.
 15. Optional drizzle (2x/3x) and optional ImageMM deconvolution.
 16. Outputs: master FITS, coverage/rejection maps, per-frame CSV, JSON report.

Settings persist between sessions in ~/.gpustacker/gui_settings.json.
Add lights, pick an output file, press Stack. Cancel stops at the next safe point.""",
    ),
    (
        "Light frames & Batch",
        """Add files...        pick individual FITS/XISF files.
Add folder...       every image in one folder.
+ subfolders...     recurse into subfolders (e.g. one folder per night or per panel).
Remove / Clear      manage the list. Previous GPUStacker outputs are skipped automatically.

Batch: one master per group of <keywords>
  Splits the list by FITS header keywords and stacks each group separately, writing one master
  per group into the output folder (the Output FITS field is treated as a folder in batch mode).
  Default keys: OBJECT, FILTER. Any header keyword works; pseudo-keys:
    NIGHT   observing night (DATE-OBS shifted by -12 h, so a session spanning midnight stays together)
    FOLDER  parent folder name
    SIZE    image dimensions (keeps different binning/ROI apart)
  Mosaic panels usually land in separate OBJECT values (e.g. "IC 1805-1-1", "IC 1805-1-2").
  Preview groups lists what would be stacked together without starting a run.""",
    ),
    (
        "Calibration masters",
        """Master bias / Master dark / Master flat
  Optional. FITS or XISF, mono or CFA matching the lights. Masters saved normalised to [0, 1]
  (common for XISF) are detected by their tiny median and rescaled to ADU before use, so you can
  feed PixInsight or Siril masters directly.

  Order: light - dark (or light - bias when no dark) then / flat. Flats are normalised by their
  median before division. A 200 ADU pedestal is added after subtraction so cold nights with sky
  below the dark level do not clip at zero.

  Tip: a master dark matching exposure and temperature matters more than a bias. With a dark
  present, Cosmetic "auto" switches off because the dark already removes hot pixels.""",
    ),
    (
        "Cosmetic correction",
        """Cosmetic: auto | on | off
  auto  = on only when no master dark is supplied.
  Compares each pixel with the median of its 8 same-colour neighbours (stride 2 on CFA data) and
  replaces it when it deviates by more than the chosen number of robust sigmas.

hot sigma  (default 3.0; 0 = off)
  Threshold for bright outliers. A pixel is only treated as hot when it is isolated: if its
  brightest neighbour carries more than ~30 % of the same excess the pixel is a star core and is
  left alone. This protects star flux (an earlier non-protected version ate ~10 % of star light).

cold sigma  (default 0 = off)
  Threshold for dark outliers (dead pixels, dust shadows after a poor flat). Leave off unless you
  see black pixels in the master.""",
    ),
    (
        "Debayer",
        """Debayer: auto | on | off
  auto  = demosaic when the header says the frame is CFA (BAYERPAT / COLORTYP etc.).
  on    = force demosaic (use "pattern" to say which).
  off   = treat the data as mono.

Method: vng | bilinear
  vng (default): Variable Number of Gradients. Interpolates along edges, keeps star peaks sharp
  and does not blur red/blue. Matches PixInsight's default.
  bilinear: fast and smooth but lowers star peaks in R/B by ~30 %. Use only if VNG is too slow.

pattern
  Override the Bayer pattern (RGGB, BGGR, GRBG, GBRG) when the header is wrong or missing.
  Empty = read from header.""",
    ),
    (
        "Rejection",
        """Rejection method (per pixel across frames):
  winsorized (default)   Sigma clipping with a Huber-style winsorised mean/sigma estimate; robust
                         with 10+ frames and gentle on faint signal.
  gesd                   Generalized Extreme Studentized Deviate (Rosner 1983). Statistically
                         principled; best for large stacks (20+ frames). PixInsight's recommendation
                         for many frames. Tuned by the GESD row.
  sigma                  Plain iterative sigma clipping around the median.
  percentile             Reject samples more than low/high fraction away from the median
                         (relative). For very small stacks (3-6 frames).
  median                 Median combine, no rejection masks.
  none                   Weighted mean of everything.

low / high (sigma)      Rejection thresholds below/above the centre in robust sigmas.
                         Lower = more aggressive. 3/3 is a safe default; raise "low" to 4 if faint
                         nebula edges look eaten; lower "high" to 2.5 for stubborn satellite trails.
iters                   Clipping iterations (sigma/winsorized). 3 is usually converged.

The log prints the rejected sample fraction. 0.3-2 % is typical; above ~5 % something else is wrong
(bad registration, wrong normalisation, clouds).""",
    ),
    (
        "GESD options",
        """outliers  (default 0.3)
  Maximum fraction of frames that may be rejected at any pixel. 0.3 = up to 30 % of samples.
significance  (default 0.05)
  Test alpha. Smaller = fewer rejections. 0.05 matches PixInsight's default; 0.01 is conservative.
low relax  (default 1.5)
  Multiplier that makes LOW (dark) outliers harder to reject than high ones. Dark outliers are
  rarely real (cold pixels are fixed earlier), while high outliers (cosmic rays, satellites, hot
  pixels) are common. 1.0 = symmetric.

Large-scale (trails)
  After rejection, where high-rejections are dense in a frame (a satellite or plane trail), the
  rejected region is grown by a few pixels so the faint trail wings go too. Costs a little speed.""",
    ),
    (
        "Weighting, normalisation, hardware",
        """Weighting
  psfsw (default)   PSF-signal weight: (transparency / noise)^2 per frame. Clear, low-noise frames
                    dominate; hazy frames are down-weighted. Equivalent in spirit to PixInsight's
                    PSF Signal Weight.
  psfsw+fwhm        As psfsw times (median FWHM / frame FWHM)^2. Sharper frames weigh more. Tends to
                    reduce effective depth slightly; use when seeing varied a lot.
  psfsw+field       PSFSW with a capped regional sharpness factor. The shared sensor-field pattern is
                    divided out, so only frame-to-frame local softness changes the weight.
  noise             (median noise / frame noise)^2 only.
  fwhm              (median FWHM / frame FWHM)^2 only.
  both              noise x fwhm.
  none              Equal weights.
  Weights are normalised to a median of 1 and clipped to [0.05, 20]. The log prints the range and
  the master header carries NEFF = (sum w)^2 / sum w^2, the effective number of frames.

Selected weighting masters (off by default)
  In the GUI, open Weighting and Ctrl-click to select additional methods. The selected primary
  weighting remains the normal output when used alone. When multiple methods are selected, every
  master, including the primary, gets a _weight-<method> suffix.
  Calibration and registration are shared; stacking and local normalisation repeat per method.

Normalise
  Subtract each frame's sky and scale its flux onto the reference (per channel) before combining.
  Scale comes from aperture photometry (1/transparency), falling back to the noise ratio. Always
  leave on unless frames are already normalised.

Local norm (2-pass)
  Fit a 128 px-block offset map for every frame against the first-pass stack and stack again with
  those maps applied. Removes gradients, light-pollution changes and vignetting mismatches so the
  rejection sees a consistent background. Costs one extra stacking pass.

Force CPU
  Ignore CUDA. Slow but useful for debugging or machines without a usable GPU.

workers
  CPU processes used for star detection and loading. 0 = all cores minus one.""",
    ),
    (
        "Frame filters",
        """The Filters checkbox enables or disables all frame exclusion filters. The CLI equivalent is
      --no-filters. When enabled, thresholds are multiples of the session median; 0 disables that individual
      filter. Excluded frames are listed in the log with the reason and in <name>.frames.csv.

transp <  (default 0.6)
  Exclude frames whose transparency (star flux relative to the session median) is below this.
  Catches cloud and heavy haze.
stars <  (default 0.6)
  Exclude frames with fewer detected stars than this fraction of the median (clouds, dew,
  tracking loss).
sky >  (default 2.0)
  Exclude frames whose background is more than this multiple of the median (moonrise, lights,
  dawn).
FWHM >  (default 1.4)
  Exclude frames whose star FWHM exceeds this multiple of the median (poor seeing, wind, focus
  drift).""",
    ),
    (
        "Registration",
        """Interpolation
  lanczos3 (default)  Sharpest; preserves star profiles. Negative lobes are not clamped (clamping
                      broadened stars ~7 %), but output pixels are floored at the minimum of the
                      contributing samples so no dark undershoot ring appears.
  bicubic             Slightly softer, slightly less noise correlation.
  bilinear            Soft; only for a quick preview.

Polynomial refine
  After astroalign's similarity transform, fit a cubic polynomial over ALL matched stars. Corrects
  field distortion between frames (needed for drizzle and for wide fields). Leave on.

star sigma  (default 5.0)
  Detection threshold for star finding (in robust sigmas above the sky). Lower finds more faint
  stars (slower, more false hits in nebulae); raise to 8-10 for very dense fields where astroalign
  struggles.

The reference frame is chosen automatically: among frames with a normal star count and clear sky
(top-30 star flux within 90 % of the best), the one with the smallest FWHM. The log prints the
chosen file.""",
    ),
    (
        "Output options",
        """autocrop: off | 100% | 95% | 90% | 80%
  Trim the master to the largest rectangle where at least that fraction of frames contributed.
  100 % keeps only pixels covered by EVERY frame (no soft edges from dither/drift). 90 % allows a
  few frames to miss the edge in exchange for a larger field. The same box is applied to the
  coverage/rejection maps, the drizzle output (scaled) and the deconvolved image, and recorded in
  the header as CROPX0 / CROPY0 / CROPDPTH. The log reports pixels trimmed per side.

Coverage/rejection maps
  Save <name>_coverage.fit (frames per pixel) and <name>_rejection.fit (rejected sample fraction
  per pixel). Useful to see trails that were removed and where dithering left thin edges.

Keep registered frames
  Keep the work folder with the registered .npy frames instead of deleting it after the run.
  Needed for diagnostics; costs disk space (frames x width x height x 4 bytes x channels).

Plate solve (ASTAP)
  After each master, drizzle and deconvolved image is saved, solve it with ASTAP (auto-detected;
  needs a D50/D80 star database) and write the WCS + SIP distortion into its header (PLTSOLVD = T).
  RA/DEC, FOCALLEN and XPIXSZ from the subs are used as hints; if that fails a whole-sky search
  with the known field size follows. The WCS of the subs is never copied to outputs (it is wrong
  after registration, crop or drizzle), so an unsolved output simply has no WCS. A failed solve
  is logged and does not stop the run. CLI: --no-solve, --astap <path>.""",
    ),
    (
        "Drizzle",
        """Drizzle: off | 2x | 3x
  Variable-pixel linear reconstruction onto a finer grid. Only worthwhile when the data are
  undersampled (FWHM below ~2 px) AND well dithered (20+ frames at different sub-pixel offsets).
  The output <name>_drizzle.fit is scale x larger per axis; <name>_drizzle_weight.fit is the
  accumulated weight map.

pixfrac  (default 0.8)
  Size of the input "drop" relative to an input pixel. Smaller = sharper but noisier and needs
  more frames; 1.0 = plain resampling. 0.7-0.9 is the usual range for 2x. Used by square and
  Gaussian kernels; point places each input sample at its mapped centre.

kernel  (default square)
  square  Uniform drop over the pixfrac footprint (current default).
  point   One mapped sample per input pixel; needs good sub-pixel dithering to avoid holes.
  gaussian Gaussian-weighted samples over the pixfrac footprint; smoother coverage, slightly softer.

min weight  (default 0.05)
  Output pixels below this fraction of the median accumulated weight are set to 0 (uncovered).

Bayer drizzle (raw CFA for OSC)
  Drizzle the raw colour samples directly from the Bayer mosaic instead of the debayered frames.
  Avoids demosaic interpolation entirely and gives genuinely sharper colour at 2x. Ignored for
  mono data.

Rejection masks from the stacking pass are applied, so trails removed in the master are also
removed from the drizzle.""",
    ),
    (
        "ImageMM multi-frame deconvolution",
        """Experimental. Produces <name>_mfdeconv.fit alongside the master by jointly deconvolving the N
        sharpest eligible registered frames with their own measured PSFs (Sukurdeep 2025, AJ 170; practical
notes Marek 2026). Expect sharper stars and tighter detail; expect also ringing, halos and noise
amplification if pushed. The master itself is never altered.

Produce deconvolved image    enable/disable.
Blend       Mix the MFDeconv result with the regular master: 0 = master unchanged, 1 = full
            deconvolution, 0.3 = 30 % deconvolved correction. This affects only the MFDeconv
            output; the regular master is untouched.
Strength: gentle | normal | strong | custom
  Presets for the fields below; editing a field switches to custom.
Frames      Number of sharpest eligible frames fed to the solver (12-24). Stack weight breaks FWHM
            ties; this selection affects only MFDeconv, not the regular master. More = better noise,
            slower.
Iters       Maximum MM iterations. Early stop usually ends it sooner.
Kappa       Per-iteration multiplicative update clip [1/kappa, kappa]. Lower = safer, slower.
Relax       Damping of each update (0.6 = take 60 % of the step). Lower = smoother convergence.
Huber       Robust-loss threshold. Negative = that many x the residual RMS (auto); positive = ADU.
Dering sigma  Floor the solution at local background minus this many sigma to stop dark rings.
            0 disables. 1.5-2.0 is typical.
Stop tol    Early stop when the median per-pixel change drops below this fraction of the sky
            noise. Larger = stops earlier.
Color
  lrgb (default)  Deconvolve luminance only and re-apply the stack's colour by signal share. No
                  colour halos.
  perchannel      Each channel on its own with its own PSF. Sharper colour but can produce
                  green/cyan halos because channels sharpen by different amounts.
  luma            Deconvolve luminance and output it as a mono image.
Star keep-mask
  Exclude saturated (flat-topped) star cores from the data term so their clipped values do not
  drive the solution. Leave on.
Variance map
  Per-pixel noise weighting (shot + read noise from GAIN/RDNOISE in the header, else a robust
  constant). Leave on.

PSF size is automatic: a star PSF is built on a wide stamp and trimmed where the wing falls below
1e-3 of the peak (typically k = 25-31 px), so bright-star wings are modelled, not left as halos.""",
    ),
    (
        "Outputs & report",
        """<name>.fit                 the master. Header carries NFRAMES, NEFF, EXPTOTAL, STACKMTH, REJFRAC,
                           WEIGHTMD, LOCNORM, INTERP, BGNOISE (ADU), autocrop keys, GPUSTACK version.
<name>_coverage.fit        frames contributing per pixel.
<name>_rejection.fit       rejected sample fraction per pixel.
<name>_drizzle.fit         drizzle output (+ _drizzle_weight.fit).
<name>_mfdeconv.fit        deconvolved image.
<name>.frames.csv          per frame: stars, FWHM, noise, sky, transparency, weight, shift/rotation,
                           flip, exclusion reason, and fwhm_grid (4x4 median FWHM per sensor region,
                           rows top to bottom, pre-registration).
<name>.report.json         everything above plus timings, settings and the session "fwhm_map"
                           (tilt check: constant pattern vs frame-to-frame scatter).

Log lines worth reading:
  "Weights (...): min/median/max"   - a very wide range means a few frames dominate.
  "Excluded <file>: ..."            - why a frame was dropped.
  "Meridian flip detected"          - frames rotated 180 deg were registered fine.
  "FWHM map ..." / "Tilt check"     - 4x4 median FWHM per sensor region. Softest vs sharpest ratio
                                      above ~1.08 with small frame-to-frame scatter = constant tilt or
                                      field curvature (mechanical); large scatter = focus/seeing drift.
  "Autocrop ...: trimmed left/right/top/bottom"
  "Saved stack -> ... noise X ADU = Yx better than a single sub, N_eff Z"

Compare... opens a window that measures a GPUStacker master against a reference stack (e.g. a
PixInsight master of the same frames): star count, FWHM, faint-star SNR, nebula SNR by band,
star/nebula flux balance, and 4x4 regional FWHM and matched-star SNR. Pairs the reference
automatically from a sibling "Pix Stacks" folder.

Tilt... opens the tilt inspector on the loaded lights (or any frames/folder you add). Per frame it
draws the sensor as a grid: cell colour = median FWHM relative to the sharpest cell, the ellipse =
median star shape (axis ratio exaggerated x3, dashed when star orientations are random), the arrow
points from the centre to the soft side with the FWHM change across the sensor. Below, the corner
FWHM trend over the session: parallel lines = constant tilt (mechanical); crossing or diverging
lines = tilt that changes with pointing (flexure) or focus drift. "Session median" is the pattern
every frame shares. The text panel separates the tilt plane (asymmetric; shim/adjust the sensor)
from curvature (corners vs centre, symmetric; coma-corrector spacing) and from elongation. A
single frame cannot tell whether the soft side is intra- or extra-focal: refocus slightly inward
and re-shoot - the side that sharpens was too far out. Save CSV... writes the per-frame table.
CLI: gpustacker tilt <frames or folder> [--cells N] [--csv out.csv] [--gui].""",
    ),
    (
        "Troubleshooting",
        """"All frames were excluded" / very few stars
  Lower "star sigma" to 3-4, or disable the stars filter (set to 0). Check that debayer is right;
  a wrong Bayer pattern halves the star count.

Soft, bloated stars
  Make sure a master dark is applied (without one, cosmetic auto is on and may still touch star
  cores on undersampled data). Use lanczos3 and keep Polynomial refine on. Try weighting
  psfsw+fwhm.

Blotchy background or visible frame edges
  Turn Local norm on, use autocrop 100 %, check that the flat matches (same optics/rotation).

Faint nebula edges eaten by rejection
  Raise sigma "low" to 4 or switch to gesd with low relax 1.5-2.0.

Satellite trails survive
  Lower sigma "high" to 2.5, enable Large-scale (trails), or use gesd.

Out of GPU memory
  The stacker streams row bands and should fit anything; if MFDeconv or drizzle fails, reduce
  Frames (MFDeconv) or enable Force CPU for a test.

Run is slow
  Star detection is CPU bound: raise workers to your core count. VNG debayer costs ~2x bilinear.
  Local norm and GESD each add a pass.""",
    ),
]
