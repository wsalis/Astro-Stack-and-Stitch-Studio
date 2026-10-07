"""Command-line entry point: ``gpustacker stack <lights...> -o out.fit``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .batch import DEFAULT_GROUP_KEYS, describe_groups, group_frames, run_batch
from .cosmetic import CosmeticSettings
from .drizzle import DrizzleSettings
from .io import discover_frames, is_gpustacker_output
from .mfdeconv import MFDECONV_PRESETS, MFDeconvSettings
from .mosaic import BLEND_MODES, INTERPOLATIONS, ORIENTATIONS
from .normalization import LocalNormSettings
from .pipeline import PipelineSettings, StackingPipeline
from .quality import WEIGHTING_CHOICES, FilterSettings
from .stacking import StackSettings


def _collect_lights(items: list[str]) -> list[Path]:
    lights: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            lights.extend(discover_frames(p))
        elif p.exists():
            lights.append(p)
        else:
            lights.extend(sorted(Path().glob(item)))
    skipped = [p for p in lights if is_gpustacker_output(p)]
    if skipped:
        print(f"Skipping {len(skipped)} GPUStacker output file(s) found among the lights", flush=True)
    return [p for p in lights if p not in skipped]


def _collect_tiles(items: list[str]) -> list[Path]:
    """Master tiles are GPUStacker outputs themselves, so nothing is skipped; maps are left out."""

    tiles: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            tiles.extend(discover_frames(p, skip_outputs=False))
        elif p.exists():
            tiles.append(p)
        else:
            tiles.extend(sorted(Path().glob(item)))
    return [p for p in tiles if not any(p.stem.endswith(s) for s in ("_coverage", "_rejection", "_drizzle_weight"))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpustacker", description="GPU-accelerated deep-sky stacker with ImageMM deconvolution")
    parser.add_argument("--version", action="version", version=f"gpustacker {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    st = sub.add_parser("stack", help="Calibrate, register and stack light frames")
    st.add_argument("lights", nargs="+", help="Light files, folders, or globs")
    st.add_argument("-o", "--output", required=True, type=Path, help="Output FITS path (a directory when --group-by is used)")
    st.add_argument("--group-by", help="Comma-separated FITS keywords; one master per group, e.g. OBJECT,FILTER (also NIGHT, FOLDER, SIZE)")
    st.add_argument("--name-template", help="Output name per group, e.g. '{OBJECT}_{FILTER}' (default: all key values joined)")
    st.add_argument("--list-groups", action="store_true", help="Only print the groups and exit")
    st.add_argument("--bias", type=Path)
    st.add_argument("--dark", type=Path)
    st.add_argument("--flat", type=Path)
    st.add_argument("--pedestal", type=float, default=0.0)
    st.add_argument("--cosmetic", choices=["auto", "on", "off"], default="auto", help="Auto hot/cold pixel repair (auto = when no --dark)")
    st.add_argument("--hot-sigma", type=float, default=3.0, help="Hot pixel threshold (0 disables)")
    st.add_argument("--cold-sigma", type=float, default=0.0, help="Cold pixel threshold (0 disables)")
    st.add_argument("--debayer", choices=["auto", "on", "off"], default="auto")
    st.add_argument("--debayer-method", choices=["vng", "bilinear"], default="vng", help="Demosaicing algorithm (VNG = PixInsight default)")
    st.add_argument("--bayer", dest="bayer_pattern", help="Override CFA pattern (RGGB, BGGR, GRBG, GBRG)")
    st.add_argument("--reference", type=Path, help="Force this light as the registration reference")
    st.add_argument("--sigma", type=float, default=5.0, help="Star detection threshold in sigma")
    st.add_argument("--method", choices=["none", "median", "sigma", "winsorized", "percentile", "gesd"], default="winsorized")
    st.add_argument("--sigma-low", type=float, default=3.0)
    st.add_argument("--sigma-high", type=float, default=3.0)
    st.add_argument("--iterations", type=int, default=3, help="Rejection iterations")
    st.add_argument("--gesd-outliers", type=float, default=0.3, help="GESD: max fraction of frames rejected per pixel")
    st.add_argument("--gesd-significance", type=float, default=0.05)
    st.add_argument("--gesd-low-relax", type=float, default=1.5, help="GESD: >1 protects dark outliers")
    st.add_argument("--large-scale", action="store_true", help="Grow dense high-rejections (satellite/plane trails)")
    st.add_argument("--weighting", choices=list(WEIGHTING_CHOICES), default="psfsw", help="psfsw = (transparency/noise)^2, PixInsight PSF-Signal-Weight style")
    st.add_argument("--also-weighting", choices=list(WEIGHTING_CHOICES), action="append", default=[], help="Also write a master with this weighting (repeatable)")
    st.add_argument("--no-normalize", action="store_true")
    st.add_argument("--no-local-norm", action="store_true", help="Skip the second, locally normalised stacking pass")
    st.add_argument("--local-norm-block", type=int, default=128)
    st.add_argument("--interp", choices=["lanczos3", "bicubic", "bilinear"], default="lanczos3", help="Registration interpolation")
    st.add_argument("--no-refine", action="store_true", help="Skip the polynomial registration refinement")
    st.add_argument("--autocrop", type=float, default=0.0, help="Crop to the region covered by this fraction of frames (1.0 = all, 0 = off)")
    st.add_argument("--no-maps", action="store_true", help="Do not write coverage/rejection map FITS files")
    st.add_argument("--no-solve", action="store_true", help="Do not plate solve the outputs with ASTAP")
    st.add_argument("--astap", type=Path, help="Path to astap_cli/astap executable (default: auto-detect)")
    dz = st.add_argument_group("drizzle")
    dz.add_argument("--drizzle", type=int, default=0, help="Also produce a drizzle integration at this scale (2 or 3; 0 = off)")
    dz.add_argument("--drizzle-kernel", choices=["square", "point", "gaussian"], default="square")
    dz.add_argument("--pixfrac", type=float, default=0.8)
    dz.add_argument("--drizzle-min-weight", type=float, default=0.05, help="Minimum drizzle weight relative to the map median")
    dz.add_argument("--no-cfa-drizzle", action="store_true", help="Drizzle debayered frames instead of raw Bayer sites (OSC)")
    fl = st.add_argument_group("frame filters (0 disables)")
    fl.add_argument("--no-filters", action="store_true", help="Disable all frame exclusion filters")
    fl.add_argument("--filter-transparency", type=float, default=0.6, help="Exclude frames with star flux below this fraction of the reference")
    fl.add_argument("--filter-stars", type=float, default=0.6, help="Exclude frames with star count below this fraction of the median")
    fl.add_argument("--filter-background", type=float, default=2.0, help="Exclude frames with sky above this multiple of the median")
    fl.add_argument("--filter-fwhm", type=float, default=1.4, help="Exclude frames with FWHM above this multiple of the median")
    st.add_argument("--keep-registered", action="store_true", help="Keep registered .npy frames in the work dir")
    st.add_argument("--work-dir", type=Path)
    st.add_argument("--cpu", action="store_true", help="Force CPU backend")
    st.add_argument("--vram-fraction", type=float, default=0.75)
    st.add_argument("--workers", type=int, default=0, help="CPU worker processes for star detection (0 = cores-1)")

    mf = st.add_argument_group("ImageMM multi-frame deconvolution")
    mf.add_argument("--mfdeconv", action="store_true", help="Also produce an ImageMM deconvolved image")
    mf.add_argument("--mf-preset", choices=list(MFDECONV_PRESETS), help="Strength preset; explicit --mf-* flags override its values")
    mf.add_argument("--mf-frames", type=int, default=None, help="Sharpest eligible frames to feed MFDeconv (default 12)")
    mf.add_argument("--mf-blend", type=float, default=0.3, help="Blend MFDeconv with master: 0 = master, 1 = full deconvolution (default 0.3)")
    mf.add_argument("--mf-iters", type=int, default=None, help="(default 30)")
    mf.add_argument("--mf-kappa", type=float, default=None, help="(default 2.0)")
    mf.add_argument("--mf-relax", type=float, default=None, help="(default 0.6)")
    mf.add_argument("--mf-huber", type=float, default=None, help="Huber delta; negative = factor of residual RMS (default -1.5)")
    mf.add_argument("--mf-color", choices=["lrgb", "luma", "perchannel"], default="lrgb", help="lrgb = deconvolve luminance, keep stack colour (no colour halos)")
    mf.add_argument("--mf-psf-size", type=int, default=0, help="Odd PSF size, 0 = auto-k")
    mf.add_argument("--mf-no-star-mask", action="store_true")
    mf.add_argument("--mf-no-variance", action="store_true")
    mf.add_argument("--mf-tile", type=int, default=1024)
    mf.add_argument("--mf-dering", type=float, default=None, help="Floor at sky - N*noise to prevent dark rings; 0 = off (default 2.0)")
    mf.add_argument("--mf-stop-tol", type=float, default=None, help="Early stop when median |change| / sky noise falls below this (default 0.02)")
    mf.add_argument("--mf-save-psfs", type=Path, help="Directory to dump per-frame PSF FITS files")

    cp = sub.add_parser("compare", help="Benchmark a GPUStacker master against a reference stack (e.g. PixInsight) of the same field")
    cp.add_argument("gpu", type=Path, nargs="?", help="GPUStacker output FITS (omit both paths to open the GUI)")
    cp.add_argument("reference", type=Path, nargs="?", help="Reference stack (FITS or XISF)")
    cp.add_argument("--also", type=Path, action="append", default=[], help="Additional GPUStacker master to compare against the same reference (repeatable)")
    cp.add_argument("--json", type=Path, help="Write the metrics to this JSON file")

    tl = sub.add_parser("tilt", help="Sensor tilt / field curvature inspector: per-region FWHM and star shape on unregistered lights")
    tl.add_argument("lights", nargs="*", help="Light files, folders, or globs (omit to open the GUI)")
    tl.add_argument("--cells", type=int, default=4, help="Grid size per axis (default 4)")
    tl.add_argument("--csv", type=Path, help="Write per-frame tilt table to this CSV")
    tl.add_argument("--gui", action="store_true", help="Open the inspector window with the given frames preloaded")

    ms = sub.add_parser("mosaic", help="Gradient-correct, register (by WCS) and seamlessly blend plate-solved master tiles into one mosaic")
    ms.add_argument("tiles", nargs="*", help="Master tile files (FITS/XISF with WCS, or plate-solvable by ASTAP); omit to open the GUI")
    ms.add_argument("-o", "--output", type=Path, help="Output mosaic FITS path")
    ms.add_argument("--gradient", type=int, default=2, help="Per-tile sky gradient polynomial degree, 0 = off (default 2)")
    ms.add_argument("--gradient-block", type=int, default=64, help="Block size in px for the sky samples (default 64)")
    ms.add_argument("--scale", type=float, default=0.0, help="Output pixel scale in arcsec/px (default: finest tile)")
    ms.add_argument("--orientation", choices=list(ORIENTATIONS), default="first", help="first = keep tile 1's rotation (default), north = north up")
    ms.add_argument("--interp", choices=list(INTERPOLATIONS), default="lanczos3")
    ms.add_argument("--no-refine", action="store_true", help="Do not refine tile positions from matched stars in the overlaps")
    ms.add_argument("--no-photometric", action="store_true", help="Skip gain/offset matching in the overlaps")
    ms.add_argument("--no-plane", action="store_true", help="Photometric match without the per-tile residual plane")
    ms.add_argument("--feather", type=float, default=100.0, help="Background blend width in px (default 100)")
    ms.add_argument("--seam", type=float, default=4.0, help="Detail/star seam width in px (default 4)")
    ms.add_argument("--blend", choices=list(BLEND_MODES), default="feather", help="feather = plain feather (default); seam = feathered background + single-tile detail")
    ms.add_argument("--no-autocrop", action="store_true", help="Keep the full canvas, including uncovered black borders")
    ms.add_argument("--no-solve", action="store_true", help="Fail on tiles without WCS instead of plate solving them with ASTAP")
    ms.add_argument("--astap", type=Path, help="Path to astap_cli/astap executable (default: auto-detect)")
    ms.add_argument("--no-maps", action="store_true", help="Do not write the coverage map")
    ms.add_argument("--cpu", action="store_true", help="Force CPU backend")
    ms.add_argument("--vram-fraction", type=float, default=0.75)
    ms.add_argument("--gui", action="store_true", help="Open the mosaic window with the given tiles preloaded")
    return parser


def _mf_values(args: argparse.Namespace) -> dict[str, float | int]:
    """Preset values (if any) overridden by explicitly given --mf-* flags."""

    values: dict[str, float | int] = {"frames": 12, "iterations": 30, "kappa": 2.0, "relax": 0.6, "huber_delta": -1.5, "dering_sigma": 2.0, "early_stop_tol": 0.02}
    if args.mf_preset:
        values.update(MFDECONV_PRESETS[args.mf_preset])
    overrides = {"frames": args.mf_frames, "iterations": args.mf_iters, "kappa": args.mf_kappa, "relax": args.mf_relax, "huber_delta": args.mf_huber, "dering_sigma": args.mf_dering, "early_stop_tol": args.mf_stop_tol}
    values.update({k: v for k, v in overrides.items() if v is not None})
    return values


def settings_from_args(args: argparse.Namespace) -> PipelineSettings:
    lights = _collect_lights(args.lights)
    if not lights:
        raise SystemExit("No light frames found")
    mf = None
    mfv = _mf_values(args)
    if args.mfdeconv:
        mf = MFDeconvSettings(
            iterations=int(mfv["iterations"]),
            kappa=float(mfv["kappa"]),
            relax=float(mfv["relax"]),
            huber_delta=float(mfv["huber_delta"]),
            color_mode=args.mf_color,
            psf_size=args.mf_psf_size,
            use_star_masks=not args.mf_no_star_mask,
            use_variance_maps=not args.mf_no_variance,
            tile_size=args.mf_tile,
            dering_sigma=float(mfv["dering_sigma"]),
            early_stop_tol=float(mfv["early_stop_tol"]),
            save_psf_dir=args.mf_save_psfs,
        )
    return PipelineSettings(
        lights=lights,
        output=args.output,
        work_dir=args.work_dir,
        bias=args.bias,
        dark=args.dark,
        flat=args.flat,
        pedestal=args.pedestal,
        cosmetic=args.cosmetic,
        cosmetic_settings=CosmeticSettings(hot_sigma=args.hot_sigma, cold_sigma=args.cold_sigma, fix_hot=args.hot_sigma > 0, fix_cold=args.cold_sigma > 0),
        debayer=args.debayer,
        debayer_method=args.debayer_method,
        bayer_pattern=args.bayer_pattern,
        reference=args.reference,
        detection_sigma=args.sigma,
        weighting=args.weighting,
        weightings=[args.weighting, *args.also_weighting],
        filters=FilterSettings(enabled=not args.no_filters, transparency_min=args.filter_transparency, stars_min_ratio=args.filter_stars, background_max_ratio=args.filter_background, fwhm_max_ratio=args.filter_fwhm),
        interpolation=args.interp,
        refine_registration=not args.no_refine,
        local_norm=LocalNormSettings(enabled=not args.no_local_norm, block=args.local_norm_block),
        autocrop=args.autocrop,
        save_maps=not args.no_maps,
        plate_solve=not args.no_solve,
        astap_path=args.astap,
        drizzle=DrizzleSettings(enabled=args.drizzle > 0, scale=max(1, args.drizzle), kernel=args.drizzle_kernel, pixfrac=args.pixfrac, min_weight=args.drizzle_min_weight, cfa=not args.no_cfa_drizzle),
        stack=StackSettings(
            method=args.method,
            sigma_low=args.sigma_low,
            sigma_high=args.sigma_high,
            iterations=args.iterations,
            gesd_outliers=args.gesd_outliers,
            gesd_significance=args.gesd_significance,
            gesd_low_relax=args.gesd_low_relax,
            normalize=not args.no_normalize,
            large_scale=args.large_scale,
        ),
        mfdeconv=mf,
        mfdeconv_frames=int(mfv["frames"]),
        mfdeconv_blend=args.mf_blend,
        keep_registered=args.keep_registered,
        device="cpu" if args.cpu else "auto",
        vram_fraction=args.vram_fraction,
        workers=args.workers,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def status(msg: str) -> None:
        print(msg, flush=True)

    if args.command == "compare":
        if args.gpu is None or args.reference is None:
            from .compare_gui import main as compare_gui_main

            return compare_gui_main()
        from .compare import compare_many, write_report

        results = compare_many([args.gpu, *args.also], args.reference, status)
        for result in results:
            print(f"\n{Path(result.gpu).name} vs {Path(result.reference).name}")
            for line in result.summary():
                print(line)
        if args.json:
            write_report(results[0] if len(results) == 1 else results, args.json)
        return 0

    if args.command == "tilt":
        lights = _collect_lights(args.lights) if args.lights else []
        if not lights or args.gui:
            from .tilt_gui import main as tilt_gui_main

            return tilt_gui_main(lights or None)
        from .tilt import analyse_tilt, describe, frame_scatter, session_median, write_tilt_csv

        maps = []
        for i, p in enumerate(lights):
            m = analyse_tilt(p, args.cells)
            maps.append(m)
            print(f"[{i + 1}/{len(lights)}] {p.name}: FWHM {m.median_fwhm:.2f} px, tilt {m.tilt_px:.2f} px soft {m.soft_side}, curvature {m.curvature:+.2f} px", flush=True)
        session = session_median(maps) if len(maps) >= 2 else maps[0]
        print()
        for row in session.fwhm:
            print("    " + "  ".join("  n/a" if not np.isfinite(v) else f"{v:5.2f}" for v in row))
        print()
        for line in describe(session, frame_scatter(maps)):
            print(line)
        if args.csv:
            write_tilt_csv(maps + ([session] if len(maps) >= 2 else []), args.csv)
            print(f"Wrote {args.csv}")
        return 0

    if args.command == "mosaic":
        tiles = _collect_tiles(args.tiles) if args.tiles else []
        if not tiles or args.gui or args.output is None:
            from .mosaic_gui import main as mosaic_gui_main

            return mosaic_gui_main(tiles or None, args.output)
        from .mosaic import MosaicBuilder, MosaicSettings

        mosaic_settings = MosaicSettings(
            tiles=tiles,
            output=args.output,
            gradient_degree=args.gradient,
            gradient_block=args.gradient_block,
            pixel_scale=args.scale,
            orientation=args.orientation,
            interpolation=args.interp,
            refine=not args.no_refine,
            photometric=not args.no_photometric,
            match_gradient=not args.no_plane,
            feather=args.feather,
            seam_width=args.seam,
            blend_mode=args.blend,
            auto_crop=not args.no_autocrop,
            plate_solve=not args.no_solve,
            astap_path=args.astap,
            save_coverage=not args.no_maps,
            device="cpu" if args.cpu else "auto",
            vram_fraction=args.vram_fraction,
        )
        last_pct = {"pct": -1}

        def mosaic_progress(frac: float, msg: str) -> None:
            pct = int(frac * 100)
            if pct != last_pct["pct"]:
                last_pct["pct"] = pct
                print(f"  [{pct:3d}%] {msg}", flush=True)

        try:
            result = MosaicBuilder(mosaic_settings, status, mosaic_progress).run()
        except KeyboardInterrupt:
            print("Interrupted", file=sys.stderr)
            return 130
        print(f"Mosaic -> {result.output}")
        return 0

    settings = settings_from_args(args)

    last = {"pct": -1}

    def progress(frac: float, msg: str) -> None:
        pct = int(frac * 100)
        if pct != last["pct"]:
            last["pct"] = pct
            print(f"  [{pct:3d}%] {msg}", flush=True)

    if args.group_by or args.list_groups:
        keys = tuple(k for k in (args.group_by or ",".join(DEFAULT_GROUP_KEYS)).split(",") if k.strip())
        groups = group_frames(settings.lights, keys, status)
        for line in describe_groups(groups):
            print(line)
        if args.list_groups:
            return 0
        out_dir = args.output if args.output.suffix.lower() not in (".fit", ".fits", ".fts") else args.output.parent
        try:
            batch = run_batch(settings, groups, out_dir, args.name_template, status, progress)
        except KeyboardInterrupt:
            print("Interrupted", file=sys.stderr)
            return 130
        failed = [g for g, p, _ in batch.outputs if p is None]
        return 1 if failed and len(failed) == len(batch.outputs) else 0

    pipeline = StackingPipeline(settings, status, progress)
    try:
        result = pipeline.run()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    print(f"Finished in {result.seconds:.1f}s -> {result.output}")
    if result.drizzle_output:
        print(f"Drizzle -> {result.drizzle_output}")
    if result.mfdeconv_output:
        print(f"MFDeconv -> {result.mfdeconv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
