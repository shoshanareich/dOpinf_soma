#!/usr/bin/env python3
"""Export longitude-section (vertical) FOM/ROM fields for a completed ROM run into
``<data-dir>/analysis/computed_fields/`` as

    <field>_vertical_<label>.nc       e.g. salinity_vertical_tau0.2.nc

with data variables ``<field>_fom`` / ``<field>_rom`` on dims (time, k, j),
where ``k`` carries depth [m] and ``j`` latitude [degN] -- the same layout
``eval_rom_forcings.plot_vertical_temperature_profiles`` writes for temperature.

Reconstruction reuses the eval_rom_forcings machinery (basis lift + un-center),
so results match the eval plots.

Usage:
    python3 src/export_vertical_fields.py \
        --config <run>/config.json \
        --data-dir <results-run-subdir> \
        [--vars S]                    # comma list of state vars (T,S,U,V,Eta)
        [--root-dir /scratch2/shoshi/soma4/dOpInf_results/] \
        [--scenarios tau0.3,tau0.5]   # substring filter on FOM-dir names
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))  # core ROM modules
import paths  # noqa: E402
import eval_rom_forcings as E  # noqa: E402

# Output field name per state variable (temperature_* files already use this).
FIELD_NAMES = {"T": "temperature", "S": "salinity", "U": "u", "V": "v", "Eta": "eta"}


def resolve_args(config, root_dir, data_dir, vertical_i):
    """Reuse eval's config resolution to fill training/test dirs, labels, preproc."""
    sys.argv = ["export", "--config", config, "--root-dir", root_dir,
                "--data-dir", data_dir, "--outdir", data_dir,
                "--vertical-i", str(vertical_i)]
    args = E.parse_args()
    args.root_dir = E.ensure_slash(args.root_dir)
    args = E.apply_config_defaults(args)
    args.n_year_train = 2 if args.n_year_train is None else args.n_year_train
    args.n_year_predict = 0 if args.n_year_predict is None else args.n_year_predict
    args.n_days = 1 if args.n_days is None else args.n_days
    args.center_opt = args.center_opt or "mean"
    args.scale = args.scale or "maxabs"
    args.r = args.r or E.infer_r_from_tr(args.data_dir) or 26
    args.data_dir = E.ensure_slash(args.data_dir)
    args.preproc_dir = E.ensure_slash(args.preproc_dir)
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--root-dir", default=paths.RESULTS_ROOT)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--outdir", default=None,
                   help="default <data-dir>/analysis (files land in its computed_fields/)")
    p.add_argument("--vars", default="S", help="comma list of state variables (default S)")
    p.add_argument("--vertical-i", type=int, default=124, help="longitude index of the section")
    p.add_argument("--scenarios", default=None,
                   help="comma-separated substring filter on FOM-dir names")
    p.add_argument("--overwrite", action="store_true", help="rewrite existing output files")
    opts = p.parse_args()

    variables = [v.strip() for v in opts.vars.split(",") if v.strip()]
    unknown = [v for v in variables if v not in FIELD_NAMES]
    if unknown:
        p.error(f"unknown state variable(s) {unknown}; choose from {list(FIELD_NAMES)}")

    args = resolve_args(opts.config, opts.root_dir, opts.data_dir, opts.vertical_i)

    training_dirs = E.split_csv(args.training_dirs)
    labels = E.split_csv(args.scenario_labels)
    rom_files = E.split_csv(args.rom_files)
    test_dirs = E.split_csv(args.test_dirs) if args.test_dirs else (
        [args.test_dir] if args.test_dir else [])
    fom_dirs = training_dirs + test_dirs
    scenarios = [
        E.Scenario(label=lab, fom_dir=fd, rom_file=rf, is_training=i < len(training_dirs))
        for i, (lab, fd, rf) in enumerate(zip(labels, fom_dirs, rom_files))
    ]

    tr = np.load(Path(args.data_dir) / "Tr.npy")
    n_steps = (args.n_year_train + args.n_year_predict) * int(360 / args.n_days)

    field_dir = E.computed_fields_dir(Path(opts.outdir or (Path(args.data_dir) / "analysis")))
    print(f"Writing vertical sections (i={opts.vertical_i}) to {field_dir}", flush=True)

    depth = E.load_depth_values(args.grid_dir, args.nz)
    lat = E.load_lat_values(args.grid_dir, args.ny)

    cache: dict = {}  # basis lift is shared across scenarios (keyed by (var, i))
    filt = opts.scenarios.split(",") if opts.scenarios else None
    written: set = set()  # a tau listed in both train and test maps to one output

    for sc in scenarios:
        name = E.scenario_name_from_dir(sc.fom_dir)  # e.g. run_10yrspinup_tau0.3
        if filt and not any(f in name for f in filt):
            continue
        if name in written:
            print(f"  skip {name}: already exported (duplicate train/test scenario)", flush=True)
            continue
        qpath = Path(args.data_dir) / sc.rom_file
        if not qpath.exists():
            print(f"  skip {name}: {sc.rom_file} not found", flush=True)
            continue
        q_rom = np.load(qpath)[:, :n_steps]

        for var in variables:
            field = FIELD_NAMES[var]
            out = field_dir / f"{field}_vertical_{E.safe_stem(sc.label)}.nc"
            if out.exists() and not opts.overwrite:
                print(f"  skip {out.name}: exists (use --overwrite)", flush=True)
                continue
            print(f"  [{sc.label}] {var} section (q_rom {q_rom.shape})", flush=True)

            _, rom_3d = E.transform_and_project_lon_forcings(
                training_dirs, var, args.n_year_train, args.n_days, args.center_opt,
                args.preproc_dir, tr, q_rom, args.nx, args.ny, args.nz,
                i=opts.vertical_i, projection_cache=cache,
            )  # (nt, nz, ny)
            _, fom_3d = E.load_var_fom_lon(
                E.ensure_slash(sc.fom_dir), var, args.n_year_train, args.n_year_predict,
                args.n_days, args.preproc_dir, args.center_opt, args.scale,
                nx=args.nx, ny=args.ny, nz=args.nz, i=opts.vertical_i, anom=False,
            )
            fom_3d = fom_3d.where(fom_3d != 0)
            fom_3d, depth_dim = E.assign_depth_coord(fom_3d, depth)
            fom_3d, _ = E.assign_lat_coord(fom_3d, lat, depth_dim)
            nt = min(fom_3d.sizes["time"], rom_3d.shape[0])
            fom_3d = fom_3d.isel(time=slice(0, nt)).astype("float32")
            rom_da = xr.DataArray(
                rom_3d[:nt].astype("float32"), coords=fom_3d.coords, dims=fom_3d.dims
            ).where(rom_3d[:nt] != 0)

            xr.Dataset(
                {f"{field}_fom": fom_3d.rename(f"{field}_fom"),
                 f"{field}_rom": rom_da.rename(f"{field}_rom")},
                attrs={
                    "scenario": sc.label,
                    "vertical_i": opts.vertical_i,
                    "description": f"Longitude-section {field} fields (FOM and ROM reconstruction).",
                },
            ).to_netcdf(out, engine="h5netcdf")
            del rom_3d, rom_da, fom_3d
            print(f"  [{sc.label}] wrote {out.name}", flush=True)
        written.add(name)


if __name__ == "__main__":
    main()
