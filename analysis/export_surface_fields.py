#!/usr/bin/env python3
"""Export full surface state (U, V, Eta, T, S) for a completed ROM run into
``<data-dir>/surface_fields/`` as split files:

    rom_surface_<scenario>.nc   (ROM reconstruction, raw physical units)
    fom_surface_<scenario>.nc   (FOM truth, raw physical units)

Each file holds the 5 state variables at the surface (level k=0) with dims
(time, y, x), matching the older eval output format. Reconstruction reuses the
eval_rom_forcings machinery (basis lift + un-center) so results match the plots.

Usage:
    python3 src/export_surface_fields.py \
        --config configs_forcings/<cfg>.json \
        --data-dir <results-run-subdir> \
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

VARS = ["U", "V", "Eta", "T", "S"]


def resolve_args(config, root_dir, data_dir):
    """Reuse eval's config resolution to fill training/test dirs, labels, preproc."""
    sys.argv = ["export", "--config", config, "--root-dir", root_dir,
                "--data-dir", data_dir, "--outdir", data_dir]
    args = E.parse_args()
    args.root_dir = E.ensure_slash(args.root_dir)
    args = E.apply_config_defaults(args)
    # mirror main() fallbacks (the config supplies these, but stay safe)
    args.n_year_train = 2 if args.n_year_train is None else args.n_year_train
    args.n_year_predict = 0 if args.n_year_predict is None else args.n_year_predict
    args.n_days = 1 if args.n_days is None else args.n_days
    args.center_opt = args.center_opt or "mean"
    args.scale = args.scale or "maxabs"
    args.r = args.r or E.infer_r_from_tr(args.data_dir) or 26
    args.data_dir = E.ensure_slash(args.data_dir)
    args.preproc_dir = E.ensure_slash(args.preproc_dir)
    return args


def to_time_yx(da: xr.DataArray, nt: int) -> np.ndarray:
    """Return (time, y, x) float32 values regardless of input dim order/names."""
    other = [d for d in da.dims if d != "time"]
    arr = np.asarray(da.transpose("time", *other).values)[:nt]
    return arr.astype("float32")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--root-dir", default=paths.RESULTS_ROOT)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--scenarios", default=None,
                   help="comma-separated substring filter on FOM-dir names")
    opts = p.parse_args()

    args = resolve_args(opts.config, opts.root_dir, opts.data_dir)

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

    outdir = Path(args.data_dir) / "surface_fields"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing split surface fields to {outdir}", flush=True)

    cache: dict = {}  # basis lift is shared across scenarios (keyed by var)
    filt = opts.scenarios.split(",") if opts.scenarios else None
    written: set = set()  # guard against train/test duplicates mapping to one name

    for sc in scenarios:
        name = E.scenario_name_from_dir(sc.fom_dir)  # e.g. run_10yrspinup_tau0.3
        if filt and not any(f in name for f in filt):
            continue
        if name in written:
            # e.g. a tau appears in both train_taus and test_taus; the training
            # rollout (processed first) is the primary one -- skip the duplicate.
            print(f"  skip {name}: already exported (duplicate train/test scenario)", flush=True)
            continue
        qpath = Path(args.data_dir) / sc.rom_file
        if not qpath.exists():
            print(f"  skip {name}: {sc.rom_file} not found", flush=True)
            continue
        q_rom = np.load(qpath)[:, :n_steps]
        print(f"  [{name}] reconstructing {VARS} (q_rom {q_rom.shape})", flush=True)

        rom_vars, fom_vars = {}, {}
        nt = q_rom.shape[1]
        for v in VARS:
            _, rom_f = E.transform_and_project_k_forcings(
                training_dirs, v, args.n_year_train, args.n_days, args.center_opt,
                args.preproc_dir, tr, q_rom, args.nx, args.ny, k=0,
                projection_cache=cache,
            )  # (nt, ny, nx)
            _, fom_da = E.load_var_fom_k(
                E.ensure_slash(sc.fom_dir), v, args.n_year_train, args.n_year_predict,
                args.n_days, args.preproc_dir, args.center_opt, args.scale,
                nx=args.nx, ny=args.ny, k=0,
            )  # DataArray with a 'time' dim
            nt = min(nt, rom_f.shape[0], fom_da.sizes["time"])
            # store float32 (halves peak memory) and drop the float64 / DataArray refs
            rom_vars[v] = rom_f.astype("float32", copy=False)
            fom_vars[v] = to_time_yx(fom_da, fom_da.sizes["time"])
            del rom_f, fom_da

        coords = {"time": np.arange(nt), "day": ("time", np.arange(nt) * args.n_days)}
        # write ROM, then free it before building the FOM dataset to cap peak memory
        xr.Dataset(
            {v: (("time", "y", "x"), rom_vars[v][:nt]) for v in VARS},
            coords=coords,
            attrs={"scenario": name, "kind": "ROM reconstruction",
                   "note": "surface (k=0) state, raw physical units"},
        ).to_netcdf(outdir / f"rom_surface_{name}.nc", engine="h5netcdf")
        del rom_vars
        xr.Dataset(
            {v: (("time", "y", "x"), fom_vars[v][:nt]) for v in VARS},
            coords=coords,
            attrs={"scenario": name, "kind": "FOM",
                   "note": "surface (k=0) state, raw physical units"},
        ).to_netcdf(outdir / f"fom_surface_{name}.nc", engine="h5netcdf")
        del fom_vars
        written.add(name)
        print(f"  [{name}] wrote rom_surface_{name}.nc + fom_surface_{name}.nc", flush=True)


if __name__ == "__main__":
    main()
