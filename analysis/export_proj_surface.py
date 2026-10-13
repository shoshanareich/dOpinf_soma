#!/usr/bin/env python3
"""Export the PROJECTION-ONLY surface state (FOM squeezed through the POD basis, no
dynamics) as ``<data-dir>/surface_fields/proj_surface_<scenario>.nc``, matching the
layout of the fom_/rom_ files written by export_surface_fields.py.

Coefficients come from either:
  * a training scenario -- its block of Qhat.npy is already the exact projection, so
    nothing needs recomputing (--from-qhat <tau>); or
  * a held-out scenario -- the cross-Gram projection produced by the MPI script
    src/project_scenario.py (--coeffs <run>/Q_PROJ_<scenario>.npy).

The lift back to physical space reuses eval_rom_forcings.transform_and_project_k_forcings,
i.e. exactly the same basis and centering the ROM reconstruction uses, so
fom / proj / rom are directly comparable.

Usage:
    python3 src/export_proj_surface.py --config <run>/config.json --data-dir <run> \
        --scenario run_10yrspinup_tau0.2 --from-qhat 0.2
    python3 src/export_proj_surface.py --config <run>/config.json --data-dir <run> \
        --scenario run_10yrspinup_tau0.3 --coeffs <run>/Q_PROJ_run_10yrspinup_tau0.3.npy
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))  # core ROM modules
import paths  # noqa: E402
import eval_rom_forcings as E  # noqa: E402
from export_surface_fields import resolve_args  # noqa: E402

VARS = ["U", "V", "Eta", "T", "S"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--root-dir", default=paths.RESULTS_ROOT)
    p.add_argument("--scenario", required=True, help="e.g. run_10yrspinup_tau0.3")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--coeffs", help="Q_PROJ_*.npy from src/project_scenario.py")
    g.add_argument("--from-qhat", help="training tau whose Qhat block to use, e.g. 0.2")
    opts = p.parse_args()

    args = resolve_args(opts.config, opts.root_dir, opts.data_dir)
    cfg = json.load(open(opts.config))
    training_dirs = E.split_csv(args.training_dirs)
    tr = np.load(Path(args.data_dir) / "Tr.npy")
    n_steps = (args.n_year_train + args.n_year_predict) * int(360 / args.n_days)

    if opts.coeffs:
        q = np.load(opts.coeffs)[:, :n_steps]
        src = opts.coeffs
    else:
        taus = [str(t) for t in cfg["train_taus"]]
        if opts.from_qhat not in taus:
            raise SystemExit(f"--from-qhat {opts.from_qhat} not in train_taus {taus}")
        i = taus.index(opts.from_qhat)
        nts = args.n_year_train * int(360 / args.n_days)
        q = np.load(Path(args.data_dir) / "Qhat.npy")[:, i * nts:(i + 1) * nts]
        src = f"Qhat.npy block {i} (tau {opts.from_qhat})"
    print(f"[{opts.scenario}] coefficients from {src}, shape {q.shape}", flush=True)

    cache, out_vars = {}, {}
    nt = q.shape[1]
    for v in VARS:
        _, f = E.transform_and_project_k_forcings(
            training_dirs, v, args.n_year_train, args.n_days, args.center_opt,
            args.preproc_dir, tr, q, args.nx, args.ny, k=0, projection_cache=cache)
        out_vars[v] = f.astype("float32", copy=False)
        nt = min(nt, f.shape[0])
        print(f"  {v} lifted", flush=True)

    outdir = Path(args.data_dir) / "surface_fields"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"proj_surface_{opts.scenario}.nc"
    xr.Dataset(
        {v: (("time", "y", "x"), out_vars[v][:nt]) for v in VARS},
        coords={"time": np.arange(nt), "day": ("time", np.arange(nt) * args.n_days)},
        attrs={"scenario": opts.scenario, "kind": "POD projection (no dynamics)",
               "note": "surface (k=0) state, raw physical units", "coeffs": src},
    ).to_netcdf(path, engine="h5netcdf")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
