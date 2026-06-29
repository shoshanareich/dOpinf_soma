#!/usr/bin/env python3
"""Save surface (level k=0) fields for ALL variables (U, V, Eta, T, S) to netCDF,
for every train/test scenario of a ROM run, plus the matching FOM fields.

Reuses the reconstruction helpers in eval_rom_forcings.py:
  - ROM:  transform_and_project_k_forcings  (reduced coords -> physical surface field)
  - FOM:  load_var_fom_k                     (raw MITgcm surface field)

Outputs one netCDF per scenario (all 5 vars stacked as (time, y, x)):
  <outdir>/rom_surface_<scenario>.nc
  <outdir>/fom_surface_<scenario>.nc

Usage (script / sbatch):
  python3 src/save_surface_fields.py \
      --data-dir /scratch2/.../save_roms/taus_2_4_sameIC/mean_var_1days_10yrs_r105/

Usage (notebook):
  from save_surface_fields import build_rom_surface, build_fom_surface, resolve_run
  cfg = resolve_run(data_dir)              # dict of run settings + scenarios
  ds  = build_rom_surface(cfg, "run_10yrspinup_tau0.2", train=True)

Note on staggering: U is on (j, i_g) and V on (j_g, i) in MITgcm; like the
evaluator, this tool stores every variable on the shared (y, x) index grid
(no de-staggering / interpolation). Fine for fields/variability; if you need
exact C-grid positions, regrid downstream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from config_utils import ensure_slash, resolve_forcing_config, infer_tau_from_dir
from eval_rom_forcings import (
    transform_and_project_k_forcings,
    load_var_fom_k,
    scenario_name_from_dir,
    infer_r_from_tr,
)

VARS = ["U", "V", "Eta", "T", "S"]
DEFAULT_FOM_DIRS = [f"/scratch/shoshi/soma4/run_10yrspinup_tau0.{i}" for i in (1, 2, 3, 4, 5)]


def resolve_run(data_dir: str, config_path: str | None = None,
                root_dir: str = "/scratch2/shoshi/soma4/dOpInf_results/") -> dict:
    """Collect everything needed to reconstruct fields from a ROM results dir."""
    data_dir = ensure_slash(data_dir)
    config_path = config_path or (data_dir + "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    fc = resolve_forcing_config(cfg)
    r = cfg.get("r") or infer_r_from_tr(data_dir)
    dir_extension = cfg.get("dir_extension", "")
    preproc_dir = ensure_slash(str(Path(ensure_slash(root_dir)) / "save_roms" / dir_extension))

    # scenarios: (label, fom_dir, rom_file, is_train)
    scenarios = []
    for d in fc["snapshot_dirs"]:
        name = scenario_name_from_dir(d)
        scenarios.append((name, ensure_slash(d), f"Q_ROM_train_{name}.npy", True))
    for d in fc["test_dirs"]:
        name = scenario_name_from_dir(d)
        scenarios.append((name, ensure_slash(d), f"Q_ROM_test_{name}.npy", False))

    return {
        "data_dir": data_dir,
        "preproc_dir": preproc_dir,
        "training_dirs": [ensure_slash(d) for d in fc["snapshot_dirs"]],
        "tr": np.load(data_dir + "Tr.npy"),
        "center_opt": cfg["center_opt"],
        "scale": cfg["scale"],
        "n_days": int(cfg["n_days"]),
        "n_year_train": int(cfg["n_year_train"]),
        "n_year_predict": int(cfg.get("n_year_predict", 0)),
        "r": int(r),
        "nx": int(cfg.get("nx", 248)),
        "ny": int(cfg.get("ny", 248)),
        "scenarios": scenarios,
    }


def _coords(nt: int, ny: int, nx: int, n_days: int) -> dict:
    return {
        "time": np.arange(nt),
        "day": ("time", np.arange(nt) * n_days),  # model days, 360-day calendar
        "y": np.arange(ny),
        "x": np.arange(nx),
    }


def _dataset(data_vars: dict, nt, ny, nx, n_days, *, label, source, tau, dtype):
    coords = _coords(nt, ny, nx, n_days)
    ds = xr.Dataset(
        {v: (("time", "y", "x"), a.astype(dtype)) for v, a in data_vars.items()},
        coords=coords,
        attrs={"scenario": label, "source": source, "tau": "" if tau is None else tau,
               "n_days": n_days, "calendar_days_per_year": 360, "level": "surface (k=0)"},
    )
    return ds


def build_rom_surface(cfg: dict, label: str, *, train: bool, dtype="float32",
                      projection_cache: dict | None = None) -> xr.Dataset:
    """Reconstruct all VARS at the surface for one ROM scenario -> Dataset."""
    rom_file = f"Q_ROM_{'train' if train else 'test'}_{label}.npy"
    q_rom = np.load(cfg["data_dir"] + rom_file)
    if projection_cache is None:
        projection_cache = {}
    data_vars = {}
    for var in VARS:
        _, rom3d = transform_and_project_k_forcings(
            cfg["training_dirs"], var, cfg["n_year_train"], cfg["n_days"],
            cfg["center_opt"], cfg["preproc_dir"], cfg["tr"], q_rom,
            cfg["nx"], cfg["ny"], k=0, projection_cache=projection_cache,
        )
        data_vars[var] = np.asarray(rom3d)
    nt = q_rom.shape[1]
    return _dataset(data_vars, nt, cfg["ny"], cfg["nx"], cfg["n_days"],
                    label=label, source="ROM", tau=infer_tau_from_dir(label), dtype=dtype)


def build_fom_surface(cfg: dict, fom_dir: str, *, dtype="float32") -> xr.Dataset:
    """Load all VARS at the surface for one FOM run -> Dataset."""
    fom_dir = ensure_slash(fom_dir)
    label = scenario_name_from_dir(fom_dir)
    data_vars = {}
    nt = None
    for var in VARS:
        _, fom3d = load_var_fom_k(
            fom_dir, var, cfg["n_year_train"], cfg["n_year_predict"], cfg["n_days"],
            cfg["preproc_dir"], cfg["center_opt"], cfg["scale"],
            nx=cfg["nx"], ny=cfg["ny"], k=0, anom=False,
        )
        arr = np.asarray(fom3d.values)  # (time, j, i) -> shared (time, y, x) index grid
        data_vars[var] = arr
        nt = arr.shape[0]
    return _dataset(data_vars, nt, cfg["ny"], cfg["nx"], cfg["n_days"],
                    label=label, source="FOM", tau=infer_tau_from_dir(label), dtype=dtype)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="ROM results dir (has Tr.npy, Q_ROM_*.npy, config.json)")
    p.add_argument("--config", default=None, help="config.json (default: <data-dir>/config.json)")
    p.add_argument("--root-dir", default="/scratch2/shoshi/soma4/dOpInf_results/")
    p.add_argument("--outdir", default=None, help="output dir (default: <data-dir>/surface_fields/)")
    p.add_argument("--fom-dirs", default=",".join(DEFAULT_FOM_DIRS),
                   help="comma-separated FOM run dirs (default: run_10yrspinup_tau0.1..0.5)")
    p.add_argument("--float64", action="store_true", help="store float64 instead of float32")
    p.add_argument("--skip-rom", action="store_true")
    p.add_argument("--skip-fom", action="store_true")
    args = p.parse_args()

    cfg = resolve_run(args.data_dir, args.config, args.root_dir)
    dtype = "float64" if args.float64 else "float32"
    outdir = Path(args.outdir or (cfg["data_dir"] + "surface_fields"))
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"center={cfg['center_opt']} scale={cfg['scale']} r={cfg['r']} "
          f"n_days={cfg['n_days']} n_year_train={cfg['n_year_train']} "
          f"n_year_predict={cfg['n_year_predict']}")
    print(f"writing to {outdir} as {dtype}")

    if not args.skip_rom:
        projection_cache: dict = {}  # lifted bases are scenario-independent -> reuse
        for label, _fom_dir, rom_file, train in cfg["scenarios"]:
            if not (Path(cfg["data_dir"]) / rom_file).exists():
                print(f"  [skip ROM] {rom_file} not found")
                continue
            ds = build_rom_surface(cfg, label, train=train, dtype=dtype,
                                   projection_cache=projection_cache)
            out = outdir / f"rom_surface_{label}.nc"
            ds.to_netcdf(out, engine="h5netcdf")
            print(f"  [ROM] {label:30s} {dict(ds.sizes)} -> {out.name}")

    if not args.skip_fom:
        for fom_dir in args.fom_dirs.split(","):
            fom_dir = fom_dir.strip()
            if not (Path(ensure_slash(fom_dir)) / "states_20yrs.nc").exists():
                print(f"  [skip FOM] {fom_dir}/states_20yrs.nc not found")
                continue
            ds = build_fom_surface(cfg, fom_dir, dtype=dtype)
            out = outdir / f"fom_surface_{scenario_name_from_dir(fom_dir)}.nc"
            ds.to_netcdf(out, engine="h5netcdf")
            print(f"  [FOM] {scenario_name_from_dir(fom_dir):30s} {dict(ds.sizes)} -> {out.name}")

    print("done.")


if __name__ == "__main__":
    main()
