#!/usr/bin/env python3
"""MPI: compute the barotropic streamfunction (psi_rom_*.nc) and u_bt (u_bt_rom_*.nc)
for ONE already-rolled-out scenario, reusing the driver's exact machinery. Use to
backfill psi for a scenario whose Q_ROM was produced outside the driver (e.g. via
predict_scenario.py, which does not compute streamfunction).

Mirrors mixed_rom_forcings.py's streamfunction step: raw training U slab + centerU
(from shiftscale) + saved Tr.npy + the scenario's Q_ROM -> compute_barotropic_streamfunction.

--rom-file/--output-label may be repeated to do several scenarios in one job. The
training U slab and its center are read once and reused, which is the dominant cost,
so N scenarios in one invocation are much cheaper than N invocations.

Run under mpiexec (same layout as the training driver):
    mpiexec -np 62 python3 src/compute_psi_scenario.py \
        --data-dir <run> --config <run>/config.json \
        --rom-file Q_ROM_test_run_20yrs_tau0.3.npy \
        --output-label Q_ROM_test_run_20yrs_tau0.3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))  # core ROM modules
import paths  # noqa: E402
from utils import reshape_data, read_grid, compute_barotropic_streamfunction  # noqa: E402
from shiftscale import shiftscale  # noqa: E402


class _SVDShim:
    """Minimal stand-in exposing the attributes compute_barotropic_streamfunction uses."""
    def __init__(self, tr, r):
        self.Tr_global = tr
        self.r = r


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--rom-file", required=True, action="append",
                   help="repeatable; pair with one --output-label each")
    p.add_argument("--output-label", required=True, action="append")
    p.add_argument("--grid-dir", default=paths.GRID_DIR)
    p.add_argument("--nx", type=int, default=248)
    p.add_argument("--ny", type=int, default=248)
    p.add_argument("--nz", type=int, default=31)
    a = p.parse_args()

    cfg = json.load(open(a.config))
    nyt = cfg["n_year_train"]
    nd = cfg["n_days"]
    center_opt = cfg["center_opt"]
    scale = cfg["scale"]
    r = cfg["r"]
    prefix = cfg["snapshot_dir_prefix"]
    train_dirs = [f"{prefix}{t}/" for t in cfg["train_taus"]]
    D = Path(a.data_dir)

    if len(a.rom_file) != len(a.output_label):
        raise SystemExit(f"--rom-file given {len(a.rom_file)}x but --output-label "
                         f"{len(a.output_label)}x; they must pair up")
    missing = [f for f in a.rom_file if not (D / f).exists()]
    if missing:
        raise SystemExit(f"missing ROM file(s) in {D}: {missing}")

    if rank == 0:
        print(f"psi for {len(a.rom_file)} scenario(s) {a.output_label}: "
              f"train {train_dirs}, nyt={nyt}, r={r}", flush=True)

    # raw training U slab (hstack over training scenarios), matching the driver
    Ul = []
    for d in train_dirs:
        ds = xr.open_dataset(d + "states_20yrs.nc", engine="h5netcdf",
                             decode_timedelta=False, decode_times=False, chunks={})
        U, _, _, _, _ = reshape_data(a.nx, nd, nyt, ds, rank, size)
        Ul.append(U)
    U_rank = np.hstack(Ul)  # RAW (materialized), as the driver passes it

    # centerU from shiftscale (discard the transformed copy); U_rank stays raw
    _, centerU, _ = shiftscale(U_rank, nyt, comm=comm, center_type=center_opt,
                               n_days=nd, scale_type=scale, nx=a.nx, ny=a.ny, nz=a.nz)

    svd = _SVDShim(np.load(D / "Tr.npy"), r)
    grid = read_grid(comm, a.grid_dir)

    # U_rank/centerU are scenario-independent, so reuse them across the whole list.
    # compute_barotropic_streamfunction does not mutate either (it rebinds its own
    # float32 views), so repeated calls are safe.
    for rom_file, label in zip(a.rom_file, a.output_label):
        if rank == 0:
            print(f"--- {label} (from {rom_file})", flush=True)
        Q_ROM_val = np.load(D / rom_file) if rank == 0 else None
        if rank == 0 and not np.isfinite(Q_ROM_val).all():
            n_ok = int(np.isfinite(Q_ROM_val).all(axis=0).sum())
            print(f"    WARNING: {label} is non-finite; last good step "
                  f"{n_ok}/{Q_ROM_val.shape[1]} (~yr {n_ok * nd / 360:.2f}). "
                  f"psi will inherit the NaNs.", flush=True)
        compute_barotropic_streamfunction(
            comm=comm, grid=grid, U_rank=U_rank, centerU=centerU, svd=svd,
            Q_ROM_val=Q_ROM_val, nx=a.nx, ny=a.ny, nz=a.nz, root_dir=str(D),
            center_opt=center_opt, scale_type=scale, n_year_train=nyt,
            output_label=label,
        )
        if rank == 0:
            print(f"done psi/u_bt for {label}", flush=True)


if __name__ == "__main__":
    main()
