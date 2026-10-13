#!/usr/bin/env python3
"""MPI: project a scenario's initial condition into the reduced POD basis of a
completed dOpInf run, so a rollout can start from that scenario's true state.

    q0 = Tr^T @ (X_train^T @ IC)

X_train = centered+scaled stacked training snapshots (same pipeline the run used);
IC = centered+scaled first snapshot of the target scenario. Reuses the saved Tr.npy
(no SVD/eigh) and recomputes centers/alphas via shiftscale (deterministic), exactly
matching the run's preprocessing. No retraining.

Run under mpiexec with enough ranks/nodes to hold the training data (like the driver):
    mpiexec -np 62 python3 src/project_ic.py \
        --data-dir <run> --config <run>/config.json \
        --ic-dir /scratch/shoshi/soma4/run_20yrs_tau0.3/ \
        --out <run>/q0_run_20yrs_tau0.3.npy
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import dask.array as da
import xarray as xr
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import reshape_data, get_new_IC  # noqa: E402
from shiftscale import shiftscale  # noqa: E402


def _np(x):
    return x.compute() if hasattr(x, "compute") else np.asarray(x)


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ic-dir", required=True, help="snapshot dir of the target scenario")
    p.add_argument("--out", required=True)
    p.add_argument("--nx", type=int, default=248)
    p.add_argument("--ny", type=int, default=248)
    p.add_argument("--nz", type=int, default=31)
    a = p.parse_args()

    cfg = json.load(open(a.config))
    nyt = cfg["n_year_train"]
    nd = cfg["n_days"]
    center_opt = cfg["center_opt"]
    scale = cfg["scale"]
    prefix = cfg["snapshot_dir_prefix"]
    train_dirs = [f"{prefix}{t}/" for t in cfg["train_taus"]]

    if rank == 0:
        print(f"projecting IC from {a.ic_dir}\n  train dirs: {train_dirs}", flush=True)

    # load + stack training data (per-rank spatial slab), matching the driver
    Ul, Vl, El, Tl, Sl = [], [], [], [], []
    for d in train_dirs:
        ds = xr.open_dataset(d + "states_20yrs.nc", engine="h5netcdf",
                             decode_timedelta=False, decode_times=False, chunks={})
        U, V, E, T, S = reshape_data(a.nx, nd, nyt, ds, rank, size)
        Ul.append(U); Vl.append(V); El.append(E); Tl.append(T); Sl.append(S)
    # keep everything lazy (dask) -- do NOT materialize the full slabs (that OOMs);
    # only the small (nt,1) projection is computed at the end.
    U = da.hstack(Ul); V = da.hstack(Vl); E = da.hstack(El); T = da.hstack(Tl); S = da.hstack(Sl)

    kw = dict(comm=comm, center_type=center_opt, n_days=nd, scale_type=scale,
              nx=a.nx, ny=a.ny, nz=a.nz)
    Ut, cU, aU = shiftscale(U, nyt, **kw)
    Vt, cV, aV = shiftscale(V, nyt, **kw)
    Tt, cT, aT = shiftscale(T, nyt, **kw)
    St, cS, aS = shiftscale(S, nyt, **kw)
    Et, cE, aE = shiftscale(E, nyt, **kw)

    # stacked training matrix (space_slab, nt), in [U,V,Eta,T,S] order -- stays lazy
    Xr = da.vstack([Ut, Vt, Et, Tt, St])

    centers = [cU, cV, cE, cT, cS]  # get_new_IC indexes U,V,Eta,T,S
    alphas = {"U": aU, "V": aV, "Eta": aE, "T": aT, "S": aS}
    IC = _np(get_new_IC(a.ic_dir, centers, alphas, nyt, nd, rank, size, nx=a.nx))  # (space_slab, 1) small

    # dask streams the contraction over space in time-chunks -> low peak memory
    partial = np.ascontiguousarray(_np(Xr.T @ IC), dtype=np.float64)  # (nt, 1)
    g = np.zeros_like(partial)
    comm.Allreduce(partial, g, op=MPI.SUM)  # X_train^T @ IC over all ranks

    if rank == 0:
        Tr = np.load(Path(a.data_dir) / "Tr.npy")  # (nt, r)
        q0 = (Tr.T @ g).reshape(-1)  # (r,)
        np.save(a.out, q0)
        print(f"saved {a.out}  q0 shape={q0.shape}  finite={bool(np.isfinite(q0).all())}",
              flush=True)


if __name__ == "__main__":
    main()
