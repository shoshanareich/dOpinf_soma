#!/usr/bin/env python3
"""MPI: project a scenario's FULL snapshot record onto a completed run's POD basis.

This is the projection-only baseline: the FOM squeezed through the r-dimensional POD
basis with NO dynamics.  Comparing FOM / projection / ROM separates the two error
sources -- FOM vs projection is basis truncation (set by r), projection vs ROM is the
learned dynamics.

Method of snapshots, with the run's own convention (svd.py):

    Tr    = eigv Lambda^{-1/2}          (saved as Tr.npy,  nt_train x r)
    V     = X_train Tr                  (the spatial POD modes, never formed)
    a     = V^T X_test = Tr^T (X_train^T X_test)

so only the cross-Gram X_train^T X_test is distributed; it is reduced over ranks
exactly like the training Gram in svd.py.  For a training scenario ``a`` must come
back equal to that scenario's block of Qhat.npy -- pass --check to verify it.

X_train and X_test are centered and scaled with the SAME centers/alphas, recomputed
via shiftscale from the training data (deterministic, matches the run).

Run under mpiexec with the same rank count as the driver:
    mpiexec -np 62 python3 src/project_scenario.py \
        --data-dir <run> --config <run>/config.json \
        --scenario-dir /scratch/shoshi/soma4/run_10yrspinup_tau0.3/
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import dask.array as da
import xarray as xr
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import reshape_data  # noqa: E402
from shiftscale import shiftscale  # noqa: E402


def cross_gram(Xa, Xb, chunk_rows):
    """(Xa^T Xb) for this rank's slab, streamed over row (space) chunks."""
    if not hasattr(Xa, "chunks"):
        return Xa.T @ Xb
    Xa = Xa.rechunk({0: chunk_rows})
    Xb = Xb.rechunk({0: chunk_rows})
    out = None
    for i in range(Xa.numblocks[0]):
        a = np.asarray(Xa.blocks[i, :].compute())
        b = np.asarray(Xb.blocks[i, :].compute())
        g = a.T @ b
        out = g if out is None else out + g
    return out


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--scenario-dir", required=True, help="snapshot dir of the target scenario")
    p.add_argument("--out", default=None, help="defaults to <data-dir>/Q_PROJ_<scenario>.npy")
    p.add_argument("--chunk-rows", type=int, default=None,
                   help="space rows per cross-Gram chunk (default: config svd_gram_chunk_rows)")
    p.add_argument("--check", action="store_true",
                   help="if the scenario is a training tau, compare against Qhat.npy")
    p.add_argument("--nx", type=int, default=248)
    p.add_argument("--ny", type=int, default=248)
    p.add_argument("--nz", type=int, default=31)
    a = p.parse_args()

    cfg = json.load(open(a.config))
    nyt, nd = cfg["n_year_train"], cfg["n_days"]
    prefix = cfg["snapshot_dir_prefix"]
    train_dirs = [f"{prefix}{t}/" for t in cfg["train_taus"]]
    chunk_rows = a.chunk_rows or int(cfg.get("svd_gram_chunk_rows", 20000))
    scen_dir = a.scenario_dir.rstrip("/") + "/"
    name = Path(scen_dir.rstrip("/")).name
    out = Path(a.out or (Path(a.data_dir) / f"Q_PROJ_{name}.npy"))

    if rank == 0:
        print(f"projecting {name}\n  train dirs: {train_dirs}\n  chunk_rows={chunk_rows}",
              flush=True)
    t0 = time.perf_counter()

    def load(dirs):
        cols = [[], [], [], [], []]
        for d in dirs:
            ds = xr.open_dataset(d + "states_20yrs.nc", engine="h5netcdf",
                                 decode_timedelta=False, decode_times=False, chunks={})
            for lst, arr in zip(cols, reshape_data(a.nx, nd, nyt, ds, rank, size)):
                lst.append(arr)
        return [da.hstack(c) for c in cols]

    U, V, E, T, S = load(train_dirs)
    kw = dict(comm=comm, center_type=cfg["center_opt"], n_days=nd,
              scale_type=cfg["scale"], nx=a.nx, ny=a.ny, nz=a.nz)
    Ut, cU, aU = shiftscale(U, nyt, **kw)
    Vt, cV, aV = shiftscale(V, nyt, **kw)
    Tt, cT, aT = shiftscale(T, nyt, **kw)
    St, cS, aS = shiftscale(S, nyt, **kw)
    Et, cE, aE = shiftscale(E, nyt, **kw)
    Xtrain = da.vstack([Ut, Vt, Et, Tt, St])

    # the target scenario, centered and scaled with the TRAINING statistics
    Un, Vn, En, Tn, Sn = load([scen_dir])
    cen = [(Un, cU, aU), (Vn, cV, aV), (En, cE, aE), (Tn, cT, aT), (Sn, cS, aS)]
    Xtest = da.vstack([(x - c[:, np.newaxis]) / al for x, c, al in cen])
    if rank == 0:
        print(f"  slabs ready ({time.perf_counter() - t0:.0f}s): "
              f"train {Xtrain.shape}, test {Xtest.shape}", flush=True)

    partial = np.ascontiguousarray(cross_gram(Xtrain, Xtest, chunk_rows), dtype=np.float64)
    g = np.empty_like(partial) if rank == 0 else None
    comm.Reduce(partial, g, op=MPI.SUM, root=0)
    del partial

    if rank == 0:
        Tr = np.load(Path(a.data_dir) / "Tr.npy")          # (nt_train, r)
        aproj = Tr.T @ g                                    # (r, nt_test)
        np.save(out, aproj)
        print(f"saved {out}  shape={aproj.shape}  finite={bool(np.isfinite(aproj).all())}  "
              f"({time.perf_counter() - t0:.0f}s total)", flush=True)
        if a.check:
            taus = [str(t) for t in cfg["train_taus"]]
            which = [i for i, t in enumerate(taus) if name.endswith(t)]
            if which:
                nts = aproj.shape[1]
                blk = np.load(Path(a.data_dir) / "Qhat.npy")[:, which[0] * nts:(which[0] + 1) * nts]
                rel = np.linalg.norm(aproj - blk) / np.linalg.norm(blk)
                print(f"  check vs Qhat block {which[0]}: relative error {rel:.3e}", flush=True)
            else:
                print(f"  --check: {name} is not a training scenario; nothing to compare",
                      flush=True)


if __name__ == "__main__":
    main()
