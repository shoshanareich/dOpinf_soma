#!/usr/bin/env python3
"""MPI: meridional overturning streamfunction (FOM vs ROM) per scenario -> moc_tau{T}.nc

Convention (matches the existing taus_2_4_sameIC/r105 moc files):
    psi(k, j) = cumsum_over_depth_from_surface( sum_i V * hFacS * DRF * DXG ) / 1e6   [Sv]
    AMOC strength at latitude L = psi.sel(lat=L, method='nearest').max('depth')

The ROM side never forms the full 3D state. The zonal+depth integral W is a linear
functional of the *rows* of the snapshot matrix and the POD lift acts on the *columns*,
so the two commute:

    W[ (V_train - c) @ Tr @ Q_ROM + c ] = W[V_train - c] @ Tr @ Q_ROM + W[c]

W[V_train - c] is only (nz*ny) x nt_train, so the lift is done on a tiny matrix. The
per-point scaling alpha cancels row-by-row (same argument as
utils.compute_barotropic_streamfunction / utils.transform_and_project_k), which is why
the raw (uncentered, unscaled) training slab is used here.

Unlike the training driver, ranks are split over TIME, not longitude: the zonal integral
consumes every i anyway, and MITgcm snapshot files are contiguous in (k, j, i), so a
per-rank time block reads whole 3D snapshots sequentially instead of strided 4-column
slivers. Each rank's partial contribution is summed/gathered over the time axis.

Run under mpiexec:

    mpiexec -np 62 python3 src/compute_moc.py \
        --data-dir /scratch2/.../save_roms/taus_2_4_equilibrium/mean_var_1days_10yrs_r90 \
        --taus 0.2 0.3 0.4 0.5
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))  # core ROM modules
import paths  # noqa: E402
from utils import slice_space  # noqa: E402
from MITgcmutils import rdmds  # noqa: E402


def read_moc_grid(comm, grid_path):
    """hFacS/DRF/DXG for the transport weights, RC/YG for the output coords."""
    rank = comm.Get_rank()
    out = {}
    for key in ('hFacS', 'DRF', 'DXG', 'RC', 'YG'):
        data = np.squeeze(rdmds(grid_path + key)) if rank == 0 else None
        out[key] = comm.bcast(data, root=0)
        del data
        gc.collect()
    return out


def load_center(center_file, nx, ny, nz):
    """Saved center in the driver's global (k, j, i) layout."""
    with xr.open_dataset(center_file, engine="h5netcdf") as ds:
        nz_saved = int(ds.attrs.get('nz', nz))
        center = ds['center'].values
    if center.ndim > 1 and center.shape[1] == 1:
        center = center[:, 0]
    if center.ndim != 1:
        raise NotImplementedError(
            f"{center_file} holds a time-varying center (shape {center.shape}); only "
            "time-constant centers (center_opt in mean/IC/global_mean) are supported here."
        )
    return center.reshape((nz_saved, ny, nx)).astype(np.float32)


def zonal_transport(path, t_lo, t_hi, n_days, weights, center, chunk_t=30):
    """Zonally integrated meridional transport for snapshots [t_lo, t_hi).

    Returns (nt_local, nz, ny) float32 = sum_i V*hFacS*DRF*DXG, with `center`
    (or None) subtracted from V first.
    """
    nz, ny, nx = weights.shape
    nt_local = t_hi - t_lo
    out = np.empty((nt_local, nz, ny), dtype=np.float32)
    if nt_local == 0:
        return out

    with h5py.File(path, 'r') as f:
        v = f['V']
        n_avail = v.shape[0]
        if (t_hi - 1) * n_days + 1 > n_avail:
            raise ValueError(f"{path} has {n_avail} snapshots, need {(t_hi - 1) * n_days + 1}")
        for s in range(0, nt_local, chunk_t):
            e = min(s + chunk_t, nt_local)
            # contiguous whole-snapshot reads, strided in time only
            blk = v[(t_lo + s) * n_days:(t_lo + e) * n_days:n_days].astype(np.float32, copy=False)
            if center is not None:
                blk = blk - center[None]
            out[s:e] = np.einsum('tzyx,zyx->tzy', blk, weights, optimize=True)
            del blk
    gc.collect()
    return out


def gather_time(comm, local, nt, nz, ny):
    """Gather (nt_local, nz, ny) float32 blocks along time onto rank 0 -> (nt, nz, ny)."""
    rank = comm.Get_rank()
    counts = np.array(comm.allgather(local.shape[0] * nz * ny), dtype=int)
    full = np.empty((nt, nz, ny), dtype=np.float32) if rank == 0 else None
    comm.Gatherv(np.ascontiguousarray(local),
                 [full, counts, np.insert(np.cumsum(counts)[:-1], 0, 0), MPI.FLOAT] if rank == 0 else None,
                 root=0)
    return full


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="ROM results dir holding Tr.npy / Q_ROM_*.npy")
    p.add_argument("--config", default=None, help="defaults to <data-dir>/config.json")
    p.add_argument("--out-dir", default=None, help="defaults to <data-dir>/analysis/moc")
    p.add_argument("--center-dir", default=None,
                   help="dir holding centerV_*.nc; defaults to the parent of --data-dir")
    p.add_argument("--taus", nargs="+", default=["0.2", "0.3", "0.4", "0.5"])
    p.add_argument("--nt", type=int, default=None,
                   help="snapshots to write; default 360*n_year_train/n_days (capped per tau by its rollout)")
    p.add_argument("--grid-dir", default=paths.GRID_DIR)
    p.add_argument("--nx", type=int, default=248)
    p.add_argument("--ny", type=int, default=248)
    p.add_argument("--nz", type=int, default=31)
    a = p.parse_args()

    D = Path(a.data_dir)
    cfg = json.load(open(a.config or D / "config.json"))
    nyt, nd = cfg["n_year_train"], cfg["n_days"]
    center_opt, scale, r = cfg["center_opt"], cfg["scale"], cfg["r"]
    prefix, train_taus = cfg["snapshot_dir_prefix"], cfg["train_taus"]
    scen = Path(prefix).name                       # e.g. "run_20yrs_tau"
    nt_default = a.nt or (360 * nyt // nd)

    out_dir = Path(a.out_dir) if a.out_dir else D / "analysis" / "moc"
    center_dir = Path(a.center_dir) if a.center_dir else D.parent
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[moc] {D}\n[moc] out -> {out_dir}\n[moc] taus {a.taus}, nt<={nt_default}", flush=True)

    nx, ny, nz = a.nx, a.ny, a.nz
    grid = read_moc_grid(comm, a.grid_dir)
    weights = (grid['hFacS'] * grid['DRF'][:, None, None] * grid['DXG'][None]).astype(np.float32)

    center_file = center_dir / f"centerV_{center_opt}_{nd}days_{nyt}yrs.nc"
    if not center_file.exists():
        raise FileNotFoundError(f"missing {center_file} (pass --center-dir)")
    centerV = comm.bcast(load_center(center_file, nx, ny, nz) if rank == 0 else None, root=0)

    # ---- POD lift of the zonally integrated transport, done once ------------
    Tr = np.load(D / "Tr.npy").astype(np.float32)
    nt_train = 360 * nyt // nd
    if Tr.shape[0] != nt_train * len(train_taus):
        raise ValueError(f"Tr has {Tr.shape[0]} rows, expected {nt_train * len(train_taus)}")
    t_lo, t_hi = slice_space(nt_train, rank, size)

    modes_local = np.zeros((nz * ny, r), dtype=np.float64)
    for s, t in enumerate(train_taus):
        z = zonal_transport(f"{prefix}{t}/states_20yrs.nc", t_lo, t_hi, nd, weights, centerV)
        blk = Tr[s * nt_train + t_lo:s * nt_train + t_hi, :]
        modes_local += (z.reshape(z.shape[0], nz * ny).T @ blk).astype(np.float64)
        del z
        gc.collect()
        if rank == 0:
            print(f"[moc] lifted training scenario tau{t}", flush=True)

    modes = np.zeros_like(modes_local)
    comm.Allreduce(modes_local, modes, op=MPI.SUM)
    del modes_local

    # W[c]: the lift above is anomaly-only; add the center's own transport back
    cz = np.einsum('zyx,zyx->zy', centerV, weights).reshape(nz * ny).astype(np.float64)

    depth = np.asarray(grid['RC'], dtype=float).ravel()
    lat = np.asarray(grid['YG'], dtype=float)[:, 0]

    # ---- per scenario -------------------------------------------------------
    for t in a.taus:
        kind = "train" if float(t) in [float(x) for x in train_taus] else "test"
        rom_file = D / f"Q_ROM_{kind}_{scen}{t}.npy"
        if not rom_file.exists():
            if rank == 0:
                print(f"[moc] SKIP tau{t}: no {rom_file.name}", flush=True)
            continue

        nt = comm.bcast(min(nt_default, np.load(rom_file, mmap_mode='r').shape[1])
                        if rank == 0 else None, root=0)
        f_lo, f_hi = slice_space(nt, rank, size)

        z_local = zonal_transport(f"{prefix}{t}/states_20yrs.nc", f_lo, f_hi, nd, weights, None)
        z_fom = gather_time(comm, z_local, nt, nz, ny)
        del z_local
        gc.collect()

        if rank == 0:
            # (nt, nz, ny) -> (nz, ny, nt), cumulative from the surface down
            psi_fom = (np.cumsum(z_fom.transpose(1, 2, 0), axis=0) / 1e6).astype(np.float32)
            del z_fom

            Q_ROM = np.load(rom_file)[:, :nt]
            rom = modes @ Q_ROM + cz[:, None]
            psi_rom = (np.cumsum(rom.reshape(nz, ny, nt), axis=0) / 1e6).astype(np.float32)
            del rom, Q_ROM

            ds = xr.Dataset(
                {
                    "psi_fom": (("depth", "lat", "time"), psi_fom,
                                {"units": "Sv", "long_name": "FOM overturning streamfunction"}),
                    "psi_rom": (("depth", "lat", "time"), psi_rom,
                                {"units": "Sv", "long_name": "ROM overturning streamfunction"}),
                },
                coords={
                    "depth": ("depth", depth, {"units": "m", "long_name": "cell-center depth"}),
                    "lat": ("lat", lat, {"units": "degrees_north", "long_name": "V-point latitude (YG)"}),
                    "time": ("time", np.arange(nt) * nd / 360.0, {"units": "years"}),
                },
                attrs={
                    "scenario": f"tau{t}",
                    "kind": kind,
                    "model": str(D) + "/",
                    "train_taus": str(train_taus),
                    "r": r,
                    "n_year_train": nyt,
                    "n_days": nd,
                    "center_opt": center_opt,
                    "scale": scale,
                    "convention": ('psi = cumsum(V*hFacS*DRF*DXG, over depth from surface)/1e6 [Sv]; '
                                   'AMOC strength at latitude L = psi.sel(lat=L,method="nearest").max("depth")'),
                    "description": ("Meridional overturning streamfunction, FOM vs ROM. Reconstructed "
                                    "from Q_ROM via training basis Tr.npy (zonal integral commutes with POD lift)."),
                },
            )
            path = out_dir / f"moc_tau{t}.nc"
            ds.to_netcdf(path)
            ds.close()
            del psi_fom, psi_rom, ds
            gc.collect()
            print(f"[moc] wrote {path} (nt={nt}, kind={kind})", flush=True)

        comm.Barrier()

    if rank == 0:
        print("[moc] done", flush=True)


if __name__ == "__main__":
    main()
