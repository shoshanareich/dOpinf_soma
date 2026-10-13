#!/usr/bin/env python3
"""MPI: MITgcm (FOM) volume salt and volume budgets per scenario -> salt_budget_fom_tau{T}.npz

Companion to fom_heat_budget.py. SOMA has no real freshwater flux (useRealFreshWaterFlux
defaults to .FALSE., no EmPmR file): the SSS restoring is a VIRTUAL salt flux, so

    salt:    dS_tot/dt = F_S(t),   S_tot = sum_{ijk} S * rA drF hFacC * rStarFacC
                                   F_S   = 1/tau_S * sum_{ij} rA drF_1 hFacC_1 (S* - S_1)
    volume:  dV/dt = 0,            V' = sum_{ij} rA eta   (exactConserv, no E-P-R)

S* (saltClimFile) is time-invariant (all 25920 records identical, and equal to the ROM's
analytic forcing.compute_salinity_forcing() * tau_S to 2e-6 psu), so F_S* is a constant.
No SRELAX diagnostic was written, so there is no monthly cross-check as for heat.

Per daily snapshot this saves
    sc_k      (nt, nz)  salt content per level, fixed thickness (no z* factor)  [psu m3]
    sc_zstar  (nt,)     total salt content with the z* thickness factor         [psu m3]
    FS_star   ()        +1/tau_S sum rA drF1 hFacC1 S*   (target part)          [psu m3/s]
    FS_S      (nt,)     +1/tau_S sum rA drF1 hFacC1 S_1  (damping part)         [psu m3/s]
                        -> restoring flux F_S = FS_star - FS_S
    eta_int   (nt,)     sum rA eta over wet columns (volume anomaly)            [m3]
    time, area, volume, drf, tau_s

Ranks are split over TIME, as in fom_heat_budget.py.

    mpiexec -np 28 python3 src/fom_salt_budget.py --taus 0.2 0.3 0.4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))  # core ROM modules
import paths  # noqa: E402
from fom_heat_budget import INPUT_BINARIES, read_grid  # noqa: E402

TAU_S = 15552000.0             # input/data: tauSaltClimRelax
SSS_FILE = 'SSS_relax_yrly_1200s_248.bin'


def budget_one(comm, run_dir, grid, out_file, chunk_t, max_t=None):
    rank, size = comm.Get_rank(), comm.Get_size()
    rac, drf, hfac, depth = grid['RAC'], grid['DRF'], grid['hFacC'], grid['Depth']
    nz, ny, nx = hfac.shape
    w3 = rac[None] * drf[:, None, None] * hfac            # cell volume, fixed thickness
    inv_depth = np.where(depth > 0, 1.0 / np.where(depth > 0, depth, 1.0), 0.0)
    w_surf = rac * drf[0] * hfac[0] / TAU_S
    wet = rac * (hfac[0] > 0)

    sss_file = run_dir + SSS_FILE
    if not Path(sss_file).exists():
        sss_file = INPUT_BINARIES + SSS_FILE
    sstar = np.array(np.memmap(sss_file, dtype='>f4', mode='r', shape=(1, ny, nx))[0],
                     dtype=np.float64)

    ds = xr.open_dataset(run_dir + 'states_20yrs.nc', engine='h5netcdf', decode_times=False)
    times = ds['time'].values.astype(np.float64)
    if max_t:
        times = times[:max_t]
    nt = len(times)
    bounds = np.linspace(0, nt, size + 1).astype(int)
    lo, hi = bounds[rank], bounds[rank + 1]

    sc_k = np.zeros((hi - lo, nz))
    sc_zs = np.zeros(hi - lo)
    fs_s = np.zeros(hi - lo)
    eta_int = np.zeros(hi - lo)
    for a in range(lo, hi, chunk_t):
        b = min(a + chunk_t, hi)
        S = ds['S'].isel(time=slice(a, b)).values.astype(np.float64)    # (n, nz, ny, nx)
        eta = ds['Eta'].isel(time=slice(a, b)).values.astype(np.float64)
        s = slice(a - lo, b - lo)
        sc_k[s] = np.einsum('tkji,kji->tk', S, w3)
        zfac = 1.0 + eta * inv_depth[None]
        sc_zs[s] = np.einsum('tkji,kji,tji->t', S, w3, zfac)
        fs_s[s] = np.einsum('tji,ji->t', S[:, 0], w_surf)
        eta_int[s] = np.einsum('tji,ji->t', eta, wet)
    ds.close()

    parts = [comm.gather(x, root=0) for x in (sc_k, sc_zs, fs_s, eta_int)]
    if rank != 0:
        return
    sc_k, sc_zs, fs_s, eta_int = (np.concatenate(p) for p in parts)
    np.savez(out_file, time=times, sc_k=sc_k, sc_zstar=sc_zs, FS_star=np.sum(w_surf * sstar),
             FS_S=fs_s, eta_int=eta_int, tau_s=TAU_S, area=np.sum(wet), volume=np.sum(w3),
             drf=drf, run_dir=np.array(run_dir))
    print(f'  wrote {out_file}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--taus', nargs='+', default=['0.2', '0.3', '0.4'])
    p.add_argument('--prefix', default=paths.FOM_PREFIX)
    p.add_argument('--grid-dir', default=paths.GRID_DIR)
    p.add_argument('--out-dir', default=paths.BUDGET_DIR)
    p.add_argument('--chunk-t', type=int, default=10)
    p.add_argument('--max-t', type=int, default=None, help='first N snapshots only (testing)')
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    grid = read_grid(comm, args.grid_dir)
    if comm.Get_rank() == 0:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    for tau in args.taus:
        if comm.Get_rank() == 0:
            print(f'tau{tau}', flush=True)
        budget_one(comm, f'{args.prefix}{tau}/', grid,
                   f'{args.out_dir}salt_budget_fom_tau{tau}.npz', args.chunk_t, args.max_t)
        comm.Barrier()


if __name__ == '__main__':
    main()
