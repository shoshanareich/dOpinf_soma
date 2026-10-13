#!/usr/bin/env python3
"""MPI: MITgcm (FOM) volume heat budget per scenario -> heat_budget_fom_tau{T}.npz

SOMA is a closed basin with no-flux walls/bottom, no shortwave penetration and no
geothermal flux, so every interior heat transport (advection, GM/Redi, KPP, diffusion)
is a flux divergence and integrates to zero. The only heat source is the SST restoring:

    dH/dt = F(t),   H = rho0 cp sum_{ijk} theta * rA drF hFacC * rStarFacC
                    F = -rho0 cp / tau_theta * sum_{ij} rA drF_1 hFacC_1 (theta_1 - theta*)

(MITgcm forcing_surf_relax.F; the z* factor rStarFacC = 1 + eta/Depth because
select_rStar=2, nonlinFreeSurf=4. MITgcm multiplies the surface forcing by
rStarExpC ~ 1, which the recomputed F below ignores.)

Per daily snapshot this saves
    hc_k      (nt, nz)  heat content per level, fixed thickness (no z* factor)   [J]
    hc_zstar  (nt,)     total heat content with the z* thickness factor          [J]
    F_star    (nt,)     +rho0 cp/tau sum rA drF1 hFacC1 theta*   (target part)    [W]
    F_theta   (nt,)     +rho0 cp/tau sum rA drF1 hFacC1 theta_1  (damping part)   [W]
                        -> recomputed restoring flux F = F_star - F_theta
    time      (nt,)     model time [s]
    area, volume, drf   wet surface area [m2], basin volume [m3], layer thicknesses [m]
    G_ctheta, G_btheta, G_sstar (nt,)  sum_ij theta_1*c_theta, theta_1*b_theta, S_1*s_star:
                        the surface functionals the imposed ROM B/C are projected from
                        (forcing.py patterns; used by rom_heat_budget.py)
and per 30-day month (from surfDiag monthly-mean diagnostics, the exact time integral
of what MITgcm applied every 1200 s step)
    trelax_int, surforct_int  (nmon,)  area-integrated TRELAX / surForcT x interval [J]
    diag_interval             (nmon, 2) averaging window [s]

theta* is rebuilt from the forcing file with MITgcm's GET_PERIODIC_INTERVAL
convention (record j, 0-based, centred at (j+1/2)*externForcingPeriod, linear interp).

Ranks are split over TIME (whole 3D snapshots read contiguously), like compute_moc.py.

    mpiexec -np 62 python3 src/fom_heat_budget.py --taus 0.1 0.2 0.3 0.4 0.5
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
from MITgcmutils import rdmds  # noqa: E402
from forcing import OceanForcing  # noqa: E402

# input/data: rhonil (rhoConst defaults to it), tauThetaClimRelax, forcing clock
RHO0 = 1025.022
CP = 3994.0                    # MITgcm default HeatCapacity_Cp
TAU_THETA = 2592000.0
FORCING_PERIOD = 1200.0
FORCING_CYCLE = 31104000.0
MONTH = 2592000.0
SST_FILE = 'SST_relax_change_max_1200s_248.bin'      # thetaClimFile in every run's data
INPUT_BINARIES = paths.INPUT_DIR


def read_grid(comm, grid_dir):
    rank = comm.Get_rank()
    out = {}
    for key in ('RAC', 'DRF', 'hFacC', 'Depth'):
        data = np.squeeze(rdmds(grid_dir + key)).astype(np.float64) if rank == 0 else None
        out[key] = comm.bcast(data, root=0)
    return out


def theta_star(sst_file, times, ny, nx):
    """theta*(t) at each model time, interpolated as MITgcm does."""
    nrec = int(round(FORCING_CYCLE / FORCING_PERIOD))
    mm = np.memmap(sst_file, dtype='>f4', mode='r', shape=(nrec, ny, nx))
    out = np.empty((len(times), ny, nx))
    for n, t in enumerate(times):
        x = t / FORCING_PERIOD - 0.5
        j0 = int(np.floor(x))
        w = x - j0
        out[n] = (1.0 - w) * mm[j0 % nrec] + w * mm[(j0 + 1) % nrec]
    return out


def monthly_diags(diag_dir, rac, nmon):
    """Area-integrated monthly TRELAX / surForcT times the averaging interval [J]."""
    trelax = np.full(nmon, np.nan)
    surf = np.full(nmon, np.nan)
    interval = np.full((nmon, 2), np.nan)
    for m in range(nmon):
        itr = 2160 * (m + 1)
        data, _, meta = rdmds(diag_dir + 'surfDiag', itrs=itr, returnmeta=True)
        names = [f.strip() for f in meta['fldlist']]
        t0, t1 = (float(v) for v in meta['timeinterval'][0])
        interval[m] = (t0, t1)
        trelax[m] = np.sum(rac * data[names.index('TRELAX')]) * (t1 - t0)
        surf[m] = np.sum(rac * data[names.index('surForcT')]) * (t1 - t0)
    return trelax, surf, interval


def find_diag_dir(run_dir):
    for d in (run_dir, run_dir.rstrip('/') + '_diags/'):
        if Path(d, 'surfDiag.0000002160.meta').exists():
            return d
    return None


def rom_forcing_fields(grid_dir, ny, nx):
    """The imposed ROM forcing patterns, built exactly as mixed_rom_forcings.py does."""
    F = OceanForcing(grid_dir=grid_dir, nx=nx, ny=ny, xo=0, yo=14, xc=31, yc=45,
                     dx=0.25, dy=0.25)
    s_star = F.compute_salinity_forcing()
    _, c_theta, b_theta = F.compute_temp_forcing()
    return (c_theta.reshape(ny, nx), b_theta.reshape(ny, nx), s_star.reshape(ny, nx))


def budget_one(comm, run_dir, grid, out_file, chunk_t, max_t=None, grid_dir=None):
    rank, size = comm.Get_rank(), comm.Get_size()
    rac, drf, hfac, depth = grid['RAC'], grid['DRF'], grid['hFacC'], grid['Depth']
    nz, ny, nx = hfac.shape
    w3 = rac[None] * drf[:, None, None] * hfac            # cell volume, fixed thickness
    inv_depth = np.where(depth > 0, 1.0 / np.where(depth > 0, depth, 1.0), 0.0)
    w_surf = rac * drf[0] * hfac[0] * RHO0 * CP / TAU_THETA

    ds = xr.open_dataset(run_dir + 'states_20yrs.nc', engine='h5netcdf', decode_times=False)
    times = ds['time'].values.astype(np.float64)
    if max_t:
        times = times[:max_t]
    nt = len(times)
    bounds = np.linspace(0, nt, size + 1).astype(int)
    lo, hi = bounds[rank], bounds[rank + 1]

    hc_k = np.zeros((hi - lo, nz))
    hc_zs = np.zeros(hi - lo)
    f_star = np.zeros(hi - lo)
    f_theta = np.zeros(hi - lo)
    # surface functionals the imposed ROM B/C are built from (validation of B, C)
    c_th, b_th, s_st = rom_forcing_fields(grid_dir, ny, nx)
    g_c = np.zeros(hi - lo)
    g_b = np.zeros(hi - lo)
    g_s = np.zeros(hi - lo)
    # the tau0.1 run dir lacks the symlink; fall back to the canonical copy
    sst_file = run_dir + SST_FILE
    if not Path(sst_file).exists():
        sst_file = INPUT_BINARIES + SST_FILE
    tstar = theta_star(sst_file, times[lo:hi], ny, nx)
    for a in range(lo, hi, chunk_t):
        b = min(a + chunk_t, hi)
        T = ds['T'].isel(time=slice(a, b)).values.astype(np.float64)    # (n, nz, ny, nx)
        eta = ds['Eta'].isel(time=slice(a, b)).values.astype(np.float64)
        s = slice(a - lo, b - lo)
        hc_k[s] = RHO0 * CP * np.einsum('tkji,kji->tk', T, w3)
        zfac = 1.0 + eta * inv_depth[None]
        hc_zs[s] = RHO0 * CP * np.einsum('tkji,kji,tji->t', T, w3, zfac)
        f_theta[s] = np.einsum('tji,ji->t', T[:, 0], w_surf)
        f_star[s] = np.einsum('tji,ji->t', tstar[s], w_surf)
        g_c[s] = np.einsum('tji,ji->t', T[:, 0], c_th)
        g_b[s] = np.einsum('tji,ji->t', T[:, 0], b_th)
        S1 = ds['S'].isel(time=slice(a, b), k=0).values.astype(np.float64)
        g_s[s] = np.einsum('tji,ji->t', S1, s_st)
    ds.close()

    parts = [comm.gather(x, root=0) for x in (hc_k, hc_zs, f_star, f_theta, g_c, g_b, g_s)]
    if rank != 0:
        return
    hc_k, hc_zs, f_star, f_theta, g_c, g_b, g_s = (np.concatenate(p) for p in parts)

    nmon = (nt - 1) // 30
    diag_dir = find_diag_dir(run_dir)
    extra = {}
    if diag_dir is not None:
        tr, sf, iv = monthly_diags(diag_dir, rac, nmon)
        extra = dict(trelax_int=tr, surforct_int=sf, diag_interval=iv,
                     diag_dir=np.array(diag_dir))
        if diag_dir != run_dir:
            print(f'  WARNING: monthly diagnostics taken from separate run {diag_dir}',
                  flush=True)
    else:
        print(f'  no surfDiag found for {run_dir}', flush=True)
    np.savez(out_file, time=times, hc_k=hc_k, hc_zstar=hc_zs, F_star=f_star,
             F_theta=f_theta, G_ctheta=g_c, G_btheta=g_b, G_sstar=g_s, rho0=RHO0, cp=CP, tau_theta=TAU_THETA,
             area=np.sum(rac * (hfac[0] > 0)), volume=np.sum(w3), drf=drf,
             run_dir=np.array(run_dir), **extra)
    print(f'  wrote {out_file}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--taus', nargs='+', default=['0.3'])
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
                   f'{args.out_dir}heat_budget_fom_tau{tau}.npz', args.chunk_t, args.max_t,
                   grid_dir=args.grid_dir)
        comm.Barrier()


if __name__ == '__main__':
    main()
