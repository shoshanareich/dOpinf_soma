#!/usr/bin/env python3
"""Refit the canonical ROM with corrected imposed B/C (conservation or FOM theta* pattern).

Reuses the canonical run's POD (Qhat, Tr) and imposed operators (B, C from model.h5); no
snapshot reading and no SVD. Only the learning step is redone, with the same Tikhonov grid
and scoring as mixed_rom_forcings.py (TikhonovSweep.run_mpi), so the selection is directly
comparable. Writes to a NEW sibling directory of the canonical run and never touches it.

--fix bc: change only the heat and salt components of each imposed column. With
W = [h s] (reduced heat / salt content functionals, rom_heat_budget.py) and per-step
physical targets T,

    X <- X + D (W^T D)^{-1} (T - W^T X)

with correction directions D = [f g] (SST / SSS functionals; default --direction surface)
or D = W (minimal-norm change; --direction minnorm). Components of X that W does not see
are unchanged in the minnorm case.

    column          heat target                       salt target
    B[:,0] (Tmax)   Dt rho0 cp sum a1 b*               0
    B[:,1] (wind)   0                                  0
    C               Dt rho0 cp sum a1 c*               Dt sum a1 S*/tau_S

a1 = rA DRF_1 hFacC_1. c*, b* are the restoring-target patterns of the forcing file MITgcm
actually ran with (model_dev4/gendata_seasonal_variations.ipynb: slope 1/((ny-2) dy), no
hFacC factor), theta*(y,t) = tau_theta (c* + b* Tmax(t)); forcing.py uses 1/((ny-8) dy)
and scales by hFacC, about 0.4 degC warmer on the basin mean.
--fix theta: no conservation change. Rebuild only the thermal parts of the imposed B[:,0]
and C with the paper run's own recipe (project_surface_forcings_multiple: the saved
centered surface SST times Tr, same scaling), swapping forcing.py's theta* pattern for the
FOM file's pattern c*, b* above (masked to wet cells). Wind and salt parts are unchanged.
--fix physical: no conservation change. Rebuild ALL imposed columns as physical per-step
increments in the ROM's scaled coordinates. A tendency g added for one step changes the scaled
state x_s = (x - xbar)/alpha by Dt g/alpha, so its reduced column is

    Dt V^T (g/alpha) = Dt Tr^T Qc^T (g / alpha^2)      (alpha per point; Qc = centered surface)

where the paper recipe has Tr^T Qc^T g. Tendencies: thermal c*, b* (FOM file pattern, as
--fix theta), salt S*/tau_S (FOM file, wet cells, no hFacC factor), wind shape/(rho0 DRF_1
hFacW) per unit tau4 (MITgcm's stress -> momentum tendency). The restoring damping stays
learned, as in the paper formulation.
--fix physical_windneutral: --fix physical, then the wind column's heat and salt removed with
the minimal-norm correction (--fix bc's formula with D = W and zero targets, applied to
B[:,1] only). Thermal columns and C stay physically scaled.
--fix constrained: exact heat and salt conservation. Imposed columns as
physical_windneutral, plus the Tmax column's heat set to the physical value (minimal-norm,
salt 0; the physically scaled column only delivers what the basis captures). The learned
operators are constrained, for w in {h, s} with surface functional phi in {f, g}:

    w^T c = Dt (F*_const - F_1(center)) - w^T C      (heat: P_c; salt: FS*)
    w^T A = w^T - Dt phi^T                            (the restoring damping)
    w^T H = 0                                         (advection moves heat/salt, never makes it)

so w^T (q_{k+1} - q_k) = Dt F(q_k, u_k) exactly (explicit in the flux). Solved by rotating
the reduced coordinates so span{h, s} is two of them: the constraints fix those two rows of
the operator matrix, the other r-2 rows are the usual Tikhonov fit (the rotation is
orthogonal, so the regularization and the sweep are unchanged). The constrained solver is
swapped into learn_rom for this process only; learn_rom.py itself is not modified.
--fix constrained_paperbc: --fix constrained without the physical rescaling. The imposed B/C
keep the paper recipe (Tr^T Qc^T g, forcing.py patterns); only the heat and salt components of
B change, as in constrained: wind column heat/salt removed and Tmax column heat set to physical
(minimal-norm, so everything outside span{h, s} is the paper's column). C is untouched (the
constraint on w^T c absorbs its heat/salt). Same hard constraints on c, A, H.
--fix none: original B/C (a control through the same code path).

Before the sweep, rank 0 refits once at the canonical regs with the ORIGINAL B/C and checks
that it reproduces the canonical model.h5 (aborts otherwise).

    mpiexec -np 62 python3 analysis/refit_conservation.py --fix bc --suffix conserve_bcfix
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))  # core ROM modules
import paths  # noqa: E402
from MITgcmutils import rdmds  # noqa: E402
from forcing import OceanForcing, build_rom_inputs  # noqa: E402
import learn_rom  # noqa: E402
import opinf  # noqa: E402
import scipy.linalg as sla  # noqa: E402
from learn_rom import TikhonovSweep, _make_model, _make_solver  # noqa: E402
from rom_heat_budget import RUN_DEFAULT, DAY, ckron, load_operators  # noqa: E402

TMIN = -1.8          # SST restoring target at the northern edge (gendata notebook)
CANON_REG = (8.91250938133746e-06, 20691.3808111479)   # canonical selection (taus24_30665_3.out)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def functionals(run, bd, grid_dir, cfg):
    """Reduced heat (h) and salt (s) content functionals + grid weights (rank 0)."""
    ny = nx = 248
    pre = str(Path(run).parent) + '/'
    nt_tr = 360 * cfg['n_year_train'] // cfg['n_days']
    rac = np.squeeze(rdmds(grid_dir + 'RAC')).astype(float)
    drf = np.squeeze(rdmds(grid_dir + 'DRF')).astype(float)
    hfac = np.squeeze(rdmds(grid_dir + 'hFacC')).astype(float)
    nz = hfac.shape[0]
    w3 = rac[None] * drf[:, None, None] * hfac
    tag = f"{cfg['center_opt']}_{cfg['n_days']}days_{cfg['n_year_train']}yrs"
    cen = {}
    for v in ('T', 'S'):
        with xr.open_dataset(pre + f'center{v}_{tag}.nc', engine='h5netcdf') as ds:
            cen[v] = np.squeeze(ds['center'].values).reshape(nz, ny, nx)
    ht = [np.load(f'{bd}heat_budget_fom_tau{t}.npz') for t in cfg['train_taus']]
    st = [np.load(f'{bd}salt_budget_fom_tau{t}.npz') for t in cfg['train_taus']]
    rho_cp = float(ht[0]['rho0'] * ht[0]['cp'])
    Tr = np.load(run + 'Tr.npy')
    cat = lambda ds, k: np.concatenate([d[k][:nt_tr] for d in ds])  # noqa: E731
    h = Tr.T @ (cat(ht, 'hc_k').sum(1) - rho_cp * np.sum(w3 * cen['T']))
    s = Tr.T @ (cat(st, 'sc_k').sum(1) - np.sum(w3 * cen['S']))
    a1 = rac * drf[0] * hfac[0]
    # surface-layer functionals: the reduced directions that change SST / SSS (restoring
    # damping parts of the heat / salt flux, as in rom_heat_budget.py)
    f = Tr.T @ (cat(ht, 'F_theta') - rho_cp / float(ht[0]['tau_theta']) * np.sum(a1 * cen['T'][0]))
    g = Tr.T @ (cat(st, 'FS_S') - np.sum(a1 * cen['S'][0]) / float(st[0]['tau_s']))
    return dict(h=h, s=s, f=f, g=g, a1=a1, rho_cp=rho_cp,
                Fth_c=rho_cp / float(ht[0]['tau_theta']) * np.sum(a1 * cen['T'][0]),
                FSs_c=np.sum(a1 * cen['S'][0]) / float(st[0]['tau_s']),
                tau_th=float(ht[0]['tau_theta']), area=float(ht[0]['area']),
                vol=float(ht[0]['volume']), FS_star=float(st[0]['FS_star']),
                F_star=ht[0]['F_star'])


def correct_bc(B, C, fn, grid_dir, dt, Tmax, direction='surface'):
    """Set the heat and salt components of the imposed B, C columns to the physical targets."""
    ny = nx = 248
    F = OceanForcing(grid_dir=grid_dir, nx=nx, ny=ny, xo=0, yo=14, xc=31, yc=45,
                     dx=0.25, dy=0.25)
    den = (ny - 2) * F.dy
    c_star = (-TMIN * (F.ynorth - F.Y) / den + TMIN) / fn['tau_th']
    b_star = ((F.ynorth - F.Y) / den) / fn['tau_th']
    P_c = fn['rho_cp'] * np.sum(fn['a1'] * c_star)          # W
    P_b = fn['rho_cp'] * np.sum(fn['a1'] * b_star)          # W per degC of Tmax
    # check: the file-consistent target reproduces MITgcm's recomputed F* month by month
    Fs = fn['F_star'][:360].reshape(12, 30).mean(1)
    rms = np.sqrt(np.mean((P_c + P_b * Tmax - Fs) ** 2)) / fn['area']
    log(f'theta* target vs MITgcm F*: RMS diff {rms:.3e} W/m^2 (F* seasonal amp '
        f'{0.5 * np.ptp(Fs) / fn["area"]:.2f})')

    W = np.column_stack([fn['h'], fn['s']])
    # correction directions: 'minnorm' -> along W itself (smallest change in q, but the heat
    # goes where the basis stores the most heat per unit q, i.e. deep T); 'surface' -> along
    # the SST / SSS functionals f, g, so the imposed forcing acts through the surface layer
    D = W if direction == 'minnorm' else np.column_stack([fn['f'], fn['g']])
    G = D @ np.linalg.inv(W.T @ D)                           # X += G (T - W^T X)
    targets = {'B0': np.array([dt * P_b, 0.0]), 'B1': np.zeros(2),
               'C': np.array([dt * P_c, dt * fn['FS_star']])}
    cols = {'B0': B[:, 0].copy(), 'B1': B[:, 1].copy(), 'C': C.copy()}
    heat_u = lambda x: x / (fn['area'] * dt)                           # noqa: E731  J/step -> W/m^2
    salt_u = lambda x: x / fn['vol'] * 1e3 / dt * 360 * DAY            # noqa: E731  -> 1e-3 psu/yr
    new = {}
    for k, x in cols.items():
        new[k] = x + G @ (targets[k] - W.T @ x)
        per = ' per degC Tmax' if k == 'B0' else (' per N/m^2' if k == 'B1' else '')
        log(f'  {k}: heat {heat_u(fn["h"] @ x):+.4g} -> {heat_u(fn["h"] @ new[k]):+.4g} W/m^2{per};'
            f'  salt {salt_u(fn["s"] @ x):+.4g} -> {salt_u(fn["s"] @ new[k]):+.4g} 1e-3 psu/yr{per};'
            f'  |change|/|col| {np.linalg.norm(new[k] - x) / np.linalg.norm(x):.3e}')
    return np.column_stack([new['B0'], new['B1']]), new['C']


def fom_theta_patterns(grid_dir):
    """c*, b* of the forcing file MITgcm ran with, as flat (ny*nx,) tendencies [degC/s]."""
    F = OceanForcing(grid_dir=grid_dir, nx=248, ny=248, xo=0, yo=14, xc=31, yc=45,
                     dx=0.25, dy=0.25)
    _, c_theta, b_theta = F.compute_temp_forcing()
    tau_th = 2592000.0                                   # tauThetaClimRelax (input/data)
    wet = (np.asarray(F.hfacc) > 0).astype(float)
    den = (F.ny - 2) * F.dy
    c_star = wet * (-TMIN * (F.ynorth - F.Y) / den + TMIN) / tau_th
    b_star = wet * ((F.ynorth - F.Y) / den) / tau_th
    return c_theta, b_theta, c_star.ravel(), b_star.ravel()


def theta_pattern_bc(B, C, src, cfg, grid_dir):
    """Paper-recipe thermal columns with the FOM theta* pattern; wind and salt untouched."""
    pre = str(Path(src).parent) + '/'
    sst = np.load(pre + f"SST_{cfg['center_opt']}_{cfg['scale']}_{cfg['n_year_train']}yrs"
                  '_transformed_surface.npy')
    V_sst = sst.astype(float) @ np.load(src + 'Tr.npy')          # as project_surface_forcings_multiple
    del sst
    c_theta, b_theta, c_star, b_star = fom_theta_patterns(grid_dir)
    B0_old, Cth_old = V_sst.T @ b_theta, V_sst.T @ c_theta
    err = np.linalg.norm(B0_old - B[:, 0]) / np.linalg.norm(B[:, 0])
    log(f'check: paper recipe rebuilds B[:,0] from model.h5 to rel err {err:.2e}')
    if err > 1e-6:
        raise RuntimeError('cannot reproduce the paper run\'s B[:,0] from the saved surfaces')
    B_new = B.copy()
    B_new[:, 0] = V_sst.T @ b_star
    C_new = C - Cth_old + V_sst.T @ c_star
    log(f'  B[:,0] change {np.linalg.norm(B_new[:, 0] - B[:, 0]) / np.linalg.norm(B[:, 0]):.3e},'
        f'  C change {np.linalg.norm(C_new - C) / np.linalg.norm(C):.3e} (relative)')
    return B_new, C_new


def _surface_dot(arr, w, chunk=8192):
    """arr^T w for a (npts, nt) float32 array, chunked over points, in float64."""
    out = np.zeros(arr.shape[1])
    for i in range(0, arr.shape[0], chunk):
        out += np.asarray(arr[i:i + chunk], dtype=float).T @ w[i:i + chunk]
    return out


def physical_bc(B, C, src, cfg, grid_dir):
    """All imposed columns as physical per-step increments (see --fix physical)."""
    ny = nx = 248
    npt = ny * nx
    pre = str(Path(src).parent) + '/'
    tag = f"{cfg['center_opt']}_{cfg['scale']}_{cfg['n_year_train']}yrs"
    dt = cfg['n_days'] * DAY
    Tr = np.load(src + 'Tr.npy')
    surf = {v: np.load(pre + f'{v}_{tag}_transformed_surface.npy', mmap_mode='r')
            for v in ('SST', 'SSS', 'SSU')}
    alpha = np.load(pre + f'alpha_{tag}.npy', allow_pickle=True).item()
    a_surf = {'SST': np.ravel(alpha['T'])[:npt], 'SSS': np.ravel(alpha['S'])[:npt],
              'SSU': np.ravel(alpha['U'])[:npt]}       # alpha stored (nz, ny, nx): level 0 first

    F = OceanForcing(grid_dir=grid_dir, nx=nx, ny=ny, xo=0, yo=14, xc=31, yc=45,
                     dx=0.25, dy=0.25)
    s_star_py = F.compute_salinity_forcing().ravel()
    g_py = F.compute_wind_forcing().ravel()
    c_theta, b_theta, c_star, b_star = fom_theta_patterns(grid_dir)

    # check 1: the paper recipe (no alpha, no Dt) rebuilds every column of model.h5
    proj = lambda v, w: Tr.T @ _surface_dot(surf[v], w)  # noqa: E731
    chk = {'B[:,0]': (proj('SST', b_theta), B[:, 0]), 'B[:,1]': (proj('SSU', g_py), B[:, 1]),
           'C': (proj('SST', c_theta) + proj('SSS', s_star_py), C)}
    for k, (a, b) in chk.items():
        e = np.linalg.norm(a - b) / np.linalg.norm(b)
        log(f'check: paper recipe rebuilds {k} to rel err {e:.2e}')
        if e > 1e-6:
            raise RuntimeError(f'cannot reproduce the paper run\'s {k} from the saved surfaces')

    # check 2: alpha rows line up with the surface rows. Qc/alpha is the scaled data, whose
    # per-point std is 1/alpha2 (the per-variable maxabs factor) at every varying point
    wet = np.asarray(F.hfacc).ravel() > 0
    for v, a in a_surf.items():
        sd = np.concatenate([np.asarray(surf[v][i:i + 8192], dtype=float).std(1)
                             for i in range(0, npt, 8192)])
        ok = wet & (sd > 0)
        ratio = sd[ok] / a[ok]
        log(f'check: {v} std(Qc)/alpha on wet points: {ratio.min():.4e} .. {ratio.max():.4e}'
            ' (should be one constant, 1/alpha2)')
        if ratio.max() / ratio.min() > 1.01:
            raise RuntimeError(f'alpha rows do not line up with the {v} surface rows')

    # FOM tendencies: salt from the MITgcm file itself; wind stress -> momentum tendency
    rho0, tau_s = 1025.022, 15552000.0                   # rhonil, tauSaltClimRelax (input/data)
    sss = np.fromfile(paths.INPUT_DIR + 'SSS_relax_yrly_1200s_248.bin',
                      dtype='>f4', count=npt).astype(float)
    s_star = np.where(wet, sss / tau_s, 0.0)
    hfw = np.squeeze(rdmds(grid_dir + 'hFacW'))[0].ravel()
    drf = np.squeeze(rdmds(grid_dir + 'DRF')).astype(float)
    shape = np.where(np.asarray(F.hfacc).ravel() > 0, g_py / np.where(wet, np.asarray(F.hfacc).ravel(), 1), 0)
    wind_file = np.fromfile(paths.INPUT_DIR +
                            'wind_sv_0.1_0.4_cos_1200s_248.bin', dtype='>f4', count=npt).astype(float)
    m = (hfw > 0) & (np.abs(shape) > 1e-3 * np.abs(shape).max())
    r_w = wind_file[m] / shape[m]
    log(f'check: MITgcm wind file / forcing.py shape on wet-W points: {r_w.min():.6f} .. '
        f'{r_w.max():.6f} (one constant = tau4 of record 0)')
    wind_tend = np.where(hfw > 0, shape / (rho0 * drf[0] * np.where(hfw > 0, hfw, 1)), 0.0)

    phys = lambda v, g: dt * proj(v, g / a_surf[v] ** 2)  # noqa: E731
    B_new = np.column_stack([phys('SST', b_star), phys('SSU', wind_tend)])
    C_new = phys('SST', c_star) + phys('SSS', s_star)
    for k, (a, b) in {'B[:,0]': (B_new[:, 0], B[:, 0]), 'B[:,1]': (B_new[:, 1], B[:, 1]),
                      'C': (C_new, C)}.items():
        log(f'  {k}: |new|/|paper| = {np.linalg.norm(a) / np.linalg.norm(b):.3e}, '
            f'cos(new, paper) = {a @ b / (np.linalg.norm(a) * np.linalg.norm(b)):+.3f}')
    return B_new, C_new


def wind_neutral(B, fn, dt):
    """Remove the heat and salt components of the wind column (minimal-norm change)."""
    W = np.column_stack([fn['h'], fn['s']])
    b1 = B[:, 1]
    b1_new = b1 - W @ np.linalg.solve(W.T @ W, W.T @ b1)
    heat_u = lambda x: x / (fn['area'] * dt)                           # noqa: E731
    salt_u = lambda x: x / fn['vol'] * 1e3 / dt * 360 * DAY            # noqa: E731
    log(f'  wind column: heat {heat_u(fn["h"] @ b1):+.4g} -> {heat_u(fn["h"] @ b1_new):+.2e} '
        f'W/m^2 per N/m^2;  salt {salt_u(fn["s"] @ b1):+.4g} -> {salt_u(fn["s"] @ b1_new):+.2e} '
        f'1e-3 psu/yr per N/m^2;  |change|/|col| {np.linalg.norm(b1_new - b1) / np.linalg.norm(b1):.3e}')
    B = B.copy()
    B[:, 1] = b1_new
    return B


def set_column_budget(col, W, targets):
    """Minimal-norm change of one column so that W^T col = targets."""
    return col + W @ np.linalg.solve(W.T @ W, targets - W.T @ col)


def conservation_constraints(B, C, fn, grid_dir, dt, r):
    """Tmax column heat -> physical; return (W, T) with W^T [c A H] = T for the fit."""
    _, _, c_star, b_star = fom_theta_patterns(grid_dir)
    a1 = fn['a1'].ravel()
    P_c = fn['rho_cp'] * np.sum(a1 * c_star)            # W, constant part of F*
    P_b = fn['rho_cp'] * np.sum(a1 * b_star)            # W per degC of Tmax
    W = np.column_stack([fn['h'], fn['s']])
    heat_u = lambda x: x / (fn['area'] * dt)            # noqa: E731
    b0 = set_column_budget(B[:, 0], W, np.array([dt * P_b, 0.0]))
    log(f'  Tmax column heat: {heat_u(fn["h"] @ B[:, 0]):.4g} -> {heat_u(fn["h"] @ b0):.4g} W/m^2 '
        f'per degC (physical {heat_u(dt * P_b):.4g}); |change|/|col| '
        f'{np.linalg.norm(b0 - B[:, 0]) / np.linalg.norm(B[:, 0]):.3e}')
    B = B.copy()
    B[:, 0] = b0
    nq = r * (r + 1) // 2
    T = np.zeros((2, 1 + r + nq))                       # opinf column order: c | A | H
    for i, (w, phi, const) in enumerate(((fn['h'], fn['f'], P_c - fn['Fth_c']),
                                         (fn['s'], fn['g'], fn['FS_star'] - fn['FSs_c']))):
        T[i, 0] = dt * const - w @ C
        T[i, 1:1 + r] = w - dt * phi
    salt_u = lambda x: x / fn['vol'] * 1e3 / dt * 360 * DAY   # noqa: E731  -> 1e-3 psu/yr
    log(f'  constraints: w^T c = [{heat_u(T[0, 0]):.4g} W/m^2, {salt_u(T[1, 0]):.4g} 1e-3 psu/yr], '
        f'{2 * (1 + r + nq)} equations on {r * (1 + r + nq)} unknowns')
    phys = dict(dt=dt, P_b=P_b, const=np.array([P_c - fn['Fth_c'], fn['FS_star'] - fn['FSs_c']]),
                Phi=np.column_stack([fn['f'], fn['g']]))
    return B, (W, T, phys)


class ConstrainedTikhonovSolver(opinf.lstsq.TikhonovSolver):
    """Tikhonov OpInf with W^T Ohat = T imposed exactly (rotated-row solve)."""

    def __init__(self, regularizer, W, T):
        super().__init__(regularizer, method='lstsq')
        Qf, Rf = np.linalg.qr(W, mode='complete')
        k = W.shape[1]
        self._Q1, self._Q2 = Qf[:, :k], Qf[:, k:]
        self._Y1 = np.linalg.solve(Rf[:k].T, T)         # Q1^T Ohat, fixed by the constraints

    def solve(self):
        DPad = np.vstack((self.data_matrix, self.regularizer))
        ZPad = np.vstack((self.lhs_matrix.T @ self._Q2, np.zeros((self.d, self._Q2.shape[1]))))
        Y2 = sla.lstsq(DPad, ZPad, **self.options)[0].T  # Q2^T Ohat, ordinary Tikhonov fit
        return self._Q1 @ self._Y1 + self._Q2 @ Y2


def use_constrained_solver(W, T):
    """Swap the constrained solver into learn_rom (this process only)."""
    def make_solver(reg_A, reg_H, r, regularization_target=None, *, learn_N=False, reg_N=None,
                    m=None):
        if regularization_target is not None or learn_N:
            raise NotImplementedError('constrained fit supports plain Tikhonov only')
        return ConstrainedTikhonovSolver(learn_rom._make_regularizer(reg_A, reg_H, r), W, T)
    learn_rom._make_solver = make_solver


def check_conservation(model, constraint, states, inputs):
    """Fitted operators vs the constraints, and the per-step heat/salt change along the
    training data vs the physical budget Dt F(q, u) computed independently of the model."""
    W, T, phys = constraint
    ops = model.model.operators
    O = np.hstack([ops[2].entries[:, None], ops[3].entries, ops[4].entries])
    log(f'check: |W^T O - T| / |T| = {np.linalg.norm(W.T @ O - T) / np.linalg.norm(T):.2e}')
    for q, u in zip(states, inputs):
        q0 = q[:, :-1]
        step = (ops[0].entries @ u[:, :-1] + (ops[1].entries + ops[2].entries)[:, None]
                + ops[3].entries @ q0 + ops[4].entries @ ckron(q0))
        rom = W.T @ (step - q0)
        forcing = np.vstack([phys['P_b'] * u[0, :-1], np.zeros(q0.shape[1])])
        budget = phys['dt'] * (phys['const'][:, None] + forcing - phys['Phi'].T @ q0)
        log('  per-step change vs physical budget, heat / salt: max |diff| / max |change| = '
            + ', '.join(f'{np.abs(rom[i] - budget[i]).max() / np.abs(budget[i]).max():.1e}'
                        for i in range(2)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--src-run', default=RUN_DEFAULT)
    p.add_argument('--fix', choices=['bc', 'theta', 'physical', 'physical_windneutral',
                                     'constrained', 'constrained_paperbc', 'none'], default='bc')
    p.add_argument('--direction', choices=['surface', 'minnorm'], default='surface',
                   help='--fix bc: correct along the SST/SSS functionals (surface) or W (minnorm)')
    p.add_argument('--suffix', required=True,
                   help='output dir = <src-run>_<suffix>/ (sibling of the canonical run)')
    p.add_argument('--budget-dir', default=paths.BUDGET_DIR)
    p.add_argument('--grid-dir', default=paths.GRID_DIR)
    p.add_argument('--stability-taus', nargs='+', type=float, default=None,
                   help='full-horizon stability gate (as enforce_full_horizon_stability in '
                        'mixed_rom_forcings.py): reject candidates whose rollout at any of these '
                        'taus goes non-finite or exceeds --stability-factor * max|Qhat|. Off if unset')
    p.add_argument('--stability-factor', type=float, default=10.0,
                   help='stability bound as a multiple of max|Qhat| (driver default 10)')
    p.add_argument('--stability-norm-factor', type=float, default=None,
                   help='also reject a gate rollout whose yearly-mean ||q|| exceeds this multiple '
                        'of the training level mean_k ||Qhat[:, k]|| in any year. Off if unset')
    p.add_argument('--stability-rho-max', type=float, default=None,
                   help='reject candidates whose A has spectral radius above this. Off if unset')
    p.add_argument('--stability-years', type=int, default=None,
                   help='gate horizon in years (default n_year_train + n_year_predict)')
    p.add_argument('--grid-A', nargs=3, type=float, default=[-10, 1, 21], metavar=('LO', 'HI', 'N'),
                   help='reg_A grid = logspace(LO, HI, N) (default: the driver grid -10 1 21)')
    p.add_argument('--grid-H', nargs=3, type=float, default=[-2, 6, 20], metavar=('LO', 'HI', 'N'),
                   help='reg_H grid = logspace(LO, HI, N) (default: the driver grid -2 6 20)')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--out-dir', default=None, help='override the output directory (testing)')
    p.add_argument('--test-grid', action='store_true',
                   help='1-candidate grid at the canonical regs (tests the pipeline end to end)')
    p.add_argument('--dry-run', action='store_true',
                   help='run the check and the B/C correction, then stop (no sweep, no output)')
    p.add_argument('--skip-check', action='store_true',
                   help='skip the refit-reproduces-canonical check')
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    src = args.src_run.rstrip('/') + '/'
    out = (args.out_dir.rstrip('/') + '/') if args.out_dir else src.rstrip('/') + f'_{args.suffix}/'
    cfg = json.load(open(src + 'config.json'))
    n_days, n_year = cfg['n_days'], cfg['n_year_train']
    nt = 360 * n_year // n_days
    dt = n_days * DAY
    t = np.linspace(0, n_year * 360 * 24 * 3600, nt)          # as in mixed_rom_forcings.py

    payload, abort = None, None
    if rank == 0:
        if Path(out).resolve() == Path(src).resolve():
            abort = 'output directory equals the source run'
        elif Path(out, 'model.h5').exists() and not args.overwrite:
            abort = f'{out}model.h5 exists (pass --overwrite to replace it)'
        if abort is None:
            log(f'source {src}\noutput {out}\nfix {args.fix}, {size} ranks')
            ops = load_operators(src)
            B, C = ops['B'], ops['C']
            Qhat = np.load(src + 'Qhat.npy')
            r = Qhat.shape[0]
            states = [Qhat[:, i * nt:(i + 1) * nt] for i in range(Qhat.shape[1] // nt)]
            Fo = OceanForcing(grid_dir=args.grid_dir, nx=248, ny=248, xo=0, yo=14, xc=31,
                              yc=45, dx=0.25, dy=0.25)
            Tmax, _, _ = Fo.compute_temp_forcing()
            tau4 = lambda tau: Fo.compute_wind_seasonal_amplitude(max_val=float(tau))  # noqa: E731
            inputs = [build_rom_inputs(Tmax, tau4(tau), n_year, n_days) for tau in cfg['train_taus']]

            if not args.skip_check:
                log('check: refit at canonical regs with original B/C')
                rom = _make_model(B, C, solver=_make_solver(*CANON_REG, r))
                rom.fit(states=states, inputs=inputs)
                errs = [np.linalg.norm(o.entries - ops[k]) / np.linalg.norm(ops[k])
                        for o, k in zip(rom.model.operators[2:], ('c', 'A', 'H'))]
                log(f'  learned c, A, H vs canonical model.h5: rel err '
                    + ', '.join(f'{e:.2e}' for e in errs))
                if max(errs) > 1e-6:
                    abort = 'refit does not reproduce the canonical model'
            if abort is None and args.fix in ('physical', 'physical_windneutral', 'constrained'):
                B, C = physical_bc(B, C, src, cfg, args.grid_dir)
            if abort is None and args.fix in ('physical_windneutral', 'constrained',
                                              'constrained_paperbc'):
                fn = functionals(src, args.budget_dir, args.grid_dir, cfg)
                B = wind_neutral(B, fn, dt)
            if abort is None and args.fix in ('constrained', 'constrained_paperbc'):
                B, constraint = conservation_constraints(B, C, fn, args.grid_dir, dt, r)
            if abort is None and args.fix == 'theta':
                B, C = theta_pattern_bc(B, C, src, cfg, args.grid_dir)
            if abort is None and args.fix == 'bc':
                fn = functionals(src, args.budget_dir, args.grid_dir, cfg)
                B, C = correct_bc(B, C, fn, args.grid_dir, dt, Tmax, args.direction)
            payload = dict(B=B, C=C, states=states, inputs=inputs, r=r, Tmax=Tmax,
                           constraint=(constraint if args.fix in ('constrained', 'constrained_paperbc')
                                       else None))
    if args.dry_run and abort is None:
        abort = 'dry run: stopping before the sweep'
    abort = comm.bcast(abort, root=0)
    if abort:
        if rank == 0:
            print(f'ERROR: {abort}', flush=True)
        sys.exit(1)
    payload = comm.bcast(payload, root=0)
    if payload['constraint'] is not None:
        use_constrained_solver(*payload['constraint'][:2])

    # optional full-horizon stability gate, built as mixed_rom_forcings.py builds it (rollouts
    # from the first training IC = Qhat[:, 0], the IC of every test rollout)
    gate = {}
    if args.stability_taus:
        n_years = args.stability_years or (n_year + cfg['n_year_predict'])
        Fo = OceanForcing(grid_dir=args.grid_dir, nx=248, ny=248, xo=0, yo=14, xc=31, yc=45,
                          dx=0.25, dy=0.25)
        gate = dict(
            stability_inputs=[build_rom_inputs(payload['Tmax'],
                                               Fo.compute_wind_seasonal_amplitude(max_val=tau),
                                               n_years, n_days) for tau in args.stability_taus],
            stability_niters=int(n_years * 360 / n_days),
            stability_bound=args.stability_factor * float(np.abs(np.hstack(payload['states'])).max()))
        norm_level = float(np.linalg.norm(np.hstack(payload['states']), axis=0).mean())
        if args.stability_norm_factor:
            gate.update(stability_norm_bound=args.stability_norm_factor * norm_level,
                        stability_steps_per_year=360 // n_days)
        if args.stability_rho_max:
            gate['stability_rho_max'] = args.stability_rho_max
        if rank == 0:
            log(f'stability gate: taus {args.stability_taus}, {gate["stability_niters"]} steps, '
                f'bound {gate["stability_bound"]:.4g} ({args.stability_factor:g} x max|Qhat|)'
                + (f', yearly-mean ||q|| <= {gate["stability_norm_bound"]:.4g} '
                   f'({args.stability_norm_factor:g} x training level {norm_level:.4g})'
                   if 'stability_norm_bound' in gate else '')
                + (f', rho(A) <= {args.stability_rho_max:g}' if 'stability_rho_max' in gate else ''))

    # same grid and scoring as mixed_rom_forcings.py
    sweep = TikhonovSweep(Qs_=payload['states'], inputs=payload['inputs'], B=payload['B'],
                          C=payload['C'], r=payload['r'], nt=nt, t=t, **gate,
                          weights_A=([CANON_REG[0]] if args.test_grid else
                                     np.logspace(args.grid_A[0], args.grid_A[1], int(args.grid_A[2]))),
                          weights_H=([CANON_REG[1]] if args.test_grid else
                                     np.logspace(args.grid_H[0], args.grid_H[1], int(args.grid_H[2]))))
    t0 = time.perf_counter()
    reg_A, reg_H, best_err = sweep.run_mpi(comm)
    if reg_A is None:
        if rank == 0:
            print('ERROR: no regularization candidate produced a finite model'
                  + (' that passes the stability gate' if gate else ''), flush=True)
        sys.exit(1)
    if rank != 0:
        return
    log(f'sweep done in {(time.perf_counter() - t0) / 60:.1f} min: best err {best_err:.4g}, '
        f'reg_A {reg_A:.4e}, reg_H {reg_H:.4e}  (canonical {CANON_REG[0]:.4e}, {CANON_REG[1]:.4e})')
    log(f'  {int(sweep.converged.sum())} of {sweep.converged.size} candidates passed'
        + (' the stability gate' if gate else ''))
    ok = np.isfinite(sweep.errors)
    if ok.any():
        for i, j in zip(*np.nonzero(ok)):
            if sweep.errors[i, j] < best_err * 1.10:
                log(f'    near-best: reg_A {sweep.weights_A[j]:.3e}, reg_H {sweep.weights_H[i]:.3e}, '
                    f'score {sweep.errors[i, j]:.4g}')

    os.makedirs(out, exist_ok=True)
    model = sweep.fit_best_model()
    model.model.save(out + 'model.h5', overwrite=True)
    A = next(op.entries for op in model.model.operators
             if isinstance(op, opinf.operators.LinearOperator))
    ev = np.abs(np.linalg.eigvals(A))
    log(f'  selected model: rho(A) = {ev.max():.6f}, {(ev > 1).sum()} eigenvalue(s) outside the unit circle')

    # rollouts, named as mixed_rom_forcings.py names them (IC for test taus = Qhat[:, 0])
    Fo = OceanForcing(grid_dir=args.grid_dir, nx=248, ny=248, xo=0, yo=14, xc=31, yc=45,
                      dx=0.25, dy=0.25)
    Tmax = payload['Tmax']
    stem = Path(cfg['snapshot_dir_prefix']).name
    Qhat = np.load(src + 'Qhat.npy')
    runs = [('train', tau, payload['states'][i][:, 0]) for i, tau in enumerate(cfg['train_taus'])]
    runs += [('test', tau, Qhat[:, 0]) for tau in cfg['test_taus'] if float(tau) == 0.3]
    for kind, tau, q0 in runs:
        u = build_rom_inputs(Tmax, Fo.compute_wind_seasonal_amplitude(max_val=float(tau)),
                             n_year + cfg['n_year_predict'], n_days)
        q = model.predict(q0, niters=int((n_year + cfg['n_year_predict']) * 360 / n_days), inputs=u)
        np.save(out + f'Q_ROM_{kind}_{stem}{tau}.npy', q)
        ref = (payload['states'][cfg['train_taus'].index(tau)] if kind == 'train'
               else np.load(src + f'Q_PROJ_{stem}{tau}.npy'))
        n = min(ref.shape[1], q.shape[1])
        err = np.linalg.norm(q[:, :n] - ref[:, :n]) / np.linalg.norm(ref[:, :n])
        log(f'  {kind} tau{tau}: finite {np.all(np.isfinite(q))}, rel err vs '
            f'{"Qhat" if kind == "train" else "projection"} {err:.3f}')

    # shared inputs of the source run, linked (not copied) so analysis scripts find them
    for name in ['Tr.npy', 'Qhat.npy', 'eigs.npy'] + [p.name for p in Path(src).glob('Q_PROJ_*.npy')]:
        if not Path(out, name).exists():
            os.symlink(src + name, out + name)
    cfg_out = dict(cfg, source_run=src, conservation_fix=args.fix,
                   bc_direction=args.direction if args.fix == 'bc' else None,
                   stability_gate=(dict(taus=args.stability_taus, factor=args.stability_factor,
                                        bound=gate['stability_bound'],
                                        years=args.stability_years,
                                        norm_factor=args.stability_norm_factor,
                                        norm_bound=gate.get('stability_norm_bound'),
                                        rho_max=args.stability_rho_max)
                                   if gate else None),
                   selected_reg=dict(reg_A=reg_A, reg_H=reg_H, best_err=best_err),
                   grid=dict(A=args.grid_A, H=args.grid_H),
                   description=cfg.get('description', '') + f' | refit, imposed B/C fix={args.fix}')
    with open(out + 'config.json', 'w') as f:
        json.dump(cfg_out, f, indent=4)
    if payload['constraint'] is not None:
        check_conservation(model, payload['constraint'], payload['states'], payload['inputs'])
    log(f'wrote {out}')


if __name__ == '__main__':
    main()
