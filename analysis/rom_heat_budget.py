#!/usr/bin/env python3
"""ROM heat budget: self-consistency and accuracy of the OpInf ROM's volume heat content.

Heat content and the SST-restoring damping flux are LINEAR functionals of the state, and
the POD lift is linear, so both are exact linear functions of the reduced coordinates
(same commuting trick as compute_moc.py / utils.compute_barotropic_streamfunction):

    x = c + Q_c Tr q                     (Q_c: centered, unscaled training snapshots;
                                           alpha cancels row by row)
    H(x)       = H(c)      + h^T q,      h = Tr^T [H(x_n) - H(c)]_n
    F_theta(x) = F_theta(c) + f^T q,     f = Tr^T [F_theta(x_n) - F_theta(c)]_n

with H = rho0 cp sum theta rA drF hFacC (fixed thickness, no z* factor) and
F_theta = rho0 cp / tau sum rA drF1 hFacC1 theta_1. The physical restoring flux of any
state is F = F_star(t) - F_theta(x), F_star = rho0 cp / tau sum rA drF1 hFacC1 theta*.

The imposed B/C are rebuilt the same way from the surface functionals
G_b = sum theta_1 b_theta, G_c = sum theta_1 c_theta, G_s = sum S_1 s_star
(B[:,0] = Tr^T[G_b - G_b(c)], C = Tr^T[G_c - G_c(c)] + Tr^T[G_s - G_s(c)]) and checked
against model.h5, which validates h and f by construction.

Contracting the ROM update  q+ = B u + C + c + A q + H(q x q)  with h splits the heat
change per step into one term per operator. The per-operator identities that exact
conservation implies (for all q, u) are, with Dt the physical step:

    (i)   quadratic       h^T H            = 0
    (ii)  linear          h^T (A - I)      = -Dt f^T
    (iii) Tmax input      h^T B[:,0]       =  Dt P_b     (P_b = rho0 cp sum rA drF1 hFacC1 b_theta)
    (iv)  wind input      h^T B[:,1]       =  0
    (v)   constant        h^T (c + C)      =  Dt (P_c - F_theta(c))
          of which the salt part h^T C_salt should be 0

Inputs come from fom_heat_budget.py (heat_budget_fom_tau{T}.npz for the training taus
and the test tau). Serial.

    python3 src/rom_heat_budget.py --test-tau 0.3
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))  # core ROM modules
import paths  # noqa: E402
from MITgcmutils import rdmds  # noqa: E402
from forcing import OceanForcing, build_rom_inputs  # noqa: E402
from fom_heat_budget import rom_forcing_fields  # noqa: E402

DAY = 86400.0
RUN_DEFAULT = paths.BASE_RUN


def load_operators(run_dir):
    with h5py.File(run_dir + 'model.h5', 'r') as f:
        ops = [f[f'operator_{i}/entries'][()] for i in range(5)]
    B, C, c, A, H = ops
    return dict(B=B, C=C, c=c, A=A, H=H)


def ckron(q):
    """Compressed q (x) q, opinf ordering: for i, for j<=i: q_i q_j."""
    q = np.asarray(q)
    return np.concatenate([q[i] * q[:i + 1] for i in range(q.shape[0])], axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-dir', default=RUN_DEFAULT)
    p.add_argument('--budget-dir', default=paths.BUDGET_DIR)
    p.add_argument('--grid-dir', default=paths.GRID_DIR)
    p.add_argument('--test-tau', default='0.3')
    args = p.parse_args()

    run = args.run_dir
    pre = str(Path(run).parent) + '/'
    cfg = json.load(open(run + 'config.json'))
    ny = nx = 248
    nt_tr = 360 * cfg['n_year_train'] // cfg['n_days']
    dt = cfg['n_days'] * DAY

    # ---------------- grid and weights
    rac = np.squeeze(rdmds(args.grid_dir + 'RAC')).astype(float)
    drf = np.squeeze(rdmds(args.grid_dir + 'DRF')).astype(float)
    hfac = np.squeeze(rdmds(args.grid_dir + 'hFacC')).astype(float)
    nz = hfac.shape[0]
    fb = np.load(f"{args.budget_dir}heat_budget_fom_tau{args.test_tau}.npz")
    rho_cp = float(fb['rho0'] * fb['cp'])
    tau_th = float(fb['tau_theta'])
    area, vol = float(fb['area']), float(fb['volume'])
    w3 = rho_cp * rac[None] * drf[:, None, None] * hfac
    w_surf = rho_cp / tau_th * rac * drf[0] * hfac[0]
    c_th, b_th, s_st = rom_forcing_fields(args.grid_dir, ny, nx)
    P_c = rho_cp * np.sum(rac * drf[0] * hfac[0] * c_th)      # W
    P_b = rho_cp * np.sum(rac * drf[0] * hfac[0] * b_th)      # W per degC of Tmax
    zbot = np.cumsum(drf)
    kz = int(np.searchsorted(zbot, 700.0, side='right'))

    # ---------------- center functionals
    tag = f"{cfg['center_opt']}_{cfg['n_days']}days_{cfg['n_year_train']}yrs"
    with xr.open_dataset(pre + f'centerT_{tag}.nc', engine='h5netcdf') as ds:
        cT = np.squeeze(ds['center'].values).reshape(nz, ny, nx)
    with xr.open_dataset(pre + f'centerS_{tag}.nc', engine='h5netcdf') as ds:
        cS = np.squeeze(ds['center'].values).reshape(nz, ny, nx)
    Hk_c = np.einsum('kji,kji->k', cT, w3)
    H_c = Hk_c.sum()
    Fth_c = np.sum(w_surf * cT[0])
    Gb_c, Gc_c, Gs_c = np.sum(b_th * cT[0]), np.sum(c_th * cT[0]), np.sum(s_st * cS[0])

    # ---------------- reduced functionals from the training snapshots
    tr = [dict(np.load(f"{args.budget_dir}heat_budget_fom_tau{t}.npz"))
          for t in cfg['train_taus']]
    cat = lambda key: np.concatenate([d[key][:nt_tr] for d in tr], axis=0)  # noqa: E731
    Tr = np.load(run + 'Tr.npy')
    hk_tr = cat('hc_k')
    h = Tr.T @ (hk_tr.sum(1) - H_c)
    h_up = Tr.T @ (hk_tr[:, :kz].sum(1) - Hk_c[:kz].sum())
    h_dn = Tr.T @ (hk_tr[:, kz:].sum(1) - Hk_c[kz:].sum())
    f = Tr.T @ (cat('F_theta') - Fth_c)
    B0_chk = Tr.T @ (cat('G_btheta') - Gb_c)
    C_th = Tr.T @ (cat('G_ctheta') - Gc_c)
    C_s = Tr.T @ (cat('G_sstar') - Gs_c)

    ops = load_operators(run)
    B, C, c, A, H = ops['B'], ops['C'], ops['c'], ops['A'], ops['H']
    r = A.shape[0]
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)  # noqa: E731
    print('=== validation')
    print(f'  B[:,0] rebuilt vs model.h5: rel err {rel(B0_chk, B[:, 0]):.2e}')
    print(f'  C rebuilt (theta+salt) vs model.h5: rel err {rel(C_th + C_s, C):.2e}')
    Qhat = np.load(run + 'Qhat.npy')
    Hrec = H_c + h @ Qhat
    Htr = hk_tr.sum(1)
    print('  training heat content, projected vs FOM: rel RMS of anomaly '
          f'{np.sqrt(np.mean((Hrec - Htr) ** 2)) / np.std(Htr):.2e}')

    # ---------------- trajectories for the test tau
    tt = args.test_tau
    q_rom = np.load(run + f'Q_ROM_test_run_10yrspinup_tau{tt}.npy')
    q_prj = np.load(run + f'Q_PROJ_run_10yrspinup_tau{tt}.npy')
    nt = q_rom.shape[1]
    F = OceanForcing(grid_dir=args.grid_dir, nx=nx, ny=ny, xo=0, yo=14, xc=31, yc=45,
                     dx=0.25, dy=0.25)
    Tmax, _, _ = F.compute_temp_forcing()
    tau4 = F.compute_wind_seasonal_amplitude(max_val=float(tt))
    u = build_rom_inputs(Tmax, tau4, cfg['n_year_train'], cfg['n_days'])
    # one step of the ROM, term by term
    Aq = A @ q_rom[:, :-1]
    Hq = H @ ckron(q_rom[:, :-1])
    Bu0 = np.outer(B[:, 0], u[0, :nt - 1])
    Bu1 = np.outer(B[:, 1], u[1, :nt - 1])
    q_next = Bu0 + Bu1 + (C + c)[:, None] + Aq + Hq
    print(f'  rollout rebuilt from operators vs saved: rel err {rel(q_next, q_rom[:, 1:]):.2e}')

    terms = {
        'learned c': np.full(nt - 1, h @ c),
        'imposed C (theta)': np.full(nt - 1, h @ C_th),
        'imposed C (salt)': np.full(nt - 1, h @ C_s),
        'learned (A-I)q': h @ (Aq - q_rom[:, :-1]),
        'learned H(qq)': h @ Hq,
        'imposed B Tmax': h @ Bu0,
        'imposed B wind': h @ Bu1,
    }
    dH_rom = h @ np.diff(q_rom, axis=1)
    closure = sum(terms.values()) - dH_rom
    print(f'  sum of operator terms vs h^T dq: max |diff|/max|dH| '
          f'{np.max(np.abs(closure)) / np.max(np.abs(dH_rom)):.2e}')

    fom = dict(np.load(f"{args.budget_dir}heat_budget_fom_tau{tt}.npz"))
    Fstar = fom['F_star'][:nt]
    series = {
        'FOM': (fom['hc_k'][:nt].sum(1), Fstar - fom['F_theta'][:nt],
                fom['hc_k'][:nt, :kz].sum(1), fom['hc_k'][:nt, kz:].sum(1)),
        'projection': (H_c + h @ q_prj, Fstar - (Fth_c + f @ q_prj),
                       Hk_c[:kz].sum() + h_up @ q_prj, Hk_c[kz:].sum() + h_dn @ q_prj),
        'ROM': (H_c + h @ q_rom, Fstar - (Fth_c + f @ q_rom),
                Hk_c[:kz].sum() + h_up @ q_rom, Hk_c[kz:].sum() + h_dn @ q_rom),
    }
    wm2 = lambda x: x / (area * dt)  # noqa: E731
    yrs = np.arange(nt // 360)
    ann = lambda x: np.array([x[360 * i:360 * (i + 1)].mean() for i in yrs])  # noqa: E731
    trend = lambda x: np.polyfit(yrs, ann(x), 1)[0] / (360 * DAY * area)  # noqa: E731

    print('\n=== interpretation-free: self-consistency and accuracy '
          f'(tau{tt}, {nt // 360} yr, fluxes in basin-mean W/m^2)')
    print(f"  {'':12s}{'resid RMS':>11s}{'resid mean':>12s}{'cum resid':>12s}"
          f"{'net flux':>10s}{'dH/dt':>9s}{'seas amp':>9s}{'trend up':>9s}{'trend dn':>9s}")
    summary = {}
    for name, (Hs, Fs, Hu, Hd) in series.items():
        dH = np.diff(Hs)
        resid = dH - dt * 0.5 * (Fs[:-1] + Fs[1:])
        cum = Hs[-1] - Hs[0] - np.sum(dt * 0.5 * (Fs[:-1] + Fs[1:]))
        doy = np.array([Fs[d::360].mean() for d in range(360)])
        summary[name] = dict(
            resid_rms=np.sqrt(np.mean(wm2(resid) ** 2)), resid_mean=np.mean(wm2(resid)),
            cum_resid_degC=cum / (rho_cp * vol),
            net_flux=np.mean(Fs) / area, dHdt=(Hs[-1] - Hs[0]) / ((nt - 1) * dt * area),
            seas_amp=0.5 * np.ptp(doy) / area, trend_up=trend(Hu), trend_dn=trend(Hd))
        s = summary[name]
        print(f"  {name:12s}{s['resid_rms']:11.3e}{s['resid_mean']:12.3e}"
              f"{s['cum_resid_degC']:10.2e}°C{s['net_flux']:10.3f}{s['dHdt']:9.3f}"
              f"{s['seas_amp']:9.2f}{s['trend_up']:9.3f}{s['trend_dn']:9.3f}")
    Hf, Ff = series['FOM'][:2]
    for name in ('projection', 'ROM'):
        Hs, Fs = series[name][:2]
        print(f'  {name}: heat content error RMS {np.sqrt(np.mean((Hs - Hf) ** 2)) / (rho_cp * vol):.2e}°C'
              f' (final {(Hs[-1] - Hf[-1]) / (rho_cp * vol):+.2e}°C);'
              f' flux error RMS {np.sqrt(np.mean((Fs - Ff) ** 2)) / area:.3f} W/m^2')

    # ---------------- operator attribution along the ROM trajectory
    print('\n=== heat change per step by operator, ROM trajectory (W/m^2)')
    print(f"  {'term':20s}{'mean':>10s}{'seas amp':>10s}{'RMS':>10s}")
    for k, v in terms.items():
        v_w = wm2(v)
        doy = np.array([v_w[d::360].mean() for d in range(360)])
        print(f'  {k:20s}{v_w.mean():10.3f}{0.5 * np.ptp(doy):10.3f}{np.sqrt(np.mean(v_w ** 2)):10.3f}')
    phys_damp = -dt * (f @ q_rom[:, :-1])
    print(f"  {'[phys -Dt f^T q]':20s}{wm2(phys_damp).mean():10.3f}"
          f"{0.5 * np.ptp([wm2(phys_damp)[d::360].mean() for d in range(360)]):10.3f}")
    print(f"  {'[phys Dt(F*-Fth(c))]':20s}{wm2(dt * (Fstar[:-1] - Fth_c)).mean():10.3f}")

    # ---------------- per-operator identities under the three readings of Dt
    aT = float(np.load(pre + f"alpha_{cfg['center_opt']}_{cfg['scale']}_{cfg['n_year_train']}yrs.npy",
                       allow_pickle=True).item()['T'][0])
    hA = (A - np.eye(r)).T @ h
    dt_A = -(hA @ f) / (f @ f)
    lin_misfit = lambda d: np.linalg.norm(hA + d * f) / np.linalg.norm(hA)  # noqa: E731
    cosA = -(hA @ f) / (np.linalg.norm(hA) * np.linalg.norm(f))
    dt_A_traj = np.polyfit(-(f @ q_rom[:, :-1]), h @ (Aq - q_rom[:, :-1]), 1)[0]
    hB0, hB1 = h @ B[:, 0], h @ B[:, 1]
    hH = H.T @ h
    hHq = wm2(terms['learned H(qq)'])
    const_target = dt * (P_c - Fth_c)
    print('\n=== per-operator identities')
    print(f'  alpha_T = {aT:.4f}   alpha_T^2 = {aT ** 2:.2f}   Dt = {dt:.0f} s')
    print(f'  P_b (Tmax pattern heat) = {P_b / area:.4e} W/m^2 per degC;'
          f'  P_c = {P_c / area:.4e} W/m^2;  F_theta(c) = {Fth_c / area:.3f} W/m^2')
    print(f'  (i)   |H^T h| / (|h| |H|_F) = {np.linalg.norm(hH) / (np.linalg.norm(h) * np.linalg.norm(H)):.3e};'
          f'  H term along trajectory RMS {np.sqrt(np.mean(hHq ** 2)):.3f} W/m^2')
    print(f'  (ii)  cos(h^T(A-I), -f) = {cosA:.4f};  best-fit Dt_A = {dt_A:.4e} s'
          f' (= {dt_A / DAY:.4f} d);  along-trajectory Dt_A = {dt_A_traj:.4e} s')
    print(f'        misfit |h^T(A-I)+Dt f|/|h^T(A-I)|: at Dt_A {lin_misfit(dt_A):.3f},'
          f' at 86400 s {lin_misfit(DAY):.3f}')
    print(f'  (iii) h^T B[:,0] = {hB0:.4e} J/degC;  Dt_B := h^T B[:,0] / P_b = {hB0 / P_b:.4e} s')
    print(f'        basis capture of b_theta: h^T B[:,0] / (alpha_T^2 P_b) = {hB0 / (aT ** 2 * P_b):.4f}')
    print(f'  (iv)  h^T B[:,1] = {hB1:.4e} J per (N/m^2); wind heat term RMS '
          f'{np.sqrt(np.mean(wm2(terms["imposed B wind"]) ** 2)):.3f} W/m^2')
    print(f'  (v)   h^T c = {wm2(h @ c):.3f}, h^T C_theta = {wm2(h @ C_th):.3e}, h^T C_salt = '
          f'{wm2(h @ C_s):.3e} W/m^2;  target Dt(P_c - F_theta(c)) = {wm2(const_target):.3f} W/m^2')
    print(f'        Dt_C := h^T C_theta / P_c = {(h @ C_th) / P_c:.4e} s;  capture of c_theta '
          f'{(h @ C_th) / (aT ** 2 * P_c):.4f}')
    # ROM seasonal forcing timing vs FOM: P_c + P_b Tmax(t) vs recomputed F_star
    fs_rom = P_c + P_b * u[0, :nt]
    print(f'  ROM input Tmax(t) vs FOM theta*(t): RMS diff of implied F_star '
          f'{np.sqrt(np.mean((fs_rom - Fstar) ** 2)) / area:.3f} W/m^2 '
          f'(F_star seasonal amp {0.5 * np.ptp(Fstar[:360]) / area:.2f})')

    out = f"{args.budget_dir}heat_budget_rom_tau{tt}.npz"
    np.savez(out, h=h, h_up=h_up, h_dn=h_dn, f=f, C_theta=C_th, C_salt=C_s, H_c=H_c,
             Fth_c=Fth_c, P_b=P_b, P_c=P_c, alpha_T=aT, dt=dt, dt_A=dt_A,
             **{f'term_{k.replace(" ", "_")}': v for k, v in terms.items()},
             **{f'{n}_H': s[0] for n, s in series.items()},
             **{f'{n}_F': s[1] for n, s in series.items()})
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
