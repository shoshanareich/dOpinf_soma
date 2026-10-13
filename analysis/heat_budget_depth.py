#!/usr/bin/env python3
"""Full-depth vs upper-layer heat budget, FOM vs ROM, for several wind amplitudes.

For a layer L (a set of whole model levels) the heat content is a linear functional of
the state, so as in rom_heat_budget.py the ROM value is exact in reduced coordinates:

    H_L(x) = H_L(c) + h_L^T q,    h_L = Tr^T [H_L(x_n) - H_L(c)]_n

(cell volume rA * DRF[k] * hFacC[k,j,i], time-invariant: no z* stretching). The surface
restoring flux F(t) = rho0 cp / tau_theta sum rA DRF_1 hFacC_1 (theta* - theta_1) is shared
by both layers. Per 30-day month:

    full depth:  R = dH_full - int F dt           self-consistency residual (0 if conservative)
    upper layer: X = (int F dt - dH_upper) / dt   inferred heat flux DOWN across the layer
                                                  base (advection + diffusion + GM + KPP)

For the ROM, X also absorbs any heat the ROM creates inside the upper layer; there is no
independent interface flux to separate the two (R bounds the total spurious source).

The per-operator split of the ROM heat change uses h_L for each layer, so the upper-layer
terms show which operator moves heat into / out of the upper layer.

Reads heat_budget_fom_tau{T}.npz (fom_heat_budget.py) for the training taus (to build h)
and each requested tau. Writes a figure (columns = tau; rows chosen with --rows) and
prints the numbers.

    python3 src/heat_budget_depth.py --taus 0.2 0.3 0.4 --kmax 21
    python3 src/heat_budget_depth.py --rows full sflux --out figs/heat_budget/heat_content_sflux_taus.pdf
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
from MITgcmutils import rdmds  # noqa: E402
from forcing import OceanForcing, build_rom_inputs  # noqa: E402
from rom_heat_budget import RUN_DEFAULT, DAY, ckron, load_operators  # noqa: E402

FS_TICK, FS_LABEL = 7.5, 8.5
C_ROM = "#0072B2"             # MITgcm dashed and OpInf ROM solid, same blue
MONTH = 30


def monthly(x_daily_flux, H, dt):
    """Per 30-day month: trapezoid integral of a daily flux [J] and change of H [J]."""
    n = (len(H) - 1) // MONTH
    idx = np.arange(n + 1) * MONTH
    intF = np.array([np.sum(dt * 0.5 * (x_daily_flux[a:a + MONTH] + x_daily_flux[a + 1:a + MONTH + 1]))
                     for a in idx[:-1]])
    return intF, np.diff(H[idx])


def cumF_fom(F, dt):
    """Trapezoid running integral of a daily flux [J]."""
    return np.concatenate([[0.0], np.cumsum(dt * 0.5 * (F[:-1] + F[1:]))])


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--taus', nargs='+', default=['0.2', '0.3', '0.4'])
    p.add_argument('--kmax', type=int, default=21,
                   help='upper layer = levels [0, kmax); 21 -> base at 954.5 m')
    p.add_argument('--run-dir', default=RUN_DEFAULT)
    p.add_argument('--budget-dir', default=paths.BUDGET_DIR)
    p.add_argument('--grid-dir', default=paths.GRID_DIR)
    p.add_argument('--rows', nargs='+', default=['full', 'upper', 'xflux'],
                   choices=['full', 'upper', 'sflux', 'xflux', 'resid'],
                   help='figure rows: heat content (full/upper), implied surface flux, '
                        'inferred flux across the upper-layer base, cumulative budget residual')
    p.add_argument('--resid-units', choices=['J', 'Wm2'], default='J',
                   help='resid row: cumulative ZJ, or R(t)/(area*t) = running-mean '
                        'residual flux in W/m^2 (first month omitted)')
    p.add_argument('--fom-color', default=C_ROM, help="MITgcm line colour (e.g. 'k')")
    p.add_argument('--sflux-ylim', type=float, nargs=2, default=None)
    p.add_argument('--resid-ylim', type=float, nargs=2, default=None)
    p.add_argument('--consistent-units', action='store_true',
                   help='surface flux row in ZJ/yr, so that (c) = [(a) - (a)(0)] - int (b) dt '
                        'in one unit (with --resid-units J); W m^-2 on a right-hand axis')
    p.add_argument('--zstar', action='store_true',
                   help='resid row: add MITgcm with the z* thickness factor in the heat content '
                        '(same daily recomputed flux)')
    p.add_argument('--exact-resid', action='store_true',
                   help='resid row: MITgcm = z* heat content minus the exact monthly TRELAX '
                        'integral (fom_relax_monthly.py), at month ends; OpInf = flux summed '
                        'per step (explicit, as the conservation constraint) with the ROM\'s '
                        'own imposed target P_c + P_b Tmax(t) (also used in the sflux row)')
    p.add_argument('--resid-lw', type=float, default=1.3, help='MITgcm line width, resid row')
    p.add_argument('--no-projection', action='store_true',
                   help='omit the projection-only curve from the residual row')
    p.add_argument('--label', default='OpInf ROM', help='legend name of the --run-dir ROM')
    p.add_argument('--run-dir2', default=None, help='optional second ROM, overlaid in all rows')
    p.add_argument('--label2', default='OpInf ROM 2', help='legend name of the --run-dir2 ROM')
    p.add_argument('--color2', default='#D55E00', help='line colour of the --run-dir2 ROM')
    p.add_argument('--out', default='figs/heat_budget/heat_budget_depth_taus.pdf')
    args = p.parse_args()

    def build(run):
        """FOM + ROM budget series for one ROM run (per tau, in data[tau])."""
        bd, kz = args.budget_dir, args.kmax
        pre = str(Path(run).parent) + '/'
        cfg = json.load(open(run + 'config.json'))
        nt_tr = 360 * cfg['n_year_train'] // cfg['n_days']
        dt = cfg['n_days'] * DAY
        ny = nx = 248

        rac = np.squeeze(rdmds(args.grid_dir + 'RAC')).astype(float)
        drf = np.squeeze(rdmds(args.grid_dir + 'DRF')).astype(float)
        hfac = np.squeeze(rdmds(args.grid_dir + 'hFacC')).astype(float)
        nz = hfac.shape[0]
        zbase = np.cumsum(drf)[kz - 1]
        f0 = np.load(f'{bd}heat_budget_fom_tau{args.taus[0]}.npz')
        rho_cp, tau_th, area = float(f0['rho0'] * f0['cp']), float(f0['tau_theta']), float(f0['area'])
        w3 = rho_cp * rac[None] * drf[:, None, None] * hfac
        w_surf = rho_cp / tau_th * rac * drf[0] * hfac[0]
        layers = {'full depth': slice(0, nz), f'upper {zbase:.0f} m': slice(0, kz)}

        # centre + reduced functionals (per layer) from the training snapshots
        tag = f"{cfg['center_opt']}_{cfg['n_days']}days_{cfg['n_year_train']}yrs"
        with xr.open_dataset(pre + f'centerT_{tag}.nc', engine='h5netcdf') as ds:
            cT = np.squeeze(ds['center'].values).reshape(nz, ny, nx)
        Hk_c = np.einsum('kji,kji->k', cT, w3)
        Fth_c = np.sum(w_surf * cT[0])
        tr = [np.load(f'{bd}heat_budget_fom_tau{t}.npz') for t in cfg['train_taus']]
        hk_tr = np.concatenate([d['hc_k'][:nt_tr] for d in tr])
        fth_tr = np.concatenate([d['F_theta'][:nt_tr] for d in tr])
        Tr = np.load(run + 'Tr.npy')
        hL = {L: Tr.T @ (hk_tr[:, s].sum(1) - Hk_c[s].sum()) for L, s in layers.items()}
        HcL = {L: Hk_c[s].sum() for L, s in layers.items()}
        f = Tr.T @ (fth_tr - Fth_c)

        ops = load_operators(run)
        B, C, c, A, Hq_op = ops['B'], ops['C'], ops['c'], ops['A'], ops['H']
        Qhat = np.load(run + 'Qhat.npy')
        F = OceanForcing(grid_dir=args.grid_dir, nx=nx, ny=ny, xo=0, yo=14, xc=31, yc=45,
                         dx=0.25, dy=0.25)
        Tmax, _, _ = F.compute_temp_forcing()
        train = [float(t) for t in cfg['train_taus']]
        if args.exact_resid:
            from refit_conservation import fom_theta_patterns
            _, _, c_star, b_star = fom_theta_patterns(args.grid_dir)
            a1 = (rac * drf[0] * hfac[0]).ravel()
            P_c, P_b = rho_cp * np.sum(a1 * c_star), rho_cp * np.sum(a1 * b_star)   # W, W/degC

        wm2 = lambda x: x / (area * dt)  # noqa: E731
        L_full, L_up = list(layers)
        data = {}
        for tau in args.taus:
            kind = 'train' if float(tau) in train else 'test'
            q = np.load(run + f'Q_ROM_{kind}_run_10yrspinup_tau{tau}.npy')
            nt = q.shape[1]
            # projection-only baseline: the FOM squeezed through the basis, no dynamics
            if kind == 'train':
                i0 = train.index(float(tau)) * nt_tr
                q_prj = Qhat[:, i0:i0 + nt]
            else:
                q_prj = np.load(run + f'Q_PROJ_run_10yrspinup_tau{tau}.npy')
            fom = np.load(f'{bd}heat_budget_fom_tau{tau}.npz')
            u = build_rom_inputs(Tmax, F.compute_wind_seasonal_amplitude(max_val=float(tau)),
                                 cfg['n_year_train'], cfg['n_days'])
            Aq, HQ = A @ q[:, :-1], Hq_op @ ckron(q[:, :-1])
            Bu0, Bu1 = np.outer(B[:, 0], u[0, :nt - 1]), np.outer(B[:, 1], u[1, :nt - 1])
            q_next = Bu0 + Bu1 + (C + c)[:, None] + Aq + HQ
            err = np.linalg.norm(q_next - q[:, 1:]) / np.linalg.norm(q[:, 1:])
            print(f'tau{tau} ({kind}): rollout rebuilt from operators, rel err {err:.1e}')
            Fstar = fom['F_star'][:nt]
            d = dict(kind=kind, nt=nt,
                     F={'MITgcm': Fstar - fom['F_theta'][:nt],
                        'projection': Fstar - (Fth_c + f @ q_prj),
                        'OpInf': Fstar - (Fth_c + f @ q)})
            if args.exact_resid:     # the flux the ROM applies: its own imposed theta* target
                d['F']['OpInf'] = P_c + P_b * u[0, :nt] - (Fth_c + f @ q)
            for L, sl in layers.items():
                d[L] = {'MITgcm': fom['hc_k'][:nt, sl].sum(1),
                        'projection': HcL[L] + hL[L] @ q_prj,
                        'OpInf': HcL[L] + hL[L] @ q}
                d[L, 'terms'] = {'c': hL[L] @ c, 'C': hL[L] @ C,
                                 '(A-I)q': hL[L] @ (Aq - q[:, :-1]), 'H(qq)': hL[L] @ HQ,
                                 'B Tmax': hL[L] @ Bu0, 'B wind': hL[L] @ Bu1}
            # cumulative budget residual  R(t) = H(t) - H(0) - int_0^t F dt   (full depth)
            d['resid'] = {}
            for m in ('MITgcm', 'projection', 'OpInf'):
                Hm, Fm = d[L_full][m], d['F'][m]
                cumF = np.concatenate([[0.0], np.cumsum(dt * 0.5 * (Fm[:-1] + Fm[1:]))])
                d['resid'][m] = Hm - Hm[0] - cumF
            # z*: exact MITgcm heat content (rStarFacC), same daily recomputed flux
            Hz = fom['hc_zstar'][:nt]
            d['resid']['MITgcm z*'] = Hz - Hz[0] - cumF_fom(d['F']['MITgcm'], dt)
            # same budget with the flux summed explicitly (left point), the form the
            # conservation constraint wT(q_k+1 - q_k) = dt F(q_k) is imposed in
            d['resid_euler'] = {m: d[L_full][m] - d[L_full][m][0]
                                - np.concatenate([[0.0], np.cumsum(dt * d['F'][m][:-1])])
                                for m in ('MITgcm', 'OpInf')}
            if args.exact_resid:
                d['resid']['OpInf'] = d['resid_euler']['OpInf']
                rm = np.load(f'{bd}relax_monthly_fom_tau{tau}.npz')
                t_end = rm['interval'][:, 1]
                k = np.searchsorted(fom['time'], t_end)
                ok = k < len(fom['time'])
                assert np.allclose(fom['time'][k[ok]], t_end[ok]) and fom['time'][0] == rm['interval'][0, 0]
                Hz = fom['hc_zstar']
                d['resid_month'] = (np.concatenate([[0], k[ok]]) / 360.0,
                                    np.concatenate([[0.0], Hz[k[ok]] - Hz[0]
                                                    - np.cumsum(rm['heat_int'][ok])]))
            # inferred flux down across the upper-layer base, monthly
            d['X'] = {}
            for m in ('MITgcm', 'OpInf'):
                iF, dH = monthly(d['F'][m], d[L_up][m], dt)
                d['X'][m] = (iF - dH) / (area * MONTH * dt)
            data[tau] = d
        return dict(data=data, dt=dt, area=area, zbase=zbase, layers=layers, L_full=L_full,
                    L_up=L_up, wm2=wm2)

    ns = build(args.run_dir)
    data, dt, area, zbase, layers, L_full, L_up, wm2 = (ns[k] for k in (
        'data', 'dt', 'area', 'zbase', 'layers', 'L_full', 'L_up', 'wm2'))
    rom_methods = ('OpInf',)
    if args.run_dir2:      # second ROM: its OpInf series join data[tau] as OpInf2
        for tau, d2 in build(args.run_dir2)['data'].items():
            d = data[tau]
            d['F']['OpInf2'] = d2['F']['OpInf']
            d['resid']['OpInf2'] = d2['resid']['OpInf']
            for Lname in layers:
                d[Lname]['OpInf2'] = d2[Lname]['OpInf']
            d['X']['OpInf2'] = d2['X']['OpInf']
        rom_methods = ('OpInf', 'OpInf2')

    # ---------------- figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style = {'MITgcm': dict(color=args.fom_color, lw=1.3, ls=':', zorder=3),
             'OpInf': dict(color=C_ROM, lw=1.1, zorder=2),
             'projection': dict(color='#888888', lw=0.9),
             'MITgcm z*': dict(color='#d95f02', lw=1.1, ls='-.', zorder=3)}
    style['OpInf2'] = dict(color=args.color2, lw=1.1, zorder=2)
    label = {'MITgcm': 'MITgcm', 'OpInf': args.label, 'OpInf2': args.label2,
             'projection': 'projection only',
             'MITgcm z*': 'MITgcm, z* thickness'}
    ylab = {'full': 'Heat content anomaly (ZJ)',
            'upper': f'Heat content anomaly,\nupper {zbase:.0f} m (ZJ)',
            'sflux': ('Surface heat flux\n(ZJ yr$^{-1}$)' if args.consistent_units
                      else r'Implied surface flux (W m$^{-2}$)'),
            'xflux': f'Heat flux down across\n{zbase:.0f} m' + r' (W m$^{-2}$)',
            'resid': ('Heat budget residual (ZJ)' if args.resid_units == 'J'
                      else r'Heat budget residual (W m$^{-2}$)')}
    zj_yr = 360 * DAY / 1e21     # W -> ZJ/yr
    nr, ntau = len(args.rows), len(args.taus)
    fig, axs = plt.subplots(nr, ntau, figsize=(2.6 * ntau + 0.7, 2.1 * nr + 0.2), sharex=True,
                            sharey='row', constrained_layout=True, squeeze=False)
    for j, tau in enumerate(args.taus):
        d = data[tau]
        t = np.arange(d['nt']) / 360.0
        tm = (np.arange(len(d['X']['OpInf'])) + 0.5) * MONTH / 360.0
        for i, rowname in enumerate(args.rows):
            ax = axs[i, j]
            if rowname in ('full', 'upper'):
                L = L_full if rowname == 'full' else L_up
                H0 = d[L]['MITgcm'][0]
                for m in ('MITgcm',) + rom_methods:
                    ax.plot(t, (d[L][m] - H0) / 1e21, label=label[m], **style[m])
            elif rowname == 'sflux':
                for m in ('MITgcm',) + rom_methods:
                    ax.plot(t, d['F'][m] * zj_yr if args.consistent_units else d['F'][m] / area,
                            label=label[m], **style[m])
                if args.sflux_ylim:
                    ax.set_ylim(*args.sflux_ylim)
                if args.consistent_units and j == ntau - 1:
                    k = zj_yr * area             # (ZJ/yr) per (W m^-2)
                    sec = ax.secondary_yaxis('right', functions=(lambda x: x / k, lambda y: y * k))
                    sec.set_ylabel('Area-mean flux\n(W m$^{-2}$)', fontsize=FS_LABEL)
                    sec.tick_params(labelsize=FS_TICK)
            elif rowname == 'xflux':
                for m in ('MITgcm',) + rom_methods:
                    ax.plot(tm, d['X'][m], label=label[m], **style[m])
            elif rowname == 'resid':
                ms = (('MITgcm',) if args.no_projection else ('projection', 'MITgcm')) + rom_methods
                if args.zstar:
                    ms = ms + ('MITgcm z*',)
                for m in ms:
                    st = dict(style[m], lw=args.resid_lw, ls='--') if m == 'MITgcm' else style[m]
                    if m == 'MITgcm' and args.exact_resid:
                        tmo, Rmo = d['resid_month']
                        ax.plot(tmo, Rmo / 1e21, label=label[m], **st)
                        continue
                    if args.resid_units == 'J':
                        ax.plot(t, d['resid'][m] / 1e21, label=label[m], **st)
                    else:
                        k0 = MONTH
                        ax.plot(t[k0:], d['resid'][m][k0:] / (area * dt * np.arange(k0, d['nt'])),
                                label=label[m], **st)
                if args.resid_ylim:
                    ax.set_ylim(*args.resid_ylim)
            if rowname in ('sflux', 'xflux', 'resid'):
                ax.axhline(0, color='#999999', lw=0.5, zorder=1)
        axs[0, j].set_title(rf'$\tau$={tau} ({"training" if d["kind"] == "train" else "held out"})',
                            fontsize=FS_LABEL)
        axs[-1, j].set_xlabel('year', fontsize=FS_LABEL)
        axs[-1, j].set_xlim(0, 10)
    for i, rowname in enumerate(args.rows):
        axs[i, 0].set_ylabel(ylab[rowname], fontsize=FS_LABEL)
        axs[i, 0].text(-0.3, 1.0, f'({"abcdefg"[i]})', transform=axs[i, 0].transAxes,
                       fontsize=FS_LABEL, fontweight='bold', va='top')
        axs[i, 0].legend(fontsize=FS_TICK, frameon=False, loc='best')
    for ax in axs.flat:
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, color='#e6e6e6', lw=0.5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f'wrote {args.out}')

    # ---------------- numbers (basin-mean W/m^2 unless noted)
    print(f"\n{'tau':>5s} {'layer':>12s} {'dH/dt FOM':>10s} {'ROM':>7s} {'F FOM':>7s} {'ROM':>7s}"
          f" {'H err yr10':>11s}  [ZJ]")
    for tau, d in data.items():
        for L in layers:
            Hf, Hr = d[L]['MITgcm'], d[L]['OpInf']
            n = len(Hf) - 1
            print(f"{tau:>5s} {L:>12s} {wm2((Hf[-1] - Hf[0]) / n):10.3f} {wm2((Hr[-1] - Hr[0]) / n):7.3f}"
                  f" {d['F']['MITgcm'].mean() / area:7.3f} {d['F']['OpInf'].mean() / area:7.3f}"
                  f" {(Hr[-1] - Hf[-1]) / 1e21:11.3f}")
    print(f"\ncumulative residual dH - int F at yr 10 [ZJ] (and as mean W/m^2)")
    for tau, d in data.items():
        n = d['nt'] - 1
        print(f"  tau{tau}: " + ', '.join(
            f"{m} {d['resid'][m][-1] / 1e21:+.3f} ({wm2(d['resid'][m][-1] / n):+.3f})"
            for m in ('MITgcm', 'MITgcm z*', 'projection', 'OpInf')))
        if args.exact_resid:
            Rmo = d['resid_month'][1]
            print(f"     exact: MITgcm (z* + TRELAX) end {Rmo[-1] / 1e21:+.2e} ZJ, max |R| "
                  f"{np.abs(Rmo).max() / 1e21:.2e}; OpInf (explicit, own target) end "
                  f"{d['resid']['OpInf'][-1] / 1e21:+.2e}, max |R| {np.abs(d['resid']['OpInf']).max() / 1e21:.2e}")
    print(f"\nsame residual with the flux summed explicitly (left point) [ZJ]: yr-10 value, max |R|")
    for tau, d in data.items():
        print(f"  tau{tau}: " + ', '.join(
            f"{m} {d['resid_euler'][m][-1] / 1e21:+.2e} / {np.abs(d['resid_euler'][m]).max() / 1e21:.2e}"
            for m in ('MITgcm', 'OpInf'))
            + f"   | trapezoid OpInf + dt/2 [F(t)-F(0)] max |.| "
              f"{np.abs(d['resid']['OpInf'] + 0.5 * dt * (d['F']['OpInf'] - d['F']['OpInf'][0])).max() / 1e21:.2e}")
    print(f"\nflux down across {zbase:.0f} m (monthly): mean FOM/ROM, seasonal amp FOM/ROM, corr")
    for tau, d in data.items():
        Xf, Xr = d['X']['MITgcm'], d['X']['OpInf']
        amp = lambda x: 0.5 * np.ptp([x[m::12].mean() for m in range(12)])  # noqa: E731
        print(f"  tau{tau}: {Xf.mean():.2f}/{Xr.mean():.2f}  {amp(Xf):.2f}/{amp(Xr):.2f}"
              f"  {np.corrcoef(Xf, Xr)[0, 1]:+.2f}")


if __name__ == '__main__':
    main()
