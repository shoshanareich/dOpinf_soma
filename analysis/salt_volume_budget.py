#!/usr/bin/env python3
"""Salt and volume budgets, FOM vs ROM, for several wind amplitudes.

Companion to heat_budget_depth.py (--rows full sflux resid --resid-units Wm2). Salt
content and basin-integrated eta are linear functionals of the state, so as in
rom_heat_budget.py the ROM value is exact in reduced coordinates:

    S_tot(x) = S_tot(c) + s^T q,     s = Tr^T [S_tot(x_n) - S_tot(c)]_n
    F_S(x)   = F_S* - (F_S1(c) + g^T q),  g = Tr^T [F_S1(x_n) - F_S1(c)]_n
    V'(x)    = V'(c) + v^T q,        v = Tr^T [V'(x_n) - V'(c)]_n

(S_tot = sum S rA DRF hFacC, fixed thickness; F_S1 = 1/tau_S sum rA DRF_1 hFacC_1 S_1;
V' = sum rA eta.) SOMA has no real freshwater flux, so:

    salt:   R_S(t) = S_tot(t) - S_tot(0) - int_0^t F_S dt     (0 if conservative)
    volume: R_V(t) = V'(t) - V'(0)                             (surface flux is 0)

Salt fluxes and residuals are shown as the equivalent freshwater flux -F_S / (S0 area),
S0 = 35 psu, in mm/yr (positive = freshening). The salt residual is plotted cumulatively as
the equivalent basin-mean salinity R_S(t)/V (same units as row a; --resid-units cum) or as
the running mean R(t)/t in mm/yr freshwater (--resid-units rate).

Figures (columns = tau):
    salt   3 rows: basin-mean salinity anomaly, implied surface freshwater flux, residual
    volume 2 rows: basin-mean eta, running-mean spurious volume source

Reads salt_budget_fom_tau{T}.npz (fom_salt_budget.py) for the training taus (to build
s, g, v) and each requested tau.

    python3 src/salt_volume_budget.py --taus 0.2 0.3 0.4
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
from rom_heat_budget import RUN_DEFAULT, DAY  # noqa: E402
from heat_budget_depth import FS_TICK, FS_LABEL, C_ROM  # noqa: E402

S0 = 35.0
YEAR = 360 * DAY
MM_YR = 1e3 * YEAR               # m/s -> mm/yr
MONTH = 30
METHODS = ('MITgcm', 'projection', 'OpInf')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--taus', nargs='+', default=['0.2', '0.3', '0.4'])
    p.add_argument('--run-dir', default=RUN_DEFAULT)
    p.add_argument('--budget-dir', default=paths.BUDGET_DIR)
    p.add_argument('--grid-dir', default=paths.GRID_DIR)
    p.add_argument('--fom-color', default=C_ROM, help="MITgcm line colour (e.g. 'k')")
    p.add_argument('--sflux-ylim', type=float, nargs=2, default=None)
    p.add_argument('--resid-ylim', type=float, nargs=2, default=None)
    p.add_argument('--resid-units', choices=['cum', 'rate'], default='cum',
                   help='salt residual row: cumulative R_S/V in 1e-3 psu, or running mean '
                        'R_S/t in mm/yr freshwater (first month omitted)')
    p.add_argument('--vol-resid-ylim', type=float, nargs=2, default=None)
    p.add_argument('--consistent-units', action='store_true',
                   help='row (b) as the basin-mean salinity tendency (1e-3 psu/yr), so that '
                        '(c) = [(a) - (a)(0)] - int (b) dt in one unit; freshwater mm/yr on a '
                        'right-hand axis')
    p.add_argument('--zstar', action='store_true',
                   help='salt resid row: add MITgcm with the z* thickness factor in the salt '
                        'content (same daily recomputed flux)')
    p.add_argument('--exact-resid', action='store_true',
                   help='salt resid row (cum): MITgcm = z* salt content minus the exact monthly '
                        'restoring integral (fom_relax_monthly.py), at month ends; OpInf = flux '
                        'summed per step (explicit, as the conservation constraint)')
    p.add_argument('--psu-sci', type=int, nargs=3, default=None, metavar=('EA', 'EB', 'EC'),
                   help='salt figure in psu (psu/yr) with the power of ten fixed per row and '
                        'shown as x10^E at the top of the y axis, e.g. -3 -3 -7; ylims are '
                        'still given in 1e-3 psu')
    p.add_argument('--resid-lw', type=float, default=1.3, help='MITgcm line width, resid rows')
    p.add_argument('--no-projection', action='store_true',
                   help='omit the projection-only curve from the residual rows')
    p.add_argument('--label', default='OpInf ROM', help='legend name of the --run-dir ROM')
    p.add_argument('--run-dir2', default=None, help='optional second ROM, overlaid in all rows')
    p.add_argument('--label2', default='OpInf ROM 2', help='legend name of the --run-dir2 ROM')
    p.add_argument('--color2', default='#D55E00', help='line colour of the --run-dir2 ROM')
    p.add_argument('--out-salt', default='figs/heat_budget/salt_content_sflux_residual_taus.pdf')
    p.add_argument('--out-vol', default='figs/heat_budget/volume_eta_residual_taus.pdf')
    args = p.parse_args()

    def build(run):
        """FOM + ROM salt/volume series for one ROM run (per tau, in data[tau])."""
        bd = args.budget_dir
        pre =str(Path(run).parent) + '/'
        cfg = json.load(open(run + 'config.json'))
        nt_tr = 360 * cfg['n_year_train'] // cfg['n_days']
        dt = cfg['n_days'] * DAY
        tr = [np.load(f'{bd}salt_budget_fom_tau{t}.npz') for t in cfg['train_taus']]
        f0 = tr[0]
        area, vol = float(f0['area']), float(f0['volume'])
        drf = f0['drf']
        nz, ny, nx = len(drf), 248, 248

        # centre functionals
        grid_dir = args.grid_dir
        rac = np.squeeze(rdmds(grid_dir + 'RAC')).astype(float)
        hfac = np.squeeze(rdmds(grid_dir + 'hFacC')).astype(float)
        w3 = rac[None] * drf[:, None, None] * hfac
        w_surf = rac * drf[0] * hfac[0] / float(f0['tau_s'])
        wet = rac * (hfac[0] > 0)
        tag = f"{cfg['center_opt']}_{cfg['n_days']}days_{cfg['n_year_train']}yrs"
        with xr.open_dataset(pre + f'centerS_{tag}.nc', engine='h5netcdf') as ds:
            cS = np.squeeze(ds['center'].values).reshape(nz, ny, nx)
        with xr.open_dataset(pre + f'centerEta_{tag}.nc', engine='h5netcdf') as ds:
            cE = np.squeeze(ds['center'].values).reshape(ny, nx)
        Sc_c, Fs_c, V_c = np.sum(w3 * cS), np.sum(w_surf * cS[0]), np.sum(wet * cE)

        Tr = np.load(run + 'Tr.npy')
        cat = lambda key: np.concatenate([d[key][:nt_tr] for d in tr], axis=0)  # noqa: E731
        Sc_tr, Fs_tr, V_tr = cat('sc_k').sum(1), cat('FS_S'), cat('eta_int')
        s, g, v = Tr.T @ (Sc_tr - Sc_c), Tr.T @ (Fs_tr - Fs_c), Tr.T @ (V_tr - V_c)
        Qhat = np.load(run + 'Qhat.npy')
        for name, ref, c0, w in (('salt content', Sc_tr, Sc_c, s), ('F_S1', Fs_tr, Fs_c, g),
                                 ('volume', V_tr, V_c, v)):
            rec = c0 + w @ Qhat
            print(f'training {name}, projected vs FOM: rel RMS of anomaly '
                  f'{np.sqrt(np.mean((rec - ref) ** 2)) / max(np.std(ref), 1e-30):.2e}'
                  f'  (FOM std {np.std(ref):.3e})')

        train = [float(t) for t in cfg['train_taus']]
        data = {}
        for tau in args.taus:
            kind = 'train' if float(tau) in train else 'test'
            q = np.load(run + f'Q_ROM_{kind}_run_10yrspinup_tau{tau}.npy')
            nt = q.shape[1]
            if kind == 'train':
                i0 = train.index(float(tau)) * nt_tr
                q_prj = Qhat[:, i0:i0 + nt]
            else:
                q_prj = np.load(run + f'Q_PROJ_run_10yrspinup_tau{tau}.npy')
            fom = np.load(f'{bd}salt_budget_fom_tau{tau}.npz')
            Fstar = float(fom['FS_star'])
            d = dict(kind=kind, nt=nt,
                     S={'MITgcm': fom['sc_k'][:nt].sum(1), 'projection': Sc_c + s @ q_prj,
                        'OpInf': Sc_c + s @ q},
                     F={'MITgcm': Fstar - fom['FS_S'][:nt], 'projection': Fstar - (Fs_c + g @ q_prj),
                        'OpInf': Fstar - (Fs_c + g @ q)},
                     V={'MITgcm': fom['eta_int'][:nt], 'projection': V_c + v @ q_prj,
                        'OpInf': V_c + v @ q},
                     Sz=fom['sc_zstar'][:nt])
            d['RS'], d['RV'] = {}, {}
            for m in METHODS:
                Fm = d['F'][m]
                cumF = np.concatenate([[0.0], np.cumsum(dt * 0.5 * (Fm[:-1] + Fm[1:]))])
                d['RS'][m] = d['S'][m] - d['S'][m][0] - cumF
                d['RV'][m] = d['V'][m] - d['V'][m][0]
            # z* (exact MITgcm) salt budget, for reference
            Fm = d['F']['MITgcm']
            cumF = np.concatenate([[0.0], np.cumsum(dt * 0.5 * (Fm[:-1] + Fm[1:]))])
            d['RSz'] = d['Sz'] - d['Sz'][0] - cumF
            d['RS']['MITgcm z*'] = d['RSz']
            # same budget with the flux summed explicitly (left point), the form the
            # conservation constraint wT(q_k+1 - q_k) = dt F(q_k) is imposed in
            d['RS_euler'] = {m: d['S'][m] - d['S'][m][0]
                             - np.concatenate([[0.0], np.cumsum(dt * d['F'][m][:-1])])
                             for m in ('MITgcm', 'OpInf')}
            if args.exact_resid:
                d['RS']['OpInf'] = d['RS_euler']['OpInf']
                rm = np.load(f'{bd}relax_monthly_fom_tau{tau}.npz')
                t_end = rm['interval'][:, 1]
                k = np.searchsorted(fom['time'], t_end)
                ok = k < len(fom['time'])
                assert np.allclose(fom['time'][k[ok]], t_end[ok]) and fom['time'][0] == rm['interval'][0, 0]
                Sz = fom['sc_zstar']
                d['RS_month'] = (np.concatenate([[0], k[ok]]) / 360.0,
                                 np.concatenate([[0.0], Sz[k[ok]] - Sz[0] - np.cumsum(rm['salt_int'][ok])]))
            data[tau] = d
        return dict(data=data, cfg=cfg, dt=dt, area=area, vol=vol)

    ns = build(args.run_dir)
    data, cfg, dt, area, vol = (ns[k] for k in ('data', 'cfg', 'dt', 'area', 'vol'))
    rom_methods = ('OpInf',)
    if args.run_dir2:      # second ROM: its OpInf series join data[tau] as OpInf2
        for tau, d2 in build(args.run_dir2)['data'].items():
            for key in ('S', 'F', 'V', 'RS', 'RV'):
                data[tau][key]['OpInf2'] = d2[key]['OpInf']
        rom_methods = ('OpInf', 'OpInf2')

    fw = lambda F: -F / (S0 * area) * MM_YR  # noqa: E731  psu m3/s -> mm/yr freshwater
    # psu m3/s -> basin-mean salinity tendency, 1e-3 psu/yr
    ps = 1.0 if args.psu_sci else 1e3        # plotted salinity unit: psu or 1e-3 psu
    yl = lambda lim: None if lim is None else [x * ps / 1e3 for x in lim]  # noqa: E731
    sflux_conv = (lambda F: F / vol * YEAR * ps) if args.consistent_units else fw

    # ---------------- figures
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style = {'MITgcm': dict(color=args.fom_color, lw=1.3, ls=':', zorder=3),
             'OpInf': dict(color=C_ROM, lw=1.1, zorder=2),
             'projection': dict(color='#888888', lw=0.9),
             'MITgcm z*': dict(color='#d95f02', lw=1.1, ls='-.', zorder=3)}
    style['OpInf2'] = dict(color=args.color2, lw=1.1, zorder=2)
    label = {'MITgcm': 'MITgcm', 'OpInf': args.label, 'OpInf2': args.label2,
             'projection': 'projection only', 'MITgcm z*': 'MITgcm, z* thickness'}
    plot_methods = (('MITgcm',) if args.no_projection else ('MITgcm', 'projection')) + rom_methods

    def running(ax, t, R, conv, ylim):
        k0 = MONTH
        for m in plot_methods:
            # MITgcm sits on the zero line here: draw it thicker so it stays visible
            st = dict(style[m], lw=args.resid_lw, ls='--') if m == 'MITgcm' else style[m]
            ax.plot(t[k0:], conv(R[m][k0:] / (dt * np.arange(k0, len(t)))), label=label[m], **st)
        if ylim:
            ax.set_ylim(*ylim)

    def make(rows, ylab, out):
        nr, ntau = len(rows), len(args.taus)
        fig, axs = plt.subplots(nr, ntau, figsize=(2.6 * ntau + 0.7, 2.1 * nr + 0.2),
                                sharex=True, sharey='row', constrained_layout=True, squeeze=False)
        for j, tau in enumerate(args.taus):
            d = data[tau]
            t = np.arange(d['nt']) / 360.0
            for i, rowname in enumerate(rows):
                ax = axs[i, j]
                if rowname == 'salt':
                    S_ref = d['S']['MITgcm'][0]
                    for m in ('MITgcm',) + rom_methods:
                        ax.plot(t, (d['S'][m] - S_ref) / vol * ps, label=label[m], **style[m])
                elif rowname == 'sflux':
                    for m in ('MITgcm',) + rom_methods:
                        ax.plot(t, sflux_conv(d['F'][m]), label=label[m], **style[m])
                    if args.sflux_ylim:
                        ax.set_ylim(*(yl(args.sflux_ylim) if args.consistent_units
                                      else args.sflux_ylim))
                    if args.consistent_units and j == ntau - 1:
                        # tendency [ps^-1 psu/yr] -> freshwater [mm/yr]; linear, sign-flipped
                        k = vol / (S0 * area) * 1e3 / ps
                        sec = ax.secondary_yaxis('right', functions=(lambda x: -k * x,
                                                                     lambda y: -y / k))
                        sec.set_ylabel('Freshwater equivalent\n(mm yr$^{-1}$)', fontsize=FS_LABEL)
                        sec.tick_params(labelsize=FS_TICK)
                elif rowname == 'sresid' and args.resid_units == 'cum':
                    for m in plot_methods + (('MITgcm z*',) if args.zstar else ()):
                        st = dict(style[m], lw=args.resid_lw, ls='--') if m == 'MITgcm' else style[m]
                        if m == 'MITgcm' and args.exact_resid:
                            tmo, Rmo = d['RS_month']
                            ax.plot(tmo, Rmo / vol * ps, label=label[m], **st)
                            continue
                        ax.plot(t, d['RS'][m] / vol * ps, label=label[m], **st)
                    if args.resid_ylim:
                        ax.set_ylim(*yl(args.resid_ylim))
                elif rowname == 'sresid':
                    running(ax, t, d['RS'], fw, args.resid_ylim)
                elif rowname == 'eta':
                    for m in ('MITgcm',) + rom_methods:
                        ax.plot(t, d['V'][m] / area * 1e3, label=label[m], **style[m])
                elif rowname == 'vresid':
                    running(ax, t, d['RV'], lambda x: x / area * MM_YR, args.vol_resid_ylim)
                if rowname not in ('salt', 'sflux'):
                    ax.axhline(0, color='#999999', lw=0.5, zorder=1)
            axs[0, j].set_title(rf'$\tau$={tau} ({"training" if d["kind"] == "train" else "held out"})',
                                fontsize=FS_LABEL)
            axs[-1, j].set_xlabel('year', fontsize=FS_LABEL)
            axs[-1, j].set_xlim(0, 10)
        for i, rowname in enumerate(rows):
            axs[i, 0].set_ylabel(ylab[rowname], fontsize=FS_LABEL)
            axs[i, 0].text(-0.3, 1.0, f'({"abcdefg"[i]})', transform=axs[i, 0].transAxes,
                           fontsize=FS_LABEL, fontweight='bold', va='top')
            axs[i, 0].legend(fontsize=FS_TICK, frameon=False, loc='best')
        for ax in axs.flat:
            ax.tick_params(labelsize=FS_TICK)
            ax.grid(True, color='#e6e6e6', lw=0.5)
        if args.psu_sci and rows[0] == 'salt':
            for i, e in enumerate(args.psu_sci[:nr]):
                for ax in axs[i]:
                    ax.ticklabel_format(axis='y', style='sci', scilimits=(e, e), useMathText=True)
                    ax.yaxis.get_offset_text().set_fontsize(FS_TICK)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        print(f'wrote {out}')

    pu = '' if args.psu_sci else '10$^{-3}$ '
    make(['salt', 'sflux', 'sresid'],
         {'salt': f'Basin-mean salinity\nanomaly ({pu}psu)',
          'sflux': (f'Surface salt flux\n({pu}psu yr$^{{-1}}$)'
                    if args.consistent_units else
                    'Implied surface freshwater\nflux (mm yr$^{-1}$)'),
          'sresid': (f'Salt budget residual\n({pu}psu)' if args.resid_units == 'cum'
                     else 'Salt budget residual\n(mm yr$^{-1}$ freshwater)')}, args.out_salt)
    make(['eta', 'vresid'],
         {'eta': r'Basin-mean $\eta$ (mm)',
          'vresid': 'Volume budget residual\n(mm yr$^{-1}$)'}, args.out_vol)

    # ---------------- numbers
    n_yr = lambda d: (d['nt'] - 1) * dt / YEAR  # noqa: E731
    print('\nsalt: mean freshwater flux FOM/ROM [mm/yr], basin-mean S drift at end FOM/ROM '
          '[1e-3 psu], S error at end [1e-3 psu]')
    for tau, d in data.items():
        Sf, Sr = d['S']['MITgcm'], d['S']['OpInf']
        print(f"  tau{tau}: {fw(d['F']['MITgcm']).mean():8.2f}/{fw(d['F']['OpInf']).mean():8.2f}"
              f"   {(Sf[-1] - Sf[0]) / vol * 1e3:+.3f}/{(Sr[-1] - Sr[0]) / vol * 1e3:+.3f}"
              f"   {(Sr[-1] - Sf[-1]) / vol * 1e3:+.3f}")
    print('\nsalt residual R_S at end, as mean freshwater-equivalent [mm/yr] '
          '(MITgcm z* = exact model budget)')
    for tau, d in data.items():
        conv = lambda R: fw(R[-1] / ((d['nt'] - 1) * dt))  # noqa: E731
        print(f"  tau{tau}: MITgcm z* {conv(d['RSz']):+.4f}, " + ', '.join(
            f"{m} {conv(d['RS'][m]):+.4f}" for m in METHODS))
    print('\nsalt residual: value at end [1e-3 psu], and drift = slope over yr 2-end '
          '[mm/yr freshwater] (excludes the early offset)')
    for tau, d in data.items():
        t = np.arange(d['nt']) * dt
        k = 720 // cfg['n_days']
        print(f'  tau{tau}: ' + ', '.join(
            f"{m} {d['RS'][m][-1] / vol * 1e3:+.3f} / "
            f"{fw(np.polyfit(t[k:], d['RS'][m][k:], 1)[0]):+.2f}" for m in METHODS))
    print('\nsame salt residual with the flux summed explicitly (left point) [1e-3 psu]: '
          'end value, max |R|')
    for tau, d in data.items():
        print(f'  tau{tau}: ' + ', '.join(
            f"{m} {d['RS_euler'][m][-1] / vol * 1e3:+.2e} / {np.abs(d['RS_euler'][m]).max() / vol * 1e3:.2e}"
            for m in ('MITgcm', 'OpInf')))
    if args.exact_resid:
        print('\nexact salt residual [1e-3 psu]: MITgcm (z* + monthly restoring) end / max |R|, '
              'OpInf (explicit) end / max |R|; and as mean mm/yr freshwater')
        for tau, d in data.items():
            tmo, Rmo = d['RS_month']
            R = d['RS']['OpInf']
            print(f'  tau{tau}: MITgcm {Rmo[-1] / vol * 1e3:+.2e} / {np.abs(Rmo).max() / vol * 1e3:.2e}'
                  f' ({fw(Rmo[-1] / (tmo[-1] * YEAR)):+.2e} mm/yr), OpInf {R[-1] / vol * 1e3:+.2e} / '
                  f'{np.abs(R).max() / vol * 1e3:.2e}')
    print('\nvolume: basin-mean eta at end [mm], mean spurious source over the run [mm/yr]')
    for tau, d in data.items():
        print(f'  tau{tau}: ' + ', '.join(
            f"{m} {d['V'][m][-1] / area * 1e3:+.3f} mm / "
            f"{d['RV'][m][-1] / area * 1e3 / n_yr(d):+.3f} mm/yr" for m in METHODS))


if __name__ == '__main__':
    main()
