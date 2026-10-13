#!/usr/bin/env python3
"""Paper figures for heat and salt conservation, FOM vs ROM.

Figure 1 (--out-resid): budget self-consistency, 2 rows x 3 taus.
    R(t) = C(t) - C(0) - int_0^t F dt for heat (ZJ) and salt (as basin-mean salinity,
    1e-3 psu), for MITgcm, projection only and the OpInf ROM. Zero = conservative.

Figure 2 (--out-ops): where the ROM gets its heat and salt, 2 rows (heat, salt) x 3 panels.
    (a) mean contribution of each operator slot along the held-out rollout, next to the
        physical target for that slot evaluated on the same trajectory:
            constant (c + C + B Tbar_max) ->  Dt (mean F* - F_1(center))
            B wind            ->  0
          (F* from MITgcm's theta*(t) for heat and the constant S* for salt; the zero-mean
          seasonal B Tmax anomaly, target Dt (F*(t) - mean F*), is shown in panel b)
            (A - I) q         ->  -Dt f^T q                  (surface-restoring damping)
            H(q x q)          ->  0
            total dC/dt       ->  implied surface flux of the ROM state
    (b) monthly climatology of the seasonal anomaly of the main terms (same rollout)
    (c) net constant input at the center state and mean inputs vs tau: physics is
        tau-independent; the ROM's imposed wind column grows linearly with tau and the
        tau-independent rest (learned c + imposed C + B Tmax) offsets it.

Heat and salt content and the state-dependent part of each restoring flux are linear
functionals of the state, exact in reduced coordinates (see rom_heat_budget.py,
salt_volume_budget.py):  C(x) = C(center) + h^T q,  h = Tr^T [C(x_n) - C(center)]_n.

    python3 src/plot_conservation_figs.py
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
from fom_heat_budget import rom_forcing_fields  # noqa: E402
from rom_heat_budget import RUN_DEFAULT, DAY, ckron, load_operators  # noqa: E402
from heat_budget_depth import FS_TICK, FS_LABEL, C_ROM  # noqa: E402

YEAR = 360 * DAY
METHODS = ('MITgcm', 'projection', 'OpInf')
# categorical slots 2-5 of the reference palette (slot 1 = the project's ROM blue)
C_WIND, C_CONST, C_TMAX, C_LIN = '#eb6834', '#1baf7a', '#eda100', '#e87ba4'


def cumint(F, dt):
    return np.concatenate([[0.0], np.cumsum(dt * 0.5 * (F[:-1] + F[1:]))])


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--taus', nargs='+', default=['0.2', '0.3', '0.4'])
    p.add_argument('--op-tau', default='0.3', help='rollout used for panels (a), (b)')
    p.add_argument('--sweep-taus', nargs='+', type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5],
                   help='taus for panel (c) (inputs only; no FOM data needed)')
    p.add_argument('--run-dir', default=RUN_DEFAULT)
    p.add_argument('--budget-dir', default=paths.BUDGET_DIR)
    p.add_argument('--grid-dir', default=paths.GRID_DIR)
    p.add_argument('--out-resid', default='figs/heat_budget/conservation_residual_heat_salt.pdf')
    p.add_argument('--out-ops', default='figs/heat_budget/conservation_operators_heat_salt.pdf')
    args = p.parse_args()

    run, bd = args.run_dir, args.budget_dir
    pre = str(Path(run).parent) + '/'
    cfg = json.load(open(run + 'config.json'))
    nt_tr = 360 * cfg['n_year_train'] // cfg['n_days']
    dt = cfg['n_days'] * DAY
    ny = nx = 248
    train = [float(t) for t in cfg['train_taus']]

    # ---------------- grid, weights, forcing patterns
    rac = np.squeeze(rdmds(args.grid_dir + 'RAC')).astype(float)
    drf = np.squeeze(rdmds(args.grid_dir + 'DRF')).astype(float)
    hfac = np.squeeze(rdmds(args.grid_dir + 'hFacC')).astype(float)
    nz = hfac.shape[0]
    h0 = np.load(f'{bd}heat_budget_fom_tau{cfg["train_taus"][0]}.npz')
    s0 = np.load(f'{bd}salt_budget_fom_tau{cfg["train_taus"][0]}.npz')
    rho_cp, tau_th = float(h0['rho0'] * h0['cp']), float(h0['tau_theta'])
    tau_s, area, vol = float(s0['tau_s']), float(h0['area']), float(h0['volume'])
    w3 = rac[None] * drf[:, None, None] * hfac
    a1 = rac * drf[0] * hfac[0]
    c_th, _, s_st = rom_forcing_fields(args.grid_dir, ny, nx)
    FS_star = float(s0['FS_star'])                    # psu m3/s

    # ---------------- centre and reduced functionals from the training snapshots
    tag = f"{cfg['center_opt']}_{cfg['n_days']}days_{cfg['n_year_train']}yrs"
    center = {}
    for v in ('T', 'S'):
        with xr.open_dataset(pre + f'center{v}_{tag}.nc', engine='h5netcdf') as ds:
            center[v] = np.squeeze(ds['center'].values).reshape(nz, ny, nx)
    H_c = rho_cp * np.sum(w3 * center['T'])
    Fth_c = rho_cp / tau_th * np.sum(a1 * center['T'][0])
    S_c = np.sum(w3 * center['S'])
    FSs_c = np.sum(a1 * center['S'][0]) / tau_s
    Gc_c, Gs_c = np.sum(c_th * center['T'][0]), np.sum(s_st * center['S'][0])
    htr = [np.load(f'{bd}heat_budget_fom_tau{t}.npz') for t in cfg['train_taus']]
    str_ = [np.load(f'{bd}salt_budget_fom_tau{t}.npz') for t in cfg['train_taus']]
    cat = lambda ds, k: np.concatenate([d[k][:nt_tr] for d in ds])  # noqa: E731
    Tr = np.load(run + 'Tr.npy')
    h = Tr.T @ (cat(htr, 'hc_k').sum(1) - H_c)
    f = Tr.T @ (cat(htr, 'F_theta') - Fth_c)
    s = Tr.T @ (cat(str_, 'sc_k').sum(1) - S_c)
    g = Tr.T @ (cat(str_, 'FS_S') - FSs_c)
    C_th = Tr.T @ (cat(htr, 'G_ctheta') - Gc_c)
    C_s = Tr.T @ (cat(htr, 'G_sstar') - Gs_c)

    ops = load_operators(run)
    B, C, c, A, Hop = ops['B'], ops['C'], ops['c'], ops['A'], ops['H']
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)  # noqa: E731
    print(f'check: C rebuilt (theta+salt) vs model.h5 rel err {rel(C_th + C_s, C):.1e}'
          ' (large if the imposed B/C were corrected, e.g. refit_conservation.py)')
    Qhat = np.load(run + 'Qhat.npy')
    Fo = OceanForcing(grid_dir=args.grid_dir, nx=nx, ny=ny, xo=0, yo=14, xc=31, yc=45,
                      dx=0.25, dy=0.25)
    Tmax, _, _ = Fo.compute_temp_forcing()
    inputs = lambda tau: build_rom_inputs(  # noqa: E731
        Tmax, Fo.compute_wind_seasonal_amplitude(max_val=float(tau)),
        cfg['n_year_train'], cfg['n_days'])

    # unit conversions: per-step J -> W/m^2; per-step psu m3 -> 1e-3 psu/yr basin mean
    heat_u = lambda x: x / (area * dt)             # noqa: E731
    salt_u = lambda x: x / vol * 1e3 / dt * YEAR   # noqa: E731
    # physical restoring target: MITgcm's own theta*(t) (identical in every run), split
    # into its mean and seasonal anomaly; salt target S* is constant in time
    Fh_star = np.load(f'{bd}heat_budget_fom_tau{args.op_tau}.npz')['F_star']
    n_whole = 360 * ((len(Fh_star) - 1) // 360)
    Fh_bar = Fh_star[:n_whole].mean()
    Q = {'heat': dict(w=h, fl=f, u=heat_u, const_phys=dt * (Fh_bar - Fth_c),
                      seas_phys=dt * (Fh_star - Fh_bar)),
         'salt': dict(w=s, fl=g, u=salt_u, const_phys=dt * (FS_star - FSs_c),
                      seas_phys=np.zeros_like(Fh_star))}

    # ---------------- residual time series (figure 1)
    res = {}
    for tau in args.taus:
        kind = 'train' if float(tau) in train else 'test'
        q = np.load(run + f'Q_ROM_{kind}_run_10yrspinup_tau{tau}.npy')
        nt = q.shape[1]
        if kind == 'train':
            i0 = train.index(float(tau)) * nt_tr
            qp = Qhat[:, i0:i0 + nt]
        else:
            qp = np.load(run + f'Q_PROJ_run_10yrspinup_tau{tau}.npy')
        fh = np.load(f'{bd}heat_budget_fom_tau{tau}.npz')
        fs = np.load(f'{bd}salt_budget_fom_tau{tau}.npz')
        Fh_star = fh['F_star'][:nt]
        cont = {'heat': {'MITgcm': fh['hc_k'][:nt].sum(1), 'projection': H_c + h @ qp,
                         'OpInf': H_c + h @ q},
                'salt': {'MITgcm': fs['sc_k'][:nt].sum(1), 'projection': S_c + s @ qp,
                         'OpInf': S_c + s @ q}}
        flux = {'heat': {'MITgcm': Fh_star - fh['F_theta'][:nt],
                         'projection': Fh_star - (Fth_c + f @ qp),
                         'OpInf': Fh_star - (Fth_c + f @ q)},
                'salt': {'MITgcm': FS_star - fs['FS_S'][:nt],
                         'projection': FS_star - (FSs_c + g @ qp),
                         'OpInf': FS_star - (FSs_c + g @ q)}}
        res[tau] = dict(kind=kind, nt=nt, R={
            b: {m: cont[b][m] - cont[b][m][0] - cumint(flux[b][m], dt) for m in METHODS}
            for b in ('heat', 'salt')})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style = {'MITgcm': dict(color='k', lw=2.0, ls='--', zorder=3),
             'OpInf': dict(color=C_ROM, lw=1.1, zorder=2),
             'projection': dict(color='#888888', lw=0.9)}
    label = {'MITgcm': 'MITgcm', 'OpInf': 'OpInf ROM', 'projection': 'projection only'}
    conv = {'heat': lambda R: R / 1e21, 'salt': lambda R: R / vol * 1e3}
    ylab = {'heat': 'Heat budget residual (ZJ)',
            'salt': 'Salt budget residual,\nbasin-mean salinity (10$^{-3}$ psu)'}

    ntau = len(args.taus)
    fig, axs = plt.subplots(2, ntau, figsize=(2.6 * ntau + 0.7, 4.6), sharex=True,
                            sharey='row', constrained_layout=True, squeeze=False)
    for j, tau in enumerate(args.taus):
        d = res[tau]
        t = np.arange(d['nt']) / 360.0
        for i, b in enumerate(('heat', 'salt')):
            ax = axs[i, j]
            ax.axhline(0, color='#999999', lw=0.5, zorder=1)
            for m in METHODS:
                ax.plot(t, conv[b](d['R'][b][m]), label=label[m], **style[m])
        axs[0, j].set_title(rf'$\tau$={tau} ({"training" if d["kind"] == "train" else "held out"})',
                            fontsize=FS_LABEL)
        axs[-1, j].set_xlabel('Time (years)', fontsize=FS_LABEL)
        axs[-1, j].set_xlim(0, 10)
    for i, b in enumerate(('heat', 'salt')):
        axs[i, 0].set_ylabel(ylab[b], fontsize=FS_LABEL)
        axs[i, 0].text(-0.32, 1.0, f'({"ab"[i]})', transform=axs[i, 0].transAxes,
                       fontsize=FS_LABEL, fontweight='bold', va='top')
    axs[0, 0].legend(fontsize=FS_TICK, frameon=False, loc='best')
    for ax in axs.flat:
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, color='#e6e6e6', lw=0.5)
    Path(args.out_resid).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_resid)
    print(f'wrote {args.out_resid}')

    print('\nresidual at yr 10: heat [ZJ], salt [1e-3 psu]')
    for tau, d in res.items():
        print(f'  tau{tau}: ' + ' | '.join(
            f"{b}: " + ', '.join(f"{m} {conv[b](d['R'][b][m][-1]):+.3f}" for m in METHODS)
            for b in ('heat', 'salt')))

    # ---------------- operator attribution along the op-tau rollout (figure 2 a, b)
    tau = args.op_tau
    kind = 'train' if float(tau) in train else 'test'
    q = np.load(run + f'Q_ROM_{kind}_run_10yrspinup_tau{tau}.npy')
    nt = q.shape[1]
    u = inputs(tau)[:, :nt - 1]
    q0 = q[:, :-1]
    Aq, HQ = A @ q0, Hop @ ckron(q0)
    print(f'check: rollout rebuilt from operators rel err '
          f'{rel(np.outer(B[:, 0], u[0]) + np.outer(B[:, 1], u[1]) + (C + c)[:, None] + Aq + HQ, q[:, 1:]):.1e}')
    # slot 0 holds every tau-independent constant (learned c, imposed C and the mean of the
    # imposed Tmax term); slot 1 is the zero-mean seasonal Tmax anomaly (panel b only)
    slots = ['constant\n' + r'($c + C + B\,\bar T_{max}$)', r'$B$ $T_{max}$ anomaly',
             r'$B$ wind', r'$(A-I)q$', r'$H(q\otimes q)$', r'total $\Delta$/$\Delta t$']
    bar_slots = [k for i, k in enumerate(slots) if i != 1]
    Tbar = inputs(tau)[0, :360].mean()
    att = {}
    for b, P in Q.items():
        w, fl = P['w'], P['fl']
        rom = {slots[0]: np.full(nt - 1, w @ (c + C) + (w @ B[:, 0]) * Tbar),
               slots[1]: (w @ B[:, 0]) * (u[0] - Tbar), slots[2]: (w @ B[:, 1]) * u[1],
               slots[3]: w @ (Aq - q0), slots[4]: w @ HQ}
        rom[slots[5]] = w @ np.diff(q, axis=1)
        phys = {slots[0]: np.full(nt - 1, P['const_phys']), slots[1]: P['seas_phys'][:nt - 1],
                slots[2]: np.zeros(nt - 1), slots[3]: -dt * (fl @ q0),
                slots[4]: np.zeros(nt - 1)}
        phys[slots[5]] = sum(phys.values())
        att[b] = dict(rom={k: P['u'](v) for k, v in rom.items()},
                      phys={k: P['u'](v) for k, v in phys.items()})
        clos = np.max(np.abs(sum(rom[k] for k in slots[:5]) - rom[slots[5]]))
        print(f'check {b}: operator terms sum to w^T dq, max|diff| {P["u"](clos):.1e}')

    # ---------------- net constant input vs tau (figure 2 c)
    sweep = {}
    for b, P in Q.items():
        w = P['w']
        rest, wind, phys = [], [], []
        for tv in args.sweep_taus:
            ub = inputs(tv)[:, :360].mean(1)
            rest.append(P['u'](w @ (c + C) + (w @ B[:, 0]) * ub[0]))
            wind.append(P['u']((w @ B[:, 1]) * ub[1]))
            phys.append(P['u'](P['const_phys']))
        sweep[b] = dict(rest=np.array(rest), wind=np.array(wind), phys=np.array(phys))

    ulab = {'heat': r'W m$^{-2}$', 'salt': r'10$^{-3}$ psu yr$^{-1}$'}
    fig, axs = plt.subplots(2, 3, figsize=(10.2, 6.3), constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    months = np.arange(1, 13)
    for i, b in enumerate(('heat', 'salt')):
        # (a) mean per slot
        ax = axs[i, 0]
        y = np.arange(len(bar_slots))[::-1]
        rm = np.array([att[b]['rom'][k].mean() for k in bar_slots])
        pm = np.array([att[b]['phys'][k].mean() for k in bar_slots])
        ax.barh(y + 0.19, rm, height=0.36, color=C_ROM, label='OpInf ROM')
        ax.barh(y - 0.19, pm, height=0.36, color='#bdbdbd', edgecolor='k', lw=0.6,
                label='physical target')
        ax.axvline(0, color='#555555', lw=0.6)
        # values printed beside the bars: most physical targets are ~0 and otherwise invisible
        fmt = '{:+.1f}' if b == 'heat' else '{:+.2f}'
        span = max(np.abs(rm).max(), np.abs(pm).max())
        for yy, vals in ((y + 0.19, rm), (y - 0.19, pm)):
            for yv, v in zip(yy, vals):
                ax.text(v + np.sign(v or 1) * 0.03 * span, yv, fmt.format(v), va='center',
                        ha='left' if v >= 0 else 'right', fontsize=FS_TICK - 1.5, color='#333333')
        ax.set_xlim(-1.3 * np.abs(np.minimum(rm, pm).min()) - 0.1 * span,
                    1.3 * np.maximum(rm, pm).max() + 0.1 * span)
        ax.set_yticks(y)
        ax.set_yticklabels(bar_slots, fontsize=FS_TICK)
        ax.set_xlabel(f'Mean contribution ({ulab[b]})', fontsize=FS_LABEL)
        ax.set_title(f'{b.capitalize()}: operator contributions, ' + rf'$\tau$={tau}',
                     fontsize=FS_LABEL)
        ax.legend(fontsize=FS_TICK, frameon=False, loc='center right')
        # (b) seasonal anomaly climatology (monthly)
        ax = axs[i, 1]
        clim = lambda x: np.array([x[:360 * ((nt - 1) // 360)].reshape(-1, 12, 30)[:, m].mean()  # noqa: E731
                                   for m in range(12)])
        anom = lambda x: clim(x) - clim(x).mean()  # noqa: E731
        ax.axhline(0, color='#999999', lw=0.5)
        ax.plot(months, anom(att[b]['phys'][slots[5]]), color='k', ls='--', lw=1.6,
                label='physical implied flux')
        ax.plot(months, anom(att[b]['rom'][slots[5]]), color=C_ROM, lw=1.6, label='ROM total')
        ax.plot(months, anom(att[b]['rom'][slots[2]]), color=C_WIND, lw=1.2, marker='o', ms=3,
                label=r'ROM $B$ wind (imposed)')
        ax.plot(months, anom(att[b]['rom'][slots[3]]), color=C_LIN, lw=1.2, marker='s', ms=3,
                label=r'ROM $(A-I)q$ (learned)')
        ax.plot(months, anom(att[b]['rom'][slots[1]]), color=C_TMAX, lw=1.2, marker='^', ms=3,
                label=r'ROM $B$ $T_{max}$ (imposed)')
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.set_xlabel('Month', fontsize=FS_LABEL)
        ax.set_ylabel(f'Seasonal anomaly ({ulab[b]})', fontsize=FS_LABEL)
        ax.set_title('Seasonal cycle of the terms', fontsize=FS_LABEL)
        # (c) net constant input vs tau
        ax = axs[i, 2]
        tv = np.array(args.sweep_taus)
        sw = sweep[b]
        is_tr = np.isin(np.round(tv, 3), np.round(train, 3))
        ax.axhline(0, color='#999999', lw=0.5)
        ax.plot(tv, sw['phys'], color='k', ls='--', lw=1.6, label='physics')
        ax.plot(tv, sw['rest'] + sw['wind'], color=C_ROM, lw=1.6, label='ROM total')
        ax.plot(tv, sw['wind'], color=C_WIND, lw=1.2, label=r'ROM $B$ wind (imposed)')
        ax.plot(tv, sw['rest'], color=C_CONST, lw=1.2,
                label=r'ROM $c + C + B\,\bar T_{max}$')
        for yv, col in ((sw['rest'] + sw['wind'], C_ROM), (sw['wind'], C_WIND)):
            ax.scatter(tv[is_tr], yv[is_tr], color=col, s=22, zorder=4)
            ax.scatter(tv[~is_tr], yv[~is_tr], facecolor='white', edgecolor=col, s=22, zorder=4)
        ax.set_xlabel(r'Wind amplitude $\tau$ (N m$^{-2}$)', fontsize=FS_LABEL)
        ax.set_ylabel(f'Net constant input ({ulab[b]})', fontsize=FS_LABEL)
        ax.set_title('At the center state, mean inputs\n(filled = training τ)', fontsize=FS_LABEL)
    # one legend per column, below the bottom row (the series are the same in both rows)
    for j in (1, 2):
        axs[1, j].legend(*axs[1, j].get_legend_handles_labels(), loc='upper center',
                         bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=FS_TICK - 0.5,
                         frameon=False)
    for i in range(2):
        for j in range(3):
            axs[i, j].text(-0.02 if j else -0.30, 1.06, f'({"abcdef"[3 * i + j]})',
                           transform=axs[i, j].transAxes, fontsize=FS_LABEL, fontweight='bold',
                           ha='right', va='bottom')
    for ax in axs.flat:
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, color='#e6e6e6', lw=0.5)
    fig.savefig(args.out_ops)
    print(f'wrote {args.out_ops}')

    for b in ('heat', 'salt'):
        print(f'\n{b} ({ulab[b]}), tau{tau} rollout: slot  ROM mean / physics mean,'
              ' seasonal amp ROM / physics')
        amp = lambda x: 0.5 * np.ptp(x[:360 * ((nt - 1) // 360)].reshape(-1, 12, 30).mean((0, 2)))  # noqa: E731
        for k in slots:
            r_, p_ = att[b]['rom'][k], att[b]['phys'][k]
            print(f"  {k.replace(chr(10), ' '):22s} {r_.mean():+9.3f} / {p_.mean():+9.3f}"
                  f"   {amp(r_):8.3f} / {amp(p_):8.3f}")
        sw = sweep[b]
        print('  net constant input vs tau: ' + ', '.join(
            f'{t_:.1f}: ROM {r_ + w_:+.2f} (wind {w_:+.2f}, rest {r_:+.2f}) phys {p_:+.2f}'
            for t_, r_, w_, p_ in zip(args.sweep_taus, sw['rest'], sw['wind'], sw['phys'])))


if __name__ == '__main__':
    main()
