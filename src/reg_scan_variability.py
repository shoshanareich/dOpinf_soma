"""Offline regularization scan with a variability-aware score.

Tests the fork: is the overdamping a *selection* problem (a vibrant, stable operator
exists in the current model family but pointwise-L2 selection skips it) or
*fundamental* (every vibrant operator is unstable)?

Refits the discrete quadratic OpInf model at a scan of reg_H (reg_A fixed) on the
SAVED reduced data, rolls each training scenario out, and scores:
  - traj_err   : the existing pointwise trajectory L2 (what the current sweep minimizes)
  - ret_hi     : per-mode std(Q_ROM)/std(Qhat) for the high (eddy) band
  - late/early : high-band rolling-1yr-std collapse over the rollout (1 = no collapse)
  - stable     : rollout over the training window is finite
  - long_stable: rollout over --stability-years stays finite AND bounded (growth < thr);
                 the stable reg region can be fragmented so the short-window stable
                 point may be only marginally stable.

Everything is reconstructed from saved artifacts (no MPI, no snapshot re-read):
  data-dir   : Tr.npy, Qhat.npy
  preproc-dir: {SST,SSS,SSU}_{center_opt}_{scale}_{n_year_train}yrs_transformed_surface.npy
  grid-dir   : SOMA grid for OceanForcing

--save-best-reg-H VALUE refits at that reg_H and writes a self-contained result dir
(Tr.npy, Qhat.npy, config.json, Q_ROM_train_*.npy) so projection_only_diag.py,
temporal_variability_diag.py, eval_rom_forcings.py, and the notebook all run on it.

Run (serial, opinf_mixed env):
  python3 src/reg_scan_variability.py \
      --config <data-dir>/config.json \
      --data-dir <data-dir> \
      --preproc-dir /scratch2/shoshi/soma4/dOpInf_results/save_roms/taus_1_5_sameIC
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_utils import resolve_forcing_config, ensure_slash
from forcing import OceanForcing, project_surface_forcings_multiple, build_rom_inputs
from learn_rom import _make_model, _make_solver, weighted_lp_error
from temporal_variability_diag import rolling_std

GROWTH_THRESHOLD = 5.0  # long-horizon |q| may not exceed this multiple of training max


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True, help="Dir with Tr.npy, Qhat.npy")
    p.add_argument("--preproc-dir", required=True, help="Dir with *_transformed_surface.npy")
    p.add_argument("--grid-dir", default="/scratch/shoshi/soma4/grid/")
    p.add_argument("--learn-N", action="store_true",
                   help="Add the bilinear state-input operator N(u⊗q)")
    p.add_argument("--reg-N", type=float, default=None,
                   help="Fixed reg for N (default: tie to reg_H). Use a SMALL value so N "
                        "isn't crushed by the large reg_H needed for H's stability.")
    p.add_argument("--reg-A", type=float, default=1e-6, help="Fixed reg_A (and reg_c) for the scan")
    p.add_argument("--reg-H", default=None,
                   help="Comma list of reg_H; default = coarse logspace(2,6) + fine grid near 1e4")
    p.add_argument("--stability-years", type=int, default=40,
                   help="Long-horizon rollout length for the stability check")
    p.add_argument("--save-best-reg-H", type=float, default=None,
                   help="Refit at this reg_H and write a self-contained result dir")
    p.add_argument("--predict-years", type=int, default=None,
                   help="Rollout length for saved Q_ROM (default = n_year_train)")
    p.add_argument("--save-dir", default=None, help="Where --save-best-reg-H writes (default outdir/refit_regH<val>)")
    p.add_argument("--outdir", default=None)
    return p.parse_args()


def build_forcing(config, preproc_dir, grid_dir, tr):
    """Reconstruct B, C and the forcing pieces exactly as the pipeline does."""
    center_opt, scale = config["center_opt"], config["scale"]
    n_year_train = config["n_year_train"]
    train_taus = resolve_forcing_config(config)["train_taus"]

    F = OceanForcing(grid_dir=grid_dir, nx=248, ny=248, xo=0, yo=14, xc=31, yc=45, dx=0.25, dy=0.25)
    sstar = F.compute_salinity_forcing()
    Tmax, c_theta, b_theta = F.compute_temp_forcing()
    g = F.compute_wind_forcing()
    tau_train = [F.compute_wind_seasonal_amplitude(max_val=float(t)) for t in train_taus]
    forcings = {"s_star": sstar.ravel(), "c_theta": c_theta, "b_theta": b_theta, "g": g.ravel()}

    stem = f"{center_opt}_{scale}_{n_year_train}yrs_transformed_surface.npy"
    SST = np.load(Path(preproc_dir) / f"SST_{stem}")
    SSS = np.load(Path(preproc_dir) / f"SSS_{stem}")
    SSU = np.load(Path(preproc_dir) / f"SSU_{stem}")
    B, C = project_surface_forcings_multiple(SST, SSS, SSU, tr, forcings)
    return B, C, Tmax, tau_train, train_taus


def make_inputs(Tmax, tau_train, n_years, n_days):
    return [build_rom_inputs(Tmax, t, n_years, n_days) for t in tau_train]


def predict_safe(model, q0, niters, inputs):
    with np.errstate(all="ignore"):
        try:
            q = model.predict(q0, niters=niters, inputs=inputs)
        except Exception:  # noqa: BLE001
            return None
    return q if np.all(np.isfinite(q)) else None


def score_rollout(q_true, q_rom, steps_per_year):
    r = q_true.shape[0]
    hi = slice(2 * r // 3, r)
    ret_hi = float(np.mean(q_rom.std(axis=1)[hi]) / max(np.mean(q_true.std(axis=1)[hi]), 1e-30))
    rs = rolling_std(q_rom, steps_per_year)[hi].mean(axis=0)
    c = np.arange(rs.shape[0])
    first2, last2 = c <= 2 * steps_per_year, c >= c[-1] - 2 * steps_per_year
    late_early = float(np.mean(rs[last2]) / max(np.mean(rs[first2]), 1e-30))
    traj = float(np.linalg.norm(weighted_lp_error(q_true, q_rom, normalize=True)))
    return {"ret_hi": ret_hi, "late_early": late_early, "traj": traj}


def main() -> None:
    args = parse_args()
    data_dir = Path(ensure_slash(args.data_dir))
    preproc_dir = ensure_slash(args.preproc_dir)
    outdir = Path(args.outdir or (data_dir / "analysis" / "reg_scan"))
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)
    n_year_train, n_days = config["n_year_train"], config["n_days"]
    steps_per_year = int(360 / n_days)
    nt = n_year_train * steps_per_year

    tr = np.load(data_dir / "Tr.npy")
    qhat = np.load(data_dir / "Qhat.npy")
    r = qhat.shape[0]
    n_scen = qhat.shape[1] // nt
    states_list = [qhat[:, i * nt:(i + 1) * nt] for i in range(n_scen)]
    train_max = max(np.max(np.abs(s)) for s in states_list)

    B, C, Tmax, tau_train, train_taus = build_forcing(config, preproc_dir, args.grid_dir, tr)
    labels = [f"tau{t:g}" if t is not None else f"scen{i}" for i, t in enumerate(train_taus)]
    inputs_train = make_inputs(Tmax, tau_train, n_year_train, n_days)
    inputs_long = make_inputs(Tmax, tau_train, args.stability_years, n_days)
    nlong = args.stability_years * steps_per_year

    if args.reg_H:
        reg_H_list = sorted(float(x) for x in args.reg_H.split(","))
    else:
        coarse = np.logspace(2, 6, 13)
        fine = np.logspace(np.log10(7e3), np.log10(2.2e4), 11)
        reg_H_list = sorted(np.unique(np.round(np.concatenate([coarse, fine]))))
    reg_A = args.reg_A
    learn_N = args.learn_N
    reg_N_val = args.reg_N
    m_in = int(np.asarray(inputs_train[0]).shape[0])  # input dim (m_in, not m: the
    # scoring loop below uses `m` as a per-scenario dict, which would clobber it)

    print(f"r={r}, nt={nt}, scenarios={n_scen}, reg_A={reg_A:.1e}, "
          f"stability check = {args.stability_years} yr\n")
    header = (f"{'reg_H':>11}{'stable':>8}{'long_st':>9}{'growth':>9}{'traj_max':>10}"
              + "".join(f"{'ret_hi['+l+']':>14}{'late/erly['+l+']':>17}" for l in labels))
    print(header)
    print("-" * len(header))

    rows = []
    for reg_H in reg_H_list:
        try:
            model = _make_model(B, C, solver=_make_solver(reg_A, reg_H, r, learn_N=learn_N, reg_N=reg_N_val, m=m_in), learn_N=learn_N).fit(
                states=states_list, inputs=inputs_train)
        except Exception as exc:  # noqa: BLE001
            print(f"{reg_H:>11.3g}{'FIT-ERR':>8}  ({exc!r})")
            continue

        per_scen, stable = [], True
        for i in range(n_scen):
            q = predict_safe(model, states_list[i][:, 0], nt, inputs_train[i])
            if q is None:
                stable = False
                break
            per_scen.append(score_rollout(states_list[i], q[:, :nt], steps_per_year))

        row = {"reg_H": reg_H, "stable": stable, "long_stable": False, "growth": np.nan, "traj_max": np.nan}
        if stable:
            # long-horizon stability on scenario 0
            qlong = predict_safe(model, states_list[0][:, 0], nlong, inputs_long[0])
            if qlong is not None:
                growth = float(np.max(np.abs(qlong)) / max(train_max, 1e-30))
                row["growth"] = growth
                row["long_stable"] = growth < GROWTH_THRESHOLD
            row["traj_max"] = max(m["traj"] for m in per_scen)
            for l, m in zip(labels, per_scen):
                row[f"ret_hi[{l}]"], row[f"late_early[{l}]"] = m["ret_hi"], m["late_early"]
            line = (f"{reg_H:>11.3g}{'yes':>8}{('yes' if row['long_stable'] else 'NO'):>9}"
                    f"{row['growth']:>9.2f}{row['traj_max']:>10.3f}")
            for l in labels:
                line += f"{row[f'ret_hi[{l}]']:>14.3f}{row[f'late_early[{l}]']:>17.3f}"
            print(line)
        else:
            print(f"{reg_H:>11.3g}{'NO':>8}{'-':>9}{'-':>9}{'-':>10}  (unstable / blow-up)")
        rows.append(row)

    # CSV
    cols = (["reg_H", "stable", "long_stable", "growth", "traj_max"]
            + [f"{p}[{l}]" for l in labels for p in ("ret_hi", "late_early")])
    with open(outdir / "reg_scan_variability.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(
                (f"{row[c]:.6g}" if isinstance(row.get(c), float) else str(row.get(c, "")))
                for c in cols) + "\n")
    print(f"\nsaved {outdir / 'reg_scan_variability.csv'}")

    # frontier plot (long-stable points only)
    stab = [x for x in rows if x["long_stable"]]
    if stab:
        reg = [x["reg_H"] for x in stab]
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        for l, c in zip(labels, ["tab:blue", "tab:orange", "tab:green"]):
            ax1.semilogx(reg, [x[f"ret_hi[{l}]"] for x in stab], "o-", color=c, label=f"ret_hi {l}")
        ax2.semilogx(reg, [x["traj_max"] for x in stab], "s--", color="tab:red", label="traj err")
        ax1.axhline(1.0, color="k", ls=":", lw=0.7)
        ax1.set_xlabel("reg_H"); ax1.set_ylabel("high-band variability retained")
        ax2.set_ylabel("trajectory L2 error", color="tab:red")
        ax1.set_title("Reg tradeoff (long-stable points): variability vs trajectory L2")
        ax1.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "reg_scan_frontier.png", dpi=150)
        plt.close(fig)
        print(f"saved {outdir / 'reg_scan_frontier.png'}")

    # --- optional: save a self-contained result dir at a chosen reg_H ---
    if args.save_best_reg_H is not None:
        reg_H = args.save_best_reg_H
        pred_years = args.predict_years or n_year_train
        save_dir = Path(args.save_dir or (outdir / f"refit_regH{reg_H:g}"))
        save_dir.mkdir(parents=True, exist_ok=True)
        inputs_pred = make_inputs(Tmax, tau_train, pred_years, n_days)
        npred = pred_years * steps_per_year

        model = _make_model(B, C, solver=_make_solver(reg_A, reg_H, r, learn_N=learn_N, reg_N=reg_N_val, m=m_in), learn_N=learn_N).fit(
            states=states_list, inputs=inputs_train)
        for fn in ("Tr.npy", "Qhat.npy"):
            shutil.copy(data_dir / fn, save_dir / fn)
        shutil.copy(args.config, save_dir / "config.json")

        fig, axes = plt.subplots(n_scen, 1, figsize=(11, 3.2 * n_scen), squeeze=False)
        modes = sorted(set([0, 1, 5, min(20, r - 1), r - 1]))
        for i, l in enumerate(labels):
            scen_name = Path(resolve_forcing_config(config)["snapshot_dirs"][i].rstrip("/")).name
            q = predict_safe(model, states_list[i][:, 0], npred, inputs_pred[i])
            if q is None:
                print(f"[warn] save: rollout for {l} blew up at reg_H={reg_H:g}")
                continue
            np.save(save_dir / f"Q_ROM_train_{scen_name}.npy", q)
            yrs_t = np.arange(nt) * n_days / 360.0
            yrs_r = np.arange(q.shape[1]) * n_days / 360.0
            ax = axes[i, 0]
            for m in modes:
                ax.plot(yrs_t, states_list[i][m], lw=0.8, alpha=0.5)
                ax.plot(yrs_r, q[m], lw=0.9)
            ax.set_title(f"{l}: reduced modes {modes} (faint=truth, solid=ROM @ reg_H={reg_H:g})")
            ax.set_xlabel("Year"); ax.set_ylabel("coeff")
        fig.tight_layout()
        fig.savefig(save_dir / "refit_mode_timeseries.png", dpi=150)
        plt.close(fig)

        print(f"\nsaved refit at reg_H={reg_H:g} -> {save_dir}")
        print("Confirm visually with the existing tools on that dir, e.g.:")
        print(f"  python3 src/projection_only_diag.py --config {save_dir}/config.json \\")
        print(f"      --data-dir {save_dir} --preproc-dir {preproc_dir}")
        print(f"  python3 src/temporal_variability_diag.py --config {save_dir}/config.json --data-dir {save_dir}")

    print("\nRead: pick the lowest reg_H with long_stable=yes AND growth~O(1); that's the")
    print("      most vibrant robustly-stable operator. If ret_hi there is still low for")
    print("      the strong regime -> fundamental (need N / stability-preserving H).")


if __name__ == "__main__":
    main()
