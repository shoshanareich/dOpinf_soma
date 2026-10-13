#!/usr/bin/env python3
"""Long-term stability figure: reduced-state norm ||q(t)||_2 over a long autonomous
rollout for every forcing amplitude, with the training/prediction boundary marked.

By default this reads the pre-computed long rollouts written by rollout_longterm.py
(<data-dir>/Q_ROM_<n_years>yr_tau<tau>.npy). Those are exact prefixes of the saved
training-horizon rollouts, so nothing is recomputed. Pass --recompute to re-roll
from model.h5 instead (identical result, just slower).

Sized for JGR: ML/AI (7.5 in max width, 8-9 pt type, embedded TrueType fonts);
writes PDF.

Usage:
    python3 src/plot_longterm_stability.py --n-years 60 --out figs/long_term_stability.pdf
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))  # core ROM modules

import numpy as np
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))
import paths  # noqa: E402

FIG_W = 7.5
FS_TICK = 7.5
FS_LABEL = 8.5
FS_ANNOT = 8.0

# per-tau palette shared with plot_moc_figs.py
TAU_COLORS = {0.1: "#009E73", 0.2: "#D55E00", 0.3: "#0072B2",
              0.4: "#E69F00", 0.5: "#CC79A7"}


def norms_from_dir(d, taus, n_years, n_days, recompute):
    """-> {tau: (years, norm)}. Overflow/NaN in the diverging tails is expected."""
    out = {}
    if recompute:
        import opinf
        from rollout_longterm import build_inputs

        model = opinf.models.DiscreteModel.load(str(d / "model.h5"))
        ic = {}
        for f in sorted(d.glob("Q_ROM_train_*.npy")) + sorted(d.glob("Q_ROM_test_*.npy")):
            ic[float(f.stem.split("tau")[-1])] = f
        niters = n_years * int(360 / n_days)
    for tau in taus:
        with np.errstate(over="ignore", invalid="ignore"):
            if recompute:
                q0 = np.load(ic[tau])[:, 0]
                Q = model.predict(q0, niters=niters, inputs=build_inputs(tau, n_years, n_days))
            else:
                Q = np.load(d / f"Q_ROM_{n_years}yr_tau{tau:g}.npy")
            nrm = np.linalg.norm(Q, axis=0)
        out[tau] = (np.arange(Q.shape[1]) * n_days / 360.0, nrm)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=paths.BASE_RUN)
    p.add_argument("--taus", default="0.2,0.3,0.4")
    p.add_argument("--train-taus", default="0.2,0.4")
    p.add_argument("--n-years", type=int, default=60)
    p.add_argument("--ymax", type=float, default=200.0)
    p.add_argument("--recompute", action="store_true",
                   help="re-roll from model.h5 instead of reading Q_ROM_<n>yr_tau*.npy")
    p.add_argument("--dash-test", action="store_true",
                   help="dash the held-out amplitudes as well as colour-coding them")
    p.add_argument("--out", default="figs/long_term_stability.pdf")
    p.add_argument("--preview-png", default=None)
    opts = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = Path(opts.data_dir)
    with open(d / "config.json") as f:
        cfg = json.load(f)
    n_days, n_year_train = cfg["n_days"], cfg["n_year_train"]

    taus = [float(v) for v in opts.taus.split(",")]
    train = [float(v) for v in opts.train_taus.split(",")]
    series = norms_from_dir(d, taus, opts.n_years, n_days, opts.recompute)

    for tau in taus:
        years, nrm = series[tau]
        fin = np.isfinite(nrm)
        over = fin & (nrm > opts.ymax)
        t_over = years[over][0] if over.any() else None
        t_nan = years[~fin][0] if (~fin).any() else None
        print(f"tau {tau:g} ({'train' if tau in train else 'test'}): "
              f"max norm over 0-{n_year_train} yr = {np.nanmax(nrm[:n_year_train * 360 // n_days]):.1f}; "
              f"leaves y <= {opts.ymax:g} at yr "
              f"{'never' if t_over is None else f'{t_over:.2f}'}; "
              f"non-finite at yr {'never' if t_nan is None else f'{t_nan:.2f}'}", flush=True)

    plt.rcParams.update({
        "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(FIG_W, 3.6), layout="constrained")

    for tau in taus:
        years, nrm = series[tau]
        is_train = tau in train
        kind = "training" if is_train else "held out"
        ax.plot(years, nrm, color=TAU_COLORS.get(tau, "0.4"),
                ls="-" if (is_train or not opts.dash_test) else "--", lw=1.4,
                label=r"$\tau_{\max} = $" + f"{tau:g} ({kind})", zorder=3)

    ax.axvline(n_year_train, color="0.5", ls="--", lw=0.9, zorder=1)
    for x, lab in ((n_year_train / 2, "Training"),
                   ((n_year_train + opts.n_years) / 2, "Prediction")):
        ax.text(x, 1.015, lab, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", color="gray", fontsize=FS_ANNOT)

    ax.set_xlabel("Year", labelpad=3)
    ax.set_ylabel(r"Reduced-State Norm, $\|\hat{q}(t)\|_2$", labelpad=4)
    ax.set_xlim(0, opts.n_years)
    ax.set_ylim(0, opts.ymax)
    ax.set_xticks(np.arange(0, opts.n_years + 1, 10))
    ax.set_xticks(np.arange(0, opts.n_years + 1, 5), minor=True)
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.5)
    ax.grid(True, which="minor", ls=":", lw=0.4, alpha=0.25)
    ax.legend(loc="upper right", ncol=1, framealpha=0.95, edgecolor="none",
              fontsize=6.8, handlelength=1.8, handletextpad=0.5, borderpad=0.35,
              labelspacing=0.3, columnspacing=1.2)

    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, facecolor="white")
    print(f"wrote {out}", flush=True)
    if opts.preview_png:
        fig.savefig(opts.preview_png, dpi=200, facecolor="white")
        print(f"wrote preview {opts.preview_png}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
