#!/usr/bin/env python3
"""POD spectrum figure: (a) normalized singular-value decay, (b) cumulative
retained energy, with the retained dimension r marked on both panels.

Reads ``eigs.npy`` from a result directory. That file holds the eigenvalues of
the method-of-snapshots Gram matrix (length nt), so singular values are
sqrt(eigs) and retained energy is cumsum(eigs) / sum(eigs).

Sized for JGR: ML/AI (7.5 in max width, 8-9 pt type, embedded TrueType fonts);
writes PDF.

Usage:
    python3 src/plot_pod_spectrum.py --r 105 --out figs/svd_decay.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))
import paths  # noqa: E402

FIG_W = 7.5
FS_TICK = 7.5
FS_LABEL = 8.5
FS_ANNOT = 8.0
FS_PANEL = 10.0
CHARCOAL = "#1A1A1A"
COLOR = "#D55E00"
R_LABEL_Y = 0.03   # axes fraction: keeps the "r = ..." tag just above the x axis in both panels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=paths.BASE_RUN)
    p.add_argument("--r", type=int, default=105, help="retained reduced dimension")
    p.add_argument("--energy-xmax", type=float, default=200.0,
                   help="x limit of the retained-energy panel")
    p.add_argument("--energy-ymin", type=float, default=0.1)
    p.add_argument("--out", default="figs/svd_decay.pdf")
    p.add_argument("--preview-png", default=None)
    opts = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib.transforms import blended_transform_factory as blend

    eigs = np.load(Path(opts.data_dir) / "eigs.npy")
    # the tail can carry O(1e-7) negative round-off from the Gram eigendecomposition
    n_neg = int((eigs < 0).sum())
    eigs_pos = np.clip(eigs, 0.0, None)
    ret = np.cumsum(eigs_pos) / eigs_pos.sum()
    svals = np.sqrt(eigs_pos)
    svals = svals / svals[0]
    idx = np.arange(1, eigs.size + 1)
    r = opts.r
    print(f"nt = {eigs.size} eigenvalues ({n_neg} clipped negative round-off); "
          f"retained energy at r = {r}: {ret[r - 1]:.4%}; "
          f"normalized sigma_r = {svals[r - 1]:.3e}", flush=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.0), layout="constrained")

    # ---- (a) singular-value decay ---------------------------------------
    ax = axes[0]
    good = svals > 0
    ax.semilogy(idx[good], svals[good], ls="-", lw=0.75, color=COLOR)
    ax.axvline(r, ls="--", lw=0.6, color=CHARCOAL)
    # pad the left edge so the steep head of the curve clears the y axis
    ax.set_xlim(-0.025 * eigs.size, eigs.size)
    ax.set_xlabel("Index", labelpad=2)
    ax.set_ylabel("Normalized Singular Values", labelpad=3)
    ax.text(r + 0.02 * eigs.size, R_LABEL_Y, f"$r = {r}$", color=CHARCOAL,
            transform=blend(ax.transData, ax.transAxes),
            fontsize=FS_ANNOT, va="bottom", ha="left")

    # ---- (b) cumulative retained energy ---------------------------------
    ax = axes[1]
    ax.plot(idx, ret, ls="-", lw=0.75, color=COLOR)
    ax.plot([r, r], [0, ret[r - 1]], ls="--", lw=0.6, color=CHARCOAL)
    ax.plot([0, r], [ret[r - 1], ret[r - 1]], ls="--", lw=0.6, color=CHARCOAL)
    ax.set_xlim(0, opts.energy_xmax)
    ax.set_ylim(opts.energy_ymin, 1.05)
    ax.set_xlabel("Reduced Dimension", labelpad=2)
    ax.set_ylabel("Retained Energy", labelpad=3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.0%}"))
    ax.text(r + 0.012 * opts.energy_xmax, R_LABEL_Y, f"$r = {r}$", color=CHARCOAL,
            transform=blend(ax.transData, ax.transAxes),
            fontsize=FS_ANNOT, va="bottom", ha="left")
    ax.text(0.45 * r, ret[r - 1] + 0.022, f"{ret[r - 1]:.2%}", color=CHARCOAL,
            fontsize=FS_ANNOT, ha="center", va="bottom")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

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
