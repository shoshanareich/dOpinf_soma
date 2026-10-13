#!/usr/bin/env python3
"""Computational cost figure: (top) core-hours for a 10-year MITgcm run, one-time
OpInf training, and OpInf inference; (bottom) breakdown of the OpInf training
wall-clock time by pipeline stage.

With --training-only, just the training breakdown is drawn, as a standalone
figure: legend above the bar, stage durations printed inside the segments, and
the x axis ticked at the cumulative stage boundaries (the running total) rather
than at round numbers.

Sized for JGR: ML/AI (8-9 pt type, embedded TrueType fonts); writes PDF.

Usage:
    python3 src/plot_compute_cost.py --out figs/compute_cost.pdf [--log]
    python3 src/plot_compute_cost.py --training-only --out figs/compute_cost_v2.pdf
"""
import argparse
from pathlib import Path

FS_TICK = 7.5
FS_LABEL = 8.5
FS_ANNOT = 8.0

# Okabe-Ito (colorblind-safe)
OI = {"orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}

LABELS_COST = ["MITgcm\n(per 10-yr run)", "OpInf training\n(one-time)",
               "OpInf inference\n(per 10-yr run)"]
COREHR = [400.0, 112.0, 1.5 / 3600]
COLORS_COST = ["#444444", OI["blue"], OI["green"]]

STEPS = ["Data transformations (center and scale)",
         "Dimension reduction (POD and snapshot projection)",
         "Surface forcing projection",
         "Grid search for regularization hyperparameters"]
MINS = [14.4, 18.0, 11.7, 63.4]
# Colours stay in this sequence rather than travelling with the stages: the
# alternating cool/warm order keeps adjacent segments far apart (worst adjacent
# CVD dE 21.9, vs 13.1 if the stages kept their previous colours after the swap).
COLORS_BREAKDOWN = [OI["skyblue"], OI["orange"], OI["blue"], OI["vermillion"]]
NCORES = 62


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", action="store_true",
                   help="log-scale the cost axis so the inference bar is visible")
    p.add_argument("--training-only", action="store_true",
                   help="draw only the training-time breakdown (writes compute_cost_v2)")
    p.add_argument("--out", default="figs/compute_cost.pdf")
    p.add_argument("--preview-png", default=None)
    opts = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    total_min = sum(MINS)
    print(f"training breakdown sums to {total_min:.1f} min x {NCORES} cores = "
          f"{total_min / 60 * NCORES:.1f} core-hours (bar shows {COREHR[1]:.0f})", flush=True)
    print(f"MITgcm / OpInf training = {COREHR[0] / COREHR[1]:.2f}x; "
          f"MITgcm / OpInf inference = {COREHR[0] / COREHR[2]:.3g}x", flush=True)

    plt.rcParams.update({
        "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    if opts.training_only:
        cum = [sum(MINS[:i + 1]) for i in range(len(MINS))]
        print("cumulative stage boundaries [min]: "
              + ", ".join(f"{c:.1f}" for c in cum), flush=True)

        # Fixed geometry, all in inches. constrained_layout is deliberately NOT
        # used: it does not reserve space for a legend anchored outside the axes,
        # and re-runs on every draw, so the legend drifts onto the bar. Explicit
        # add_axes rects cannot drift.
        BAR_IN, LEG_IN, BOT_IN, GAP_IN, TOP_IN = 0.34, 0.30, 0.40, 0.10, 0.04
        H = BOT_IN + BAR_IN + GAP_IN + LEG_IN + TOP_IN
        L, W = 0.02, 0.96
        fig = plt.figure(figsize=(6.8, H))
        axb = fig.add_axes([L, BOT_IN / H, W, BAR_IN / H])
        ax_leg = fig.add_axes([L, (BOT_IN + BAR_IN + GAP_IN) / H, W, LEG_IN / H])
        ax_leg.axis("off")

        left = 0.0
        for st, m, c in zip(STEPS, MINS, COLORS_BREAKDOWN):
            axb.barh(0, m, left=left, height=0.8, color=c, edgecolor="white",
                     linewidth=0.6, label=st)
            if m > 3:
                axb.text(left + m / 2, 0, f"{m:.1f}", ha="center", va="center",
                         color="white", fontsize=FS_ANNOT)
            left += m
        axb.set_xlim(0, left)
        # the bar spans 0.8 in data units, so this makes it fill the axes exactly
        # and BAR_IN is then literally the drawn thickness
        axb.set_ylim(-0.4, 0.4)
        axb.set_yticks([])
        # ticks at the running total; round numbers would collide with these
        # (20 vs 32, 40 vs 44, 100 vs 108) so they replace rather than join them
        axb.set_xticks([0] + cum)
        axb.set_xticklabels([f"{v:.0f}" for v in [0] + cum])
        axb.set_xlabel(f"Wall-clock time [min]  ({NCORES} cores)", labelpad=3)
        axb.spines[["top", "right", "left"]].set_visible(False)

        # legend lives in its own strip and expands to fill it, so it is exactly
        # as wide as the bar
        ax_leg.legend(*axb.get_legend_handles_labels(), loc="upper left",
                      bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), mode="expand", ncol=2,
                      frameon=False, fontsize=FS_ANNOT, handlelength=1.4,
                      handletextpad=0.5, labelspacing=0.4, borderaxespad=0.0)

        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        bb = lambda a: inv.transform_bbox(a.get_window_extent(rend))
        lg_bb = bb(ax_leg.get_legend())
        bar_top = axb.get_position().y1
        xlab_bb = bb(axb.xaxis.label)
        gaps = {"legend->bar": (lg_bb.y0 - bar_top) * H,
                "bar->xlabel": (axb.get_position().y0 - xlab_bb.y1) * H}
        print(f"bar thickness {BAR_IN:.2f} in; legend "
              f"{lg_bb.width * fig.get_figwidth():.3f} in vs bar "
              f"{W * fig.get_figwidth():.3f} in", flush=True)
        for k, v in gaps.items():
            print(f"  gap {k}: {v:+.3f} in "
                  f"{'OK' if v > 0.01 else 'OVERLAP'}", flush=True)
        if any(v <= 0.01 for v in gaps.values()):
            raise SystemExit("layout collision -- adjust BAR_IN/LEG_IN/GAP_IN/BOT_IN")

        out = Path(opts.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=600, facecolor="white")
        print(f"wrote {out}", flush=True)
        if opts.preview_png:
            fig.savefig(opts.preview_png, dpi=200, facecolor="white")
        plt.close(fig)
        return

    fig = plt.figure(figsize=(6.8, 4.4))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 0.22], hspace=0.55)

    # ---- top: total cost per stage --------------------------------------
    ax = fig.add_subplot(gs[0])
    bars = ax.bar(LABELS_COST, COREHR, color=COLORS_COST, width=0.45)
    ax.set_ylabel("Compute cost [core-hours]", labelpad=3)
    if opts.log:
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 3e3)
    else:
        ax.set_ylim(0, 440)
    for b, v in zip(bars, COREHR):
        txt = f"{v:.0f} core-hr" if v >= 1 else "1.5 s"
        y = v * 1.6 if opts.log else v + 6
        ax.text(b.get_x() + b.get_width() / 2, y, txt, ha="center", va="bottom",
                fontsize=FS_ANNOT)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- bottom: training wall-clock breakdown --------------------------
    axb = fig.add_subplot(gs[1])
    left = 0.0
    for s, m, c in zip(STEPS, MINS, COLORS_BREAKDOWN):
        axb.barh(0, m, left=left, color=c, edgecolor="white", linewidth=0.6,
                 label=f"{s}  ({m:.1f} min)")
        if m > 3:
            axb.text(left + m / 2, 0, f"{m:.0f}", ha="center", va="center",
                     color="white", fontsize=FS_ANNOT)
        left += m
    axb.set_xlim(0, left)
    axb.set_yticks([])
    axb.set_xlabel(f"Wall-clock time [min]  ({NCORES} cores)", labelpad=3)
    axb.spines[["top", "right", "left"]].set_visible(False)
    axb.legend(loc="upper center", bbox_to_anchor=(0.5, -1.05), ncol=2, frameon=False,
               fontsize=FS_ANNOT, handlelength=1.4, handletextpad=0.5,
               columnspacing=1.2, labelspacing=0.4)

    fig.subplots_adjust(left=0.10, right=0.97, top=0.96, bottom=0.30)

    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}", flush=True)
    if opts.preview_png:
        fig.savefig(opts.preview_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
