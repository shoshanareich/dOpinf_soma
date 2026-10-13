#!/usr/bin/env python3
"""Gyre strength versus wind-stress amplitude: time-mean barotropic streamfunction
extrema (subtropical max, subpolar |min|) for MITgcm and the OpInf ROM, with
training and test amplitudes marked.

FOM  : <fom-prefix><tau>/psi_20yrs.nc   (psi_bt, converted to Sv)
ROM  : <data-dir>/psi_rom_Q_ROM_{train,test}_run_10yrspinup_tau<tau>_<suffix>.nc

A ROM rollout that has gone unstable inside the averaging window is detected
(non-finite samples, or |psi| beyond --diverge-sv) and reported as a divergence
instead of being averaged: the FOM point is still plotted, the ROM curve is
broken at that amplitude, and the amplitude column is shaded and annotated with
the year the rollout left the plausible range. This is deliberate -- at tau0.1
the saved rollout is already non-finite over most of years 3-10, so a silent
mean would be NaN (or, worse, finite-but-meaningless over a shorter window).

For a diverged amplitude a PRE-DIVERGENCE pair is also plotted, as hollow
markers: the ROM mean over the part of the window it survived, together with the
MITgcm mean over *exactly the same samples*. The matched window is the point --
comparing a short ROM mean against the full-window FOM mean would be unfair at
tau0.1, whose FOM is itself still spinning down. A pre-divergence value above the
axis limit is drawn as a clipped arrow carrying its numeric label, so the main
curves keep a readable scale.

Sized for JGR: ML/AI (8-9 pt type, embedded TrueType fonts); writes PDF.

Usage:
    python3 src/plot_gyre_extremes.py --taus 0.2,0.3,0.4 --train-taus 0.2,0.4 \
        --year0 3 --nyears 7 --out figs/gyre_extremes_234.pdf

    # all five amplitudes, incl. the tau0.1 failure case and its pre-divergence pair
    python3 src/plot_gyre_extremes.py --taus 0.1,0.2,0.3,0.4,0.5 --train-taus 0.2,0.4 \
        --year0 3 --nyears 7 --ylim 25,200 --out figs/gyre_extremes_12345.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))
import paths  # noqa: E402

FS_TICK = 7.5
FS_LABEL = 8.5
C_SUBTROP = "#D55E00"
C_SUBPOL = "#0072B2"

# Grid spacing. The axes aspect is locked to GRID_DX/GRID_DY so that one
# gridline interval in x has the same display length as one in y, i.e. the grid
# cells are squares; the axes box is then letterboxed inside the figure.
GRID_DX = 0.05      # tau
GRID_DY = 25.0      # Sv
C_DIVERGE = "#999999"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=paths.BASE_RUN)
    p.add_argument("--fom-prefix", default=paths.FOM_PREFIX)
    p.add_argument("--rom-suffix", default="mean_var_r105_10trainingyrs")
    p.add_argument("--taus", default="0.2,0.3,0.4")
    p.add_argument("--train-taus", default="0.2,0.4")
    p.add_argument("--rom-only-taus", default="",
                   help="amplitudes with a ROM rollout but NO MITgcm run (e.g. "
                        "0.25,0.35,0.45). They join the ROM curve with the ordinary "
                        "test-amplitude star; no FOM file is opened and no FOM point "
                        "is drawn, so the missing MITgcm marker is what sets them apart.")
    p.add_argument("--year0", type=int, default=3, help="first year of the averaging window")
    p.add_argument("--nyears", type=int, default=7, help="length of the averaging window")
    p.add_argument("--spy", type=int, default=360, help="samples per year")
    p.add_argument("--xg-min", type=float, default=0.0, help="western longitude cutoff")
    p.add_argument("--ylim", default="50,150")
    p.add_argument("--diverge-sv", type=float, default=1e4,
                   help="|psi| above this (Sv) counts the ROM rollout as diverged")
    p.add_argument("--predivergence", choices=("window", "start", "none"), default="window",
                   help="for a diverged amplitude, also show the ROM mean over the part "
                        "of the window it survived ('window'), or over the whole rollout "
                        "up to divergence ('start'); MITgcm is averaged over the same "
                        "samples either way")
    p.add_argument("--label", default="OpInf ROM", help="legend name of the --data-dir ROM")
    p.add_argument("--data-dir2", default=None,
                   help="optional second ROM, drawn dashed with hollow markers (same taus and "
                        "--rom-suffix); a rollout diverged in the window is dropped, no "
                        "pre-divergence pair")
    p.add_argument("--label2", default="OpInf ROM 2", help="legend name of the --data-dir2 ROM")
    p.add_argument("--out", default="figs/gyre_extremes_234.pdf")
    p.add_argument("--preview-png", default=None)
    opts = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = [float(v) for v in opts.taus.split(",")]
    rom_only = [float(v) for v in opts.rom_only_taus.split(",") if v.strip()]
    train = [float(v) for v in opts.train_taus.split(",")]
    overlap = sorted(set(taus) & set(rom_only))
    if overlap:
        raise SystemExit(f"--rom-only-taus overlaps --taus at {overlap}; a tau is "
                         f"either MITgcm-backed or not")
    taus = sorted(taus + rom_only)
    xlim = (min(taus) - GRID_DX / 2, max(taus) + GRID_DX / 2)
    t0, t1 = opts.year0 * opts.spy, (opts.year0 + opts.nyears) * opts.spy
    print(f"averaging window: samples {t0}:{t1} (years {opts.year0}-{opts.year0 + opts.nyears})",
          flush=True)

    rows = []
    for tau in taus:
        only = tau in rom_only
        kind = "train" if tau in train else "test"
        rom = xr.open_dataset(
            Path(opts.data_dir) /
            f"psi_rom_Q_ROM_{kind}_run_10yrspinup_tau{tau:g}_{opts.rom_suffix}.nc",
            chunks={"time": 360})
        # A rom-only amplitude has no MITgcm run at all; keep fm undefined rather
        # than substituting anything, so a FOM point can never be invented for it.
        fom = None if only else xr.open_dataset(
            f"{opts.fom_prefix}{tau:g}/psi_20yrs.nc", chunks={"time": 360})
        fm = None if only else (
            fom["psi_bt"].isel(time=slice(t0, t1)).mean("time") / 1e6).sel(
            XG=slice(opts.xg_min, None)).compute()
        rpsi = rom["psi"].isel(time=slice(t0, t1)).sel(XG=slice(opts.xg_min, None))
        # Screen the rollout BEFORE averaging: a blown-up tail poisons the mean
        # (NaN at best, a meaningless finite number if it only partly diverged).
        rmax = np.abs(rpsi).max(dim=[d for d in rpsi.dims if d != "time"]).compute().values
        bad = ~np.isfinite(rmax) | (rmax > opts.diverge_sv)
        if bad.any():
            k = int(np.argmax(bad))
            yr = opts.year0 + k / opts.spy
            row = dict(tau=tau, kind=kind, only=only, diverge_year=yr,
                       st_f=np.nan if only else float(fm.max()),
                       sp_f=np.nan if only else -float(fm.min()),
                       st_r=np.nan, sp_r=np.nan)
            print(f"  tau {tau:g} ({'ROM only' if only else kind}): subtropical "
                  f"{row['st_f']:.1f} / -- Sv   subpolar {row['sp_f']:.1f} / -- Sv  "
                  f"(MITgcm / ROM)  "
                  f"** ROM DIVERGED at year {yr:.2f} "
                  f"({bad.mean() * 100:.0f}% of the window unusable) -- ROM point dropped",
                  flush=True)
            # Matched pre-divergence pair: same samples for both models. k == 0 means
            # it was already gone when the window opened, so fall back to the rollout start.
            mode = "start" if (opts.predivergence == "window" and k == 0) else opts.predivergence
            if mode != "none" and fom is not None:   # rom-only: no FOM to pair against
                p0 = 0 if mode == "start" else t0
                pre = slice(p0, t0 + k)
                if t0 + k - p0 >= 2:
                    pf = (fom["psi_bt"].isel(time=pre).mean("time") / 1e6).sel(
                        XG=slice(opts.xg_min, None)).compute()
                    pr = rom["psi"].isel(time=pre).sel(
                        XG=slice(opts.xg_min, None)).mean("time").compute()
                    row.update(pre_y0=p0 / opts.spy, pre_y1=(t0 + k) / opts.spy,
                               st_f_pre=float(pf.max()), sp_f_pre=-float(pf.min()),
                               st_r_pre=float(pr.max()), sp_r_pre=-float(pr.min()))
                    print(f"      pre-divergence, yrs {row['pre_y0']:.2f}-{row['pre_y1']:.2f} "
                          f"(matched samples): subtropical {row['st_f_pre']:.1f} / "
                          f"{row['st_r_pre']:.1f} Sv (ROM/FOM = {row['st_r_pre'] / row['st_f_pre']:.2f}x)"
                          f"   subpolar {row['sp_f_pre']:.1f} / {row['sp_r_pre']:.1f} Sv "
                          f"({row['sp_r_pre'] / row['sp_f_pre']:.2f}x)  (MITgcm / ROM)", flush=True)
            rows.append(row)
        else:
            rm = rpsi.mean("time").compute()
            rows.append(dict(tau=tau, kind=kind, only=only, diverge_year=None,
                             st_f=np.nan if only else float(fm.max()),
                             sp_f=np.nan if only else -float(fm.min()),
                             st_r=float(rm.max()), sp_r=-float(rm.min())))
            f_st = "  --  " if only else f"{rows[-1]['st_f']:6.1f}"
            f_sp = "  --  " if only else f"{rows[-1]['sp_f']:6.1f}"
            print(f"  tau {tau:g} ({'ROM only' if only else kind}): subtropical {f_st} / "
                  f"{rows[-1]['st_r']:.1f} Sv   subpolar {f_sp} / "
                  f"{rows[-1]['sp_r']:.1f} Sv  (MITgcm / ROM)", flush=True)
        if fom is not None:
            fom.close()
        rom.close()

    if opts.data_dir2:
        for r in rows:
            rom2 = xr.open_dataset(
                Path(opts.data_dir2) / f"psi_rom_Q_ROM_{r['kind']}_run_10yrspinup_tau"
                f"{r['tau']:g}_{opts.rom_suffix}.nc", chunks={"time": 360})
            rpsi = rom2["psi"].isel(time=slice(t0, t1)).sel(XG=slice(opts.xg_min, None))
            rmax = np.abs(rpsi).max(dim=[d for d in rpsi.dims if d != "time"]).compute().values
            bad = ~np.isfinite(rmax) | (rmax > opts.diverge_sv)
            if bad.any():
                r["st_r2"] = r["sp_r2"] = np.nan
                print(f"  tau {r['tau']:g} ({opts.label2}): ROM DIVERGED at year "
                      f"{opts.year0 + int(np.argmax(bad)) / opts.spy:.2f} -- point dropped", flush=True)
            else:
                rm = rpsi.mean("time").compute()
                r["st_r2"], r["sp_r2"] = float(rm.max()), -float(rm.min())
                print(f"  tau {r['tau']:g} ({opts.label2}): subtropical {r['st_r2']:.1f} Sv   "
                      f"subpolar {r['sp_r2']:.1f} Sv", flush=True)
            rom2.close()

    plt.rcParams.update({
        "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(6.0, 4.2), layout="constrained")
    x = [r["tau"] for r in rows]
    series = [
        ("st_f", C_SUBTROP, ":", r"Subtropical $\langle \Psi_{\mathrm{bt}} \rangle_{\max}$ (MITgcm)"),
        ("st_r", C_SUBTROP, "-", r"Subtropical $\langle \Psi_{\mathrm{bt}} \rangle_{\max}$ (" + opts.label + ")"),
        ("sp_f", C_SUBPOL, ":", r"Subpolar $\left| \langle \Psi_{\mathrm{bt}} \rangle_{\min} \right|$ (MITgcm)"),
        ("sp_r", C_SUBPOL, "-", r"Subpolar $\left| \langle \Psi_{\mathrm{bt}} \rangle_{\min} \right|$ (" + opts.label + ")"),
    ]
    if opts.data_dir2:
        series[2:2] = [("st_r2", C_SUBTROP, "--", r"Subtropical $\langle \Psi_{\mathrm{bt}} \rangle_{\max}$ (" + opts.label2 + ")")]
        series.append(("sp_r2", C_SUBPOL, "--", r"Subpolar $\left| \langle \Psi_{\mathrm{bt}} \rangle_{\min} \right|$ (" + opts.label2 + ")"))
    lo, hi = [float(v) for v in opts.ylim.split(",")]

    # Amplitudes where the rollout blew up inside the window: shade the column so
    # the gap in the ROM curves reads as "no model point", not "missing data".
    diverged = [r for r in rows if r.get("diverge_year") is not None]
    for r in diverged:
        ax.axvspan(r["tau"] - GRID_DX / 2, r["tau"] + GRID_DX / 2,
                   color=C_DIVERGE, alpha=0.16, lw=0, zorder=0)

    # Matched pre-divergence pair for each diverged amplitude: hollow markers, and a
    # thin connector showing how far the ROM has already drifted from MITgcm over the
    # SAME samples. Off-scale ROM values become a clipped arrow carrying its value.
    # The two gyres get their own sub-column (and their labels lean apart) because at
    # tau0.1 both ROM values are off-scale and would otherwise overprint each other.
    has_pre = False
    for r in diverged:
        if "st_r_pre" not in r:
            continue
        has_pre = True
        mk = "o" if r["kind"] == "train" else "*"
        for fk, rk, color, off, ha in (
                ("st_f_pre", "st_r_pre", C_SUBTROP, +0.16, "right"),
                ("sp_f_pre", "sp_r_pre", C_SUBPOL, +0.40, "left")):
            xp = r["tau"] + off * GRID_DX
            yf, yr_ = r[fk], r[rk]
            ytop = min(yr_, hi - 0.015 * (hi - lo))
            ax.plot([xp, xp], [yf, ytop], color=color, lw=0.8, alpha=0.55, zorder=3)
            ax.scatter([xp], [yf], facecolor="white", edgecolor=color, linewidths=0.9,
                       marker=mk, s=22 if mk == "o" else 70, zorder=6)
            if yr_ <= hi:
                ax.scatter([xp], [yr_], facecolor="white", edgecolor=color, linewidths=0.9,
                           marker=mk, s=22 if mk == "o" else 70, zorder=6)
            else:
                ax.annotate("", xy=(xp, hi), xytext=(xp, ytop),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                                            shrinkA=0, shrinkB=0), zorder=6)
                ax.annotate(f"{yr_:.0f} Sv",
                            xy=(xp + (-0.08 if ha == "right" else 0.08) * GRID_DX,
                                hi - 0.035 * (hi - lo)),
                            ha=ha, va="top", fontsize=6.2, color=color, zorder=7)
    for key, color, ls, label in series:
        # A rom-only amplitude is dropped from the MITgcm series entirely (not set to
        # NaN), so the MITgcm line spans straight across it instead of breaking there.
        pts = [r for r in rows if not (key.endswith("_f") and r["only"])]
        y = [r.get(key, np.nan) for r in pts]
        ax.plot([r["tau"] for r in pts], y, color=color, ls=ls, lw=1.3,
                label=label, zorder=2)
        for r, yi in zip(pts, y):
            if not np.isfinite(yi):
                continue
            # rom-only taus are kind == "test" and share the test star; what marks
            # them out is the absence of a MITgcm marker at that tau.
            mk, sz = ("o", 22) if r["kind"] == "train" else ("*", 70)
            if key.endswith("_r2"):     # second ROM: hollow markers
                ax.scatter([r["tau"]], [yi], facecolor="white", edgecolor=color, linewidths=0.9,
                           marker=mk, s=sz, zorder=5)
            else:
                ax.scatter([r["tau"]], [yi], color=color, marker=mk, s=sz, zorder=5)
    ax.plot([], [], color="gray", marker="o", ls="None", ms=4, label="Training amplitude")
    ax.plot([], [], color="gray", marker="*", ls="None", ms=7, label="Test amplitude")
    if diverged:
        yrs = ", ".join(rf"$\tau${r['tau']:g} from yr {r['diverge_year']:.1f}" for r in diverged)
        ax.fill_between([], [], [], color=C_DIVERGE, alpha=0.16, lw=0,
                        label=f"ROM rollout unstable\n({yrs})")
    if has_pre:
        w = ", ".join(f"{r['pre_y0']:.1f}\u2013{r['pre_y1']:.1f}"
                      for r in diverged if "st_r_pre" in r)
        ax.plot([], [], color="gray", marker="o", ls="None", ms=4, mfc="white", mew=0.9,
                label=f"Pre-divergence mean, yrs {w}\n(matched MITgcm & ROM samples)")

    ax.set_ylim(lo, hi)
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"Wind Stress Forcing Amplitude, $\tau_{\max}$ [N m$^{-2}$]", labelpad=3)
    ax.set_ylabel(r"Barotropic Streamfunction, $\Psi_{\mathrm{bt}}$ [Sv]", labelpad=3)
    # gridlines every GRID_DX/GRID_DY, anchored to multiples of the spacing so
    # they pass through the sampled amplitudes; only those carry an x label
    ticks = lambda a, b, d: np.arange(int(np.ceil(a / d - 1e-9)),
                                      int(np.floor(b / d + 1e-9)) + 1) * d
    xt = ticks(*xlim, GRID_DX)
    ax.set_xticks(xt)
    # ':g' not ':.1f' -- with rom-only taus the gridlines land on half-steps, and
    # '.1f' silently rounds 0.25 -> "0.2" and 0.45 -> "0.5".
    ax.set_xticklabels([f"{v:g}" if any(abs(v - t) < 1e-9 for t in x) else "" for v in xt])
    ax.set_yticks(ticks(lo, hi, GRID_DY))
    ax.set_aspect(GRID_DX / GRID_DY, adjustable="box")
    ax.grid(True, ls=":", alpha=0.5, lw=0.5)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=6.8,
               loc="outside upper center", ncol=3, framealpha=0.95,
               handlelength=1.8, handletextpad=0.5, borderpad=0.3, labelspacing=0.3,
               columnspacing=1.0)

    # The locked aspect letterboxes the axes inside whichever of the two figure
    # dimensions binds, so size the FIGURE from the axes box we want rather than
    # nudging one dimension and hoping. Target axes width = legend width; the
    # matching height follows from the cell aspect. The non-axes overhead (legend,
    # labels, ticks) is measured with the aspect temporarily released, since a
    # letterboxed axes reports the shrunken box and would hide that overhead.
    n_cells_x = len(ticks(*xlim, GRID_DX)) - 1 + 1.0   # full cells + 2 half margins
    n_cells_y = (hi - lo) / GRID_DY
    for _ in range(4):
        fig.canvas.draw()
        aw = ax.get_position().width * fig.get_figwidth()
        lw = fig.legends[0].get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
        if abs(lw - aw) < 0.01:
            break
        ax.set_aspect("auto")
        fig.canvas.draw()
        pos = ax.get_position()
        pad_x = (1 - pos.width) * fig.get_figwidth()
        pad_y = (1 - pos.height) * fig.get_figheight()
        ax.set_aspect(GRID_DX / GRID_DY, adjustable="box")
        fig.set_size_inches(lw + pad_x, lw * n_cells_y / n_cells_x + pad_y)

    fig.canvas.draw()
    tr = ax.transData
    x0, y0 = tr.transform((xlim[0], lo))
    dx_px = tr.transform((xlim[0] + GRID_DX, lo))[0] - x0
    dy_px = tr.transform((xlim[0], lo + GRID_DY))[1] - y0
    print(f"grid cell: {dx_px:.1f} x {dy_px:.1f} px  (x/y = {dx_px / dy_px:.4f}, "
          f"1.0 = square)", flush=True)
    _lg = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
    print(f"[fit] axes {ax.get_position().width * fig.get_figwidth():.3f} in wide, "
          f"legend {_lg.width / fig.dpi:.3f} in wide, "
          f"fig {fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in", flush=True)

    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, facecolor="white")
    print(f"wrote {out}", flush=True)
    if opts.preview_png:
        fig.savefig(opts.preview_png, dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
