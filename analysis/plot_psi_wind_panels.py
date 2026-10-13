#!/usr/bin/env python3
"""Wind forcing and barotropic streamfunction across forcing amplitudes:
one row per tau_max, columns = wind-stress profile, MITgcm Psi, OpInf ROM Psi,
and (with --error) the Error panel.

SIGN CONVENTION: the Error column is **OpInf ROM minus MITgcm** (model minus
reference), matching plot_surface_snapshots.py. Red = the ROM's gyre is too
strong/positive there, blue = too weak/negative. It is the signed difference of
the two time-reduced fields, NOT <|ROM - FOM|>.

Psi fields are reduced over time in one of four ways, in this order of precedence:
``--day N`` (a single snapshot); ``--month M`` (climatology of calendar month M
over years ``--year0``..``--year0 + --nyears``, e.g. every January of years 3-10);
``--last-days N`` (mean over the final N samples of the rollout, the default: 30,
i.e. the last month); or ``--year0``/``--nyears`` (mean over a contiguous year
range). Months are the 30-day months of the model's 360-day calendar. The same
sample indices are used for MITgcm and the ROM, so the two are aligned in time.
Land is masked with hFacW at the surface, which lives on the same (YC, XG) grid
as Psi, and left white.

The wind-stress profile is the analytic pattern used by forcing.OceanForcing,
g(y) = (1 - xi*dY) exp(-dY^2) cos(pi*dY) with dY = (y - y_c)/30, scaled by
tau_max and evaluated on the Psi latitude grid (unmasked, so it is the forcing
function rather than its land-masked projection).

Sized for JGR: ML/AI (7.5 in max width, 8-9 pt type, embedded TrueType fonts,
rasterized maps); writes PDF.

Usage:
    python3 src/plot_psi_wind_panels.py --taus 0.2,0.3,0.4 --month 1 --year0 3 --nyears 7 \
        --out figs/psi_jan_avg_234.pdf

    # four amplitudes + the ROM-minus-MITgcm Error column
    python3 src/plot_psi_wind_panels.py --taus 0.2,0.3,0.4,0.5 --error \
        --month 1 --year0 3 --nyears 7 --out figs/psi_jan_avg.pdf
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import xarray as xr
import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))
import paths  # noqa: E402

FS_TICK = 7.5
FS_LABEL = 8.5
FS_TITLE = 9.5

LAT_TICKS = [15, 35, 55, 75]
# the model runs a 360-day calendar with twelve 30-day months
DAYS_PER_MONTH = 30
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
# width of the reserved left strip holding the per-row tau_max labels, in figure
# fractions; keeps them clear of the latitude axis labels without clipping
TAG_STRIP = 0.052


def wind_profile(lat, tau_max, y_c=45.0, xi=0.5, width=30.0):
    d = (np.asarray(lat) - y_c) / width
    return tau_max * (1 - xi * d) * np.exp(-d**2) * np.cos(np.pi * d)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=paths.BASE_RUN)
    p.add_argument("--fom-prefix", default=paths.FOM_PREFIX)
    p.add_argument("--rom-suffix", default="mean_var_r105_10trainingyrs")
    p.add_argument("--grid-dir", default=paths.GRID_DIR)
    p.add_argument("--taus", default="0.2,0.3,0.4")
    p.add_argument("--train-taus", default="0.2,0.4")
    p.add_argument("--nt", type=int, default=3600, help="length of the rollout in samples")
    p.add_argument("--day", type=int, default=None,
                   help="single snapshot index instead of a time average (e.g. 1080)")
    p.add_argument("--month", type=int, default=None, choices=range(1, 13),
                   metavar="M",
                   help="climatology of calendar month M over the --year0/--nyears range")
    p.add_argument("--last-days", type=int, default=30,
                   help="mean over the final N samples of the rollout (30 = last month)")
    p.add_argument("--year0", type=int, default=3, help="first year of the averaging window")
    p.add_argument("--nyears", type=int, default=7, help="length of the averaging window")
    p.add_argument("--spy", type=int, default=360, help="samples per year")
    p.add_argument("--vmax", type=float, default=90.0)
    p.add_argument("--cmap", default="RdBu_r")
    p.add_argument("--error", action="store_true",
                   help="add a fourth column, Error = OpInf ROM - MITgcm (signed)")
    p.add_argument("--err-vmax", type=float, default=None,
                   help="colour limit for the Error column; default is --err-pctl "
                        "of |ROM-MITgcm| over all rows, rounded up")
    p.add_argument("--err-pctl", type=float, default=99.5,
                   help="percentile of |error| used when --err-vmax is not given")
    p.add_argument("--single-cbar", action="store_true",
                   help="with --error: one colorbar, right of the Error column, shared by "
                        "all maps (requires the Error limit to equal --vmax)")
    p.add_argument("--out", default="figs/psi_jan_avg_234.pdf")
    p.add_argument("--preview-png", default=None)
    opts = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    taus = [float(v) for v in opts.taus.split(",")]
    train = [float(v) for v in opts.train_taus.split(",")]
    # a snapshot and a window mean are both reductions over the time axis
    if opts.day is not None:
        if not 0 <= opts.day < opts.nt:
            raise SystemExit(f"--day {opts.day} outside the rollout (0..{opts.nt - 1})")
        print(f"single snapshot: day {opts.day} "
              f"(year {opts.day / opts.spy:.3g} of the rollout)", flush=True)
        reduce_t = lambda da: da.isel(time=opts.day)
    elif opts.month is not None:
        m0 = (opts.month - 1) * DAYS_PER_MONTH
        years = range(opts.year0, opts.year0 + opts.nyears)
        idx = np.concatenate([np.arange(y * opts.spy + m0,
                                        y * opts.spy + m0 + DAYS_PER_MONTH)
                              for y in years])
        idx = idx[idx < opts.nt]
        if idx.size == 0:
            raise SystemExit(f"--month {opts.month} over years {opts.year0}-"
                             f"{opts.year0 + opts.nyears} selects no samples "
                             f"within the {opts.nt}-sample rollout")
        print(f"{MONTH_NAMES[opts.month - 1]} climatology over years "
              f"{opts.year0}-{opts.year0 + opts.nyears}: {idx.size} samples "
              f"({idx.size // DAYS_PER_MONTH} months, first {idx[0]}, last {idx[-1]})",
              flush=True)
        reduce_t = lambda da: da.isel(time=idx).mean("time")
    else:
        if opts.last_days:
            t0, t1 = max(opts.nt - opts.last_days, 0), opts.nt
            print(f"averaging the final {t1 - t0} days "
                  f"(samples {t0}:{t1}, years {t0 / opts.spy:.3g}-{t1 / opts.spy:.3g})",
                  flush=True)
        else:
            t0 = opts.year0 * opts.spy
            t1 = min(t0 + opts.nyears * opts.spy, opts.nt)
            print(f"averaging years {opts.year0}-{opts.year0 + opts.nyears} "
                  f"(samples {t0}:{t1} of {opts.nt})", flush=True)
        reduce_t = lambda da: da.isel(time=slice(t0, t1)).mean("time")

    import xmitgcm
    g = xmitgcm.open_mdsdataset(opts.grid_dir, iters=None)
    wet = np.asarray(g["hFacW"].isel(Z=0).values) > 0   # (YC, XG), the Psi grid

    fom_data, rom_data, xg, yc = {}, {}, None, None
    for tau in taus:
        kind = "train" if tau in train else "test"
        dsf = xr.open_dataset(f"{opts.fom_prefix}{tau:g}/psi_20yrs.nc", chunks={"time": 90})
        dsr = xr.open_dataset(
            Path(opts.data_dir) /
            f"psi_rom_Q_ROM_{kind}_run_10yrspinup_tau{tau:g}_{opts.rom_suffix}.nc",
            chunks={"time": 90})
        fm = (reduce_t(dsf["psi_bt"]) / 1e6).compute()
        rm = reduce_t(dsr["psi"]).compute()
        xg = np.asarray(fm["XG"].values)
        yc = np.asarray(fm["YC"].values)
        fom_data[tau] = np.where(wet, fm.values, np.nan)
        rom_data[tau] = np.where(wet, rm.values, np.nan)
        print(f"  tau {tau:g} ({kind}): Psi range MITgcm "
              f"[{np.nanmin(fom_data[tau]):.1f}, {np.nanmax(fom_data[tau]):.1f}] "
              f"ROM [{np.nanmin(rom_data[tau]):.1f}, {np.nanmax(rom_data[tau]):.1f}] Sv, "
              f"{100 * np.nanmean(np.abs(fom_data[tau]) > opts.vmax):.1f}% / "
              f"{100 * np.nanmean(np.abs(rom_data[tau]) > opts.vmax):.1f}% beyond "
              f"+/-{opts.vmax:g}", flush=True)
        dsf.close(); dsr.close()

    plt.rcParams.update({
        "font.size": FS_LABEL, "axes.titlesize": FS_TITLE, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    cmap = copy.copy(plt.get_cmap(opts.cmap))
    cmap.set_bad("white")
    extent = [xg.min(), xg.max(), yc.min(), yc.max()]

    n = len(taus)
    err_data = {t: rom_data[t] - fom_data[t] for t in taus} if opts.error else {}
    if opts.error:
        allerr = np.concatenate([e[np.isfinite(e)] for e in err_data.values()])
        elim = opts.err_vmax or float(np.ceil(
            np.percentile(np.abs(allerr), opts.err_pctl) / 5.0) * 5.0)
        for tau in taus:
            e = err_data[tau]
            print(f"  tau {tau:g} error (ROM - MITgcm): mean {np.nanmean(e):+.2f}, "
                  f"RMS {np.sqrt(np.nanmean(e**2)):.2f}, max|.| {np.nanmax(np.abs(e)):.1f} Sv",
                  flush=True)
        print(f"  error colour limit +/-{elim:g} Sv "
              f"({100 * np.mean(np.abs(allerr) > elim):.2f}% of wet points saturate)",
              flush=True)
        if opts.single_cbar and elim != opts.vmax:
            raise SystemExit(f"--single-cbar needs the Error limit ({elim:g}) to equal "
                             f"--vmax ({opts.vmax:g}); pass --err-vmax {opts.vmax:g}")

    # One colorbar per scale, each attached to the group of axes it describes, so
    # matplotlib sizes and places them. (plot_surface_snapshots.py gives its colorbars
    # dedicated gridspec columns instead, but there each bar serves a single row; here
    # a bar spans all rows, and at that height a fixed-ratio column is a huge slab.)
    ncol = 4 if opts.error else 3
    ratios = [1.15, 3, 3, 3] if opts.error else [1.15, 3, 3]
    # maps are square (aspect="equal"). With the Error column the panels get narrow
    # enough to be width-limited, so size rows from the panel width; the 3-column
    # layout keeps its original height, where the panels are height-limited instead.
    map_w = 7.5 * 3 / sum(ratios)
    fig_h = map_w * n + 0.85 if opts.error else 2.55 * n
    fig = plt.figure(figsize=(7.5, fig_h), layout="constrained")
    gs = gridspec.GridSpec(n, ncol, figure=fig, width_ratios=ratios)
    map_axes, psi_axes, err_axes = [], [], []
    wind_axes, im, im_e = [], None, None
    for i, tau in enumerate(taus):
        ax_w = fig.add_subplot(gs[i, 0])
        ax_f = fig.add_subplot(gs[i, 1])
        ax_r = fig.add_subplot(gs[i, 2])
        ax_e = fig.add_subplot(gs[i, 3]) if opts.error else None
        psi_axes += [ax_f, ax_r]
        if opts.error:
            err_axes.append(ax_e)
        map_axes += [ax_f, ax_r] + ([ax_e] if opts.error else [])
        wind_axes.append(ax_w)

        ax_w.plot(wind_profile(yc, tau), yc, lw=1.3, color="teal")
        ax_w.axvline(0, color="gray", ls="--", lw=0.7, alpha=0.7)
        ax_w.set_xlim(-0.4, 0.6)
        ax_w.set_ylim(yc.min(), yc.max())
        ax_w.set_yticks(LAT_TICKS)
        ax_w.grid(True, ls=":", lw=0.5, alpha=0.5)
        ax_w.set_ylabel("Latitude [°N]", labelpad=2)

        panels = [(ax_f, fom_data[tau], cmap, opts.vmax),
                  (ax_r, rom_data[tau], cmap, opts.vmax)]
        if opts.error:
            panels.append((ax_e, err_data[tau], cmap, elim))
        for ax, fld, cm, lim in panels:
            h = ax.imshow(fld, origin="lower", extent=extent, cmap=cm,
                          vmin=-lim, vmax=lim, aspect="equal", rasterized=True)
            if ax is ax_e:
                im_e = h
            else:
                im = h
            # ticks mark the same latitudes as the wind axis; values only there
            ax.set_yticks(LAT_TICKS)
            ax.tick_params(axis="y", left=True, labelleft=False)
        last = i == n - 1
        labelled = [(ax_w, r"$\tau$ [N m$^{-2}$]"), (ax_f, "Longitude [°E]"),
                    (ax_r, "Longitude [°E]")]
        if opts.error:
            labelled.append((ax_e, "Longitude [°E]"))
        for ax, lab in labelled:
            if last:
                ax.set_xlabel(lab, labelpad=2)
            else:
                ax.set_xticklabels([])
        if i == 0:
            ax_w.set_title(r"$\tau_{\max}$ profile", pad=5)
            ax_f.set_title("MITgcm", pad=5)
            ax_r.set_title("OpInf ROM", pad=5)
            if opts.error:
                ax_e.set_title("Error", pad=5)

    psi_lab = r"Barotropic Streamfunction, $\Psi_{\mathrm{bt}}$ [Sv]"
    if opts.error and opts.single_cbar:
        # the Error panels share the Psi range, so one bar at the far right serves all
        groups = [(map_axes, im, psi_lab)]
    elif opts.error:
        groups = [(psi_axes, im, psi_lab),
                  (err_axes, im_e, r"Error (OpInf ROM $-$ MITgcm) [Sv]")]
    else:
        groups = [(map_axes, im, psi_lab)]
    for axs, handle, lab in groups:
        cbar = fig.colorbar(handle, ax=axs, location="right", shrink=0.85, pad=0.035,
                            aspect=35, extend="both")
        cbar.set_label(lab, fontsize=FS_LABEL, labelpad=4)
        cbar.ax.tick_params(labelsize=FS_TICK, width=0.6, length=2.5)
        cbar.outline.set_linewidth(0.6)

    # reserve a strip at the left for the row tags, then place them in figure
    # coords -- axes-relative text would either collide with the latitude labels
    # or fall off the canvas, since constrained_layout ignores stray text
    fig.get_layout_engine().set(rect=(TAG_STRIP, 0.0, 1.0 - TAG_STRIP, 1.0))

    # With the Error column the maps are width-limited, so aspect="equal" shrinks each
    # axes box inside a taller row slot and leaves stripes of dead space. Squeeze the
    # figure height until the row gaps are just ROW_GAP, which also pulls the wind
    # profiles (no fixed aspect, so they fill their slot) down to the map height.
    if opts.error:
        # Solve for the height rather than subtracting the excess repeatedly: once the
        # figure is short enough the maps become HEIGHT-limited and shrink with it, so
        # a subtract-and-repeat loop chases its own tail all the way down.
        ROW_GAP = 0.10   # inches between stacked map panels
        for _ in range(3):
            # Release the aspect so each map fills its gridspec cell: that gives the
            # width actually available, which is what the square side should be.
            # Measuring it while the maps are aspect-limited instead reports whatever
            # they have already shrunk to, and the solve converges downward on itself.
            for a in map_axes:
                a.set_aspect("auto")
            fig.canvas.draw()
            fh = fig.get_figheight()
            bx = sorted((a.get_position() for a in psi_axes[::2]), key=lambda b: b.y0)
            w_cell = bx[0].width * fig.get_figwidth()
            fixed = (1 - bx[-1].y1) * fh + bx[0].y0 * fh  # titles, xlabels, margins
            for a in map_axes:
                a.set_aspect("equal")
            target = n * w_cell + fixed + (n - 1) * ROW_GAP
            if abs(target - fh) < 0.03:
                break
            fig.set_figheight(target)
        fig.canvas.draw()
        bx = sorted((a.get_position() for a in psi_axes[::2]), key=lambda b: b.y0)
        print(f"  fitted figure: {fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in, "
              f"map panel {bx[0].width * fig.get_figwidth():.2f} x "
              f"{bx[0].height * fig.get_figheight():.2f} in", flush=True)

    fig.canvas.draw()
    for tau, ax_w in zip(taus, wind_axes):
        pos = ax_w.get_position()
        fig.text(TAG_STRIP * 0.35, 0.5 * (pos.y0 + pos.y1),
                 r"$\tau_{\max} = $" + f"{tau:g}", fontsize=FS_TITLE,
                 va="center", ha="center", rotation=90)

    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, facecolor="white")
    print(f"wrote {out}", flush=True)
    if opts.preview_png:
        fig.savefig(opts.preview_png, dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
