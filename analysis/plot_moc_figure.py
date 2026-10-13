#!/usr/bin/env python3
"""Journal-ready overturning figure for one scenario:
(a) time-mean meridional overturning streamfunction, MITgcm vs OpInf ROM
(b) overturning strength index, FOM vs ROM -- either at a section latitude
    (--index section, the default) or the basin-wide maximum over both latitude
    and depth restricted to the positive upper cell (--index basin).

The maximum is always taken per snapshot and only then smoothed, so the plotted
level is comparable with a time series and not with max-of-time-mean.

Reads analysis/moc/moc_tau<TAU>.nc (dims depth, lat, time; time in years) as
written by compute_moc.py.

Sized for JGR: ML/AI (7.5 in max width, 8-9 pt type, embedded TrueType fonts,
rasterized filled contours); writes PDF.

Usage:
    python3 src/plot_moc_figure.py --tau 0.3 --lat 26.5 --out figs/amoc.pdf
    python3 src/plot_moc_figure.py --tau 0.3 --index basin --out figs/amoc_v2.pdf
    python3 src/plot_moc_figure.py --tau 0.3 --index basin --min-depth 200 \
            --out figs/amoc_deepmax.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))
import paths  # noqa: E402

FIG_W = 7.5
FS_TICK = 7.5
FS_LABEL = 8.5
FS_TITLE = 9.5
FS_PANEL = 10.0
COLOR = "#0072B2"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moc-dir", default=paths.BASE_RUN + "analysis/moc/")
    p.add_argument("--tau", default="0.3")
    p.add_argument("--lat", type=float, default=26.5, help="section latitude for panel (b)")
    p.add_argument("--lat-band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="average the section index over this latitude band instead of a "
                        "single --lat; the deep-cell peak is flat to within 2%% over "
                        "~58-62N and carries a ~0.1 Sv two-gridpoint zigzag, so a single "
                        "argmax latitude is not reproducible across scenarios")
    p.add_argument("--index", choices=("section", "basin"), default="section",
                   help="panel (b): max over depth at --lat, or over both lat and depth")
    p.add_argument("--min-depth", type=float, default=0.0,
                   help="restrict the panel-(b) index to depths below this many metres, "
                        "so the shallow wind-driven Ekman cell is excluded and the index "
                        "measures the deep (AMOC-analogue) cell")
    p.add_argument("--n-days", type=int, default=1)
    p.add_argument("--roll-days", type=int, default=15, help="centred rolling-mean window")
    p.add_argument("--roll-which", choices=("both", "fom", "rom", "none"), default="both",
                   help="which panel-(b) series to smooth; anything but 'both' makes the "
                        "two curves unequally treated, so the ROM's variability can no "
                        "longer be read against the MITgcm's")
    p.add_argument("--clim", type=float, default=10.0, help="contour limit in Sv")
    p.add_argument("--nlev", type=int, default=21)
    p.add_argument("--lat-line", default=":",
                   help="linestyle of the vertical marker drawn at --lat in panel (a)")
    p.add_argument("--ts-label", default=None,
                   help=r"panel (b) y-label; default 'MOC @ <lat>$^\circ$N (Sv)'")
    p.add_argument("--out", default="figs/amoc.pdf")
    p.add_argument("--preview-png", default=None)
    opts = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    ds = xr.open_dataset(Path(opts.moc_dir) / f"moc_tau{opts.tau}.nc")
    levels = np.linspace(-opts.clim, opts.clim, opts.nlev)
    fld_fom = ds["psi_fom"].mean("time")
    fld_rom = ds["psi_rom"].mean("time")
    for nm, fld in (("MITgcm", fld_fom), ("ROM", fld_rom)):
        v = np.abs(fld.values)
        print(f"time-mean psi {nm}: max |psi| = {np.nanmax(v):.2f} Sv, "
              f"{100 * np.nanmean(v > opts.clim):.2f}% of cells beyond +/-{opts.clim:g} Sv",
              flush=True)

    win = max(int(opts.roll_days / opts.n_days), 1)
    # the global maximum of psi in SOMA is a ~50 m wind-driven subtropical Ekman
    # cell, not the deep buoyancy-driven cell that is the North Atlantic AMOC
    # analogue; --min-depth cuts the former out of the index
    deep = (lambda a: a.where(ds.depth < -opts.min_depth, drop=True)) if opts.min_depth > 0 \
        else (lambda a: a)
    if opts.index == "section" and opts.lat_band:
        lo, hi = sorted(opts.lat_band)
        band = ds.lat.sel(lat=slice(lo, hi))
        idx = lambda var: deep(ds[var].sel(lat=band)).max("depth").mean("lat")
    elif opts.index == "section":
        idx = lambda var: deep(ds[var].sel(lat=opts.lat, method="nearest")).max("depth")
    else:
        # clip at zero first, so the maximum is the strength of the positive
        # (upper) cell by construction rather than whichever extremum is larger
        idx = lambda var: deep(ds[var]).clip(min=0.0).max(("depth", "lat"))
    roll = lambda a: a.rolling(time=win, center=True, min_periods=1).mean()
    keep = lambda a: a
    r_fom = roll if opts.roll_which in ("both", "fom") else keep
    r_rom = roll if opts.roll_which in ("both", "rom") else keep
    ts_fom, ts_rom = r_fom(idx("psi_fom")), r_rom(idx("psi_rom"))
    lat_used = float(ds.lat.sel(lat=opts.lat, method="nearest"))
    if opts.index == "section" and opts.lat_band:
        lo, hi = sorted(opts.lat_band)
        print(f"panel (b): max over depth averaged over {lo:g}-{hi:g}N "
              f"({ds.lat.sel(lat=slice(lo, hi)).size} grid rows), "
              f"{win}-sample rolling mean on {opts.roll_which}", flush=True)
    elif opts.index == "section":
        print(f"panel (b): section at {lat_used:.2f}N (requested {opts.lat}), "
              f"{win}-sample rolling mean on {opts.roll_which}", flush=True)
    else:
        print(f"panel (b): basin-wide positive-cell maximum over lat and depth"
              + (f" below {opts.min_depth:g} m" if opts.min_depth > 0 else "")
              + f", {win}-sample rolling mean", flush=True)
        for nm, var in (("MITgcm", "psi_fom"), ("ROM", "psi_rom")):
            fd = deep(ds[var]).clip(min=0.0).transpose("time", "depth", "lat")
            a = fd.values
            nt, nz, ny = a.shape
            kz, kl = np.unravel_index(a.reshape(nt, -1).argmax(1), (nz, ny))
            la, z = fd.lat.values[kl], fd.depth.values[kz]
            print(f"    {nm:>7}: max at {np.median(la):.1f}N, {np.median(z):.0f} m "
                  f"(median); {100 * np.mean(la < 45):.0f}% of the time south of 45N",
                  flush=True)
    for nm, ts in (("MITgcm", ts_fom), ("ROM", ts_rom)):
        print(f"    {nm:>7} index: mean {float(ts.mean()):.2f} Sv, "
              f"std {float(ts.std()):.2f} Sv", flush=True)

    plt.rcParams.update({
        "font.size": FS_LABEL, "axes.titlesize": FS_TITLE, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(FIG_W, 5.0), layout="constrained")
    # middle row is an empty spacer between the maps and the time series
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 0.10, 0.62])
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.06)
    ax_fom = fig.add_subplot(gs_top[0, 0])
    ax_rom = fig.add_subplot(gs_top[0, 1])

    cf = None
    for ax, fld, title in ((ax_fom, fld_fom, "MITgcm"), (ax_rom, fld_rom, "OpInf ROM")):
        cf = fld.plot.contourf(ax=ax, x="lat", y="depth", levels=levels, cmap="RdBu_r",
                               extend="both", add_colorbar=False)
        # contours stay vector: the field is only 31 x 248, so file size is fine
        fld.plot.contour(ax=ax, x="lat", y="depth", levels=levels, colors="k",
                         linewidths=0.3, alpha=0.5)
        ax.set_title(title, pad=4)
        ax.set_xlabel("Latitude (°N)", labelpad=2)
        ax.set_ylabel("")
        ax.grid(True, ls="--", alpha=0.4, lw=0.4)
        ax.set_ylim(float(ds.depth.min()), 0)
        if opts.index == "section" and opts.lat_band:
            ax.axvspan(*sorted(opts.lat_band), color="0.25", alpha=0.10, lw=0)
        elif opts.index == "section":
            # marks the section used in panel (b); meaningless for a basin-wide max
            ax.axvline(lat_used, color="0.25", ls=opts.lat_line, lw=0.8)
    ax_fom.set_ylabel("Depth (m)", labelpad=2)
    ax_rom.tick_params(axis="y", labelleft=False)

    cbar = fig.colorbar(cf, ax=[ax_fom, ax_rom], location="right", shrink=0.95, pad=0.015,
                        extend="both")
    cbar.set_ticks(np.arange(-opts.clim, opts.clim + 1, 2))
    cbar.set_label("Streamfunction (Sv)", fontsize=FS_LABEL, labelpad=4)
    cbar.ax.tick_params(labelsize=FS_TICK, width=0.6, length=2.5)
    cbar.outline.set_linewidth(0.6)

    ax_ts = fig.add_subplot(gs[2])
    ax_ts.plot(ds.time, ts_rom, ls="-", lw=1.4, color=COLOR, label="OpInf ROM", zorder=2)
    ax_ts.plot(ds.time, ts_fom, ls="--", lw=1.0, color=COLOR, label="MITgcm", zorder=3)
    ax_ts.set_xlabel("Year", labelpad=2)
    if opts.index != "section":
        default_label = "MOC strength (Sv)"
    elif opts.lat_band:
        default_label = "MOC, {:g}–{:g}°N mean (Sv)".format(*sorted(opts.lat_band))
    else:
        default_label = f"MOC @ {lat_used:.1f}°N (Sv)"
    ax_ts.set_ylabel(opts.ts_label or default_label, labelpad=3)
    ax_ts.set_xlim(0, float(np.ceil(ds.time.max())))
    ax_ts.set_xticks(np.arange(0, int(np.ceil(ds.time.max())) + 1, 1))
    lo = float(min(ts_fom.min(), ts_rom.min()))
    hi = float(max(ts_fom.max(), ts_rom.max()))
    ax_ts.set_ylim(lo - 0.08 * (hi - lo), hi + 0.24 * (hi - lo))  # headroom for the legend
    ax_ts.grid(True, ls="--", alpha=0.3, lw=0.5)
    ax_ts.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none",
                 fontsize=6.5, handlelength=1.5, handletextpad=0.5, borderpad=0.25,
                 labelspacing=0.3)

    fig.canvas.draw()
    # both panel labels share (a)'s x so they line up down the left edge
    tags = ((ax_fom, "(a)", 0.012), (ax_ts, "(b)", 0.018))
    tag_y = {tag: min(ax_ref.get_position().y1 + dy, 0.995) for ax_ref, tag, dy in tags}
    x_tag = max(ax_fom.get_position().x0 - 0.075, 0.004)
    # a y-label long enough to fill its axes overflows above it, into the tag row;
    # only then move the tag column further left, so a short label (the section
    # index) leaves the placement exactly as it was
    inv, rend = fig.transFigure.inverted(), fig.canvas.get_renderer()
    for ax_ref, tag, _ in tags:
        lbl = ax_ref.yaxis.get_label()
        if not lbl.get_text():
            continue
        bb = inv.transform_bbox(lbl.get_window_extent(rend))
        if bb.y1 > tag_y[tag]:
            x_tag = min(x_tag, bb.x0 - 0.014)
    x_tag = max(x_tag, 0.004)
    for _, tag, _ in tags:
        fig.text(x_tag, tag_y[tag], tag, fontsize=FS_PANEL, va="bottom", ha="left")

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
