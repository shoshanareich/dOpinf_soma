#!/usr/bin/env python3
"""Journal-ready combined figure for one surface variable:
rows 1-2  spatial snapshots at several times (MITgcm / OpInf ROM, shared colorbar)
row 3     domain-mean time series, FOM vs ROM, with the snapshot times marked.

Reads the split surface files from export_surface_fields.py
(<data-dir>/surface_fields/{fom,rom}_surface_<scenario>.nc, dims (time, y, x)
= (time, lat, lon)).

The land mask is taken from the temperature field (T = 0 only on land) rather
than from ``var != 0``, so that genuine zero crossings of Eta/U/V are kept.

Sized for JGR: ML/AI (7.5 in max width, 8-9 pt type, embedded TrueType fonts,
rasterized pcolormesh); writes PDF.

Usage:
    python3 src/plot_snapshots_timeseries.py --var Eta --times 100,300,930,2880 \
        --out figs/ssh_10yrs.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))
import paths  # noqa: E402

VAR_SETTINGS = {
    "T":   {"name": "Potential Temperature", "unit": r"$^\circ$C", "cmap": "thermal",
            "vmin": 3.5, "vmax": 22.5, "cfmt": "%.1f"},
    "S":   {"name": "Salinity", "unit": "psu", "cmap": "haline",
            "vmin": 35.2, "vmax": 35.7, "cfmt": "%.2f"},
    "Eta": {"name": "Sea Surface Height", "unit": "m", "cmap": "RdBu_r",
            "vmin": -1.0, "vmax": 1.0, "cfmt": "%.1f"},
    "U":   {"name": "Zonal Velocity", "unit": r"m s$^{-1}$", "cmap": "RdBu_r",
            "vmin": -0.8, "vmax": 0.8, "cfmt": "%.1f"},
    "V":   {"name": "Meridional Velocity", "unit": r"m s$^{-1}$", "cmap": "RdBu_r",
            "vmin": -0.8, "vmax": 0.8, "cfmt": "%.1f"},
}

FIG_W = 7.5
FS_TICK = 7.5
FS_LABEL = 8.5
FS_TITLE = 9.0
COLOR = "#0072B2"


def load_grid_1d(grid_dir, ny, nx):
    try:
        import xmitgcm

        g = xmitgcm.open_mdsdataset(grid_dir, iters=None)
        lon = np.asarray(g["XC"].values)
        lat = np.asarray(g["YC"].values)
        lon = lon[0, :] if lon.ndim == 2 else lon
        lat = lat[:, 0] if lat.ndim == 2 else lat
        area = np.asarray(g["rA"].values) if "rA" in g else None
        if lon.size == nx and lat.size == ny:
            return lon, lat, area
    except Exception as exc:
        print(f"Could not read grid from {grid_dir}: {exc}; using uniform axes")
    return np.linspace(0, 60, nx), np.linspace(15, 75, ny), None


def domain_mean(da, mask, area):
    """Masked spatial mean over the record -> (nt,). Area-weighted if area given."""
    w = mask.astype("float64") if area is None else np.where(mask, area, 0.0)
    flat = np.nan_to_num(da.values.astype("float64"), nan=0.0)
    flat = np.where(mask[None, :, :], flat, 0.0).reshape(da.shape[0], -1)
    return (flat @ w.reshape(-1)) / w.sum()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=paths.BASE_RUN)
    p.add_argument("--scenario", default="run_10yrspinup_tau0.3")
    p.add_argument("--var", default="Eta", choices=sorted(VAR_SETTINGS))
    p.add_argument("--times", default="100,300,930,2880", help="comma list of days")
    p.add_argument("--filter-window", type=int, default=15,
                   help="centred rolling-mean window in days (0 disables)")
    p.add_argument("--area-weight", action="store_true",
                   help="area-weight the domain mean with rA (default: unweighted over wet points)")
    p.add_argument("--grid-dir", default=paths.GRID_DIR)
    p.add_argument("--n-years", type=int, default=10)
    p.add_argument("--out", default="figs/ssh_10yrs.pdf")
    p.add_argument("--preview-png", default=None)
    opts = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, FuncFormatter, LinearLocator
    import cmocean as cmo

    s = VAR_SETTINGS[opts.var]
    times = [int(v) for v in opts.times.split(",")]
    sd = Path(opts.data_dir) / "surface_fields"
    ds_fom = xr.open_dataset(sd / f"fom_surface_{opts.scenario}.nc", engine="h5netcdf")
    ds_rom = xr.open_dataset(sd / f"rom_surface_{opts.scenario}.nc", engine="h5netcdf")
    ny, nx = ds_fom["T"].shape[1:]
    lon, lat, area = load_grid_1d(opts.grid_dir, ny, nx)
    X, Y = np.meshgrid(lon, lat)
    mask = ds_fom["T"].isel(time=0).values != 0  # T = 0 only on land
    if not opts.area_weight:
        area = None
    print(f"domain mean: {'area-weighted (rA)' if area is not None else 'unweighted'}"
          f" over {mask.sum()} wet points", flush=True)

    plt.rcParams.update({
        "font.size": FS_LABEL, "axes.titlesize": FS_TITLE, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.fontsize": FS_TICK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    ncol = len(times)
    panel = (FIG_W - 1.30) / ncol           # leave room for row labels + colorbar
    fig_h = 2 * panel + 2.25
    fig = plt.figure(figsize=(FIG_W, fig_h), layout="constrained")
    # row 2 is an empty spacer so the time series clears the map block
    gs = gridspec.GridSpec(4, ncol, figure=fig, height_ratios=[1, 1, 0.16, 1.15])

    lon_targets = np.array([0, 20, 40, 60])
    lat_targets = np.array([15, 35, 55, 75])
    spatial_axes, im = [], None
    for col, t_val in enumerate(times):
        fom = ds_fom[opts.var].sel(time=t_val, method="nearest").values.astype(float)
        rom = ds_rom[opts.var].sel(time=t_val, method="nearest").values.astype(float)
        fom[~mask] = np.nan
        rom[~mask] = np.nan
        ax_fom = fig.add_subplot(gs[0, col])
        ax_rom = fig.add_subplot(gs[1, col])
        spatial_axes += [ax_fom, ax_rom]
        cmap = getattr(cmo.cm, s["cmap"]) if s["cmap"] in dir(cmo.cm) else s["cmap"]
        kw = dict(cmap=cmap, vmin=s["vmin"], vmax=s["vmax"], rasterized=True)
        ax_fom.pcolormesh(X, Y, fom, **kw)
        im = ax_rom.pcolormesh(X, Y, rom, **kw)
        for row, ax in enumerate((ax_fom, ax_rom)):
            ax.set_aspect("equal")
            ax.set_xticks(lon_targets)
            ax.set_yticks(lat_targets)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if row == 0:
                ax.set_title(f"Day {t_val}", pad=4)
            if col == 0:
                ax.set_ylabel("MITgcm" if row == 0 else "OpInf ROM",
                              fontsize=FS_TITLE, fontweight="bold", labelpad=18)
                ax.set_yticklabels([f"{v:.0f}°N" for v in lat_targets])
            if row == 1:
                ax.set_xticklabels([f"{v:.0f}°E" for v in lon_targets])
    cbar = fig.colorbar(im, ax=spatial_axes, location="right", shrink=0.92, pad=0.015,
                        extend="both")
    cbar.set_label(f"{s['name']} [{s['unit']}]", fontsize=FS_LABEL, labelpad=4)
    cbar.ax.tick_params(labelsize=FS_TICK, width=0.6, length=2.5)
    cbar.ax.yaxis.set_major_locator(LinearLocator(numticks=5))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(s["cfmt"]))
    cbar.outline.set_linewidth(0.6)

    # ---- domain-mean time series ------------------------------------------
    ax = fig.add_subplot(gs[3, :])
    days = np.asarray(ds_fom["time"].values, dtype=float)
    fom_m = domain_mean(ds_fom[opts.var], mask, area)
    rom_m = domain_mean(ds_rom[opts.var], mask, area)
    if opts.filter_window and opts.filter_window > 1:
        k = np.ones(opts.filter_window) / opts.filter_window
        pad = opts.filter_window // 2
        sm = lambda x: np.convolve(np.pad(x, pad, mode="edge"), k, mode="same")[pad:-pad]
        fom_p, rom_p = sm(fom_m), sm(rom_m)
    else:
        fom_p, rom_p = fom_m, rom_m
    # monochrome: ROM thicker and underneath, dashed FOM on top
    ax.plot(days, rom_p, ls="-", lw=1.4, color=COLOR, alpha=0.95, label="OpInf ROM", zorder=2)
    ax.plot(days, fom_p, ls="--", lw=1.0, color=COLOR, alpha=0.95, label="MITgcm", zorder=3)
    idx = [int(np.abs(days - t).argmin()) for t in times]
    ax.scatter(days[idx], fom_p[idx], color="black", marker="o", s=22, zorder=5,
               label="Snapshot times")

    max_days = opts.n_years * 360
    ax.set_xlim(0, max_days)
    ax.set_xticks(np.arange(0, max_days + 360, 360))
    ax.set_xticks(np.arange(0, max_days + 90, 90), minor=True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x / 360)}"))
    ax.set_xlabel("Year", labelpad=2)
    ax.set_ylabel(f"Mean {s['name']} [{s['unit']}]", labelpad=3)
    ax.grid(True, which="major", ls="--", alpha=0.4, lw=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.2, lw=0.4)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none",
              fontsize=6.5, handlelength=1.5, handletextpad=0.5, borderpad=0.25,
              labelspacing=0.3, markerscale=0.7)

    # panel labels (a)/(b), placed from the laid-out axes positions
    fig.canvas.draw()
    # both panel labels share (a)'s x so they line up down the left edge
    x_tag = max(spatial_axes[0].get_position().x0 - 0.085, 0.004)
    for ax_ref, tag, dy in ((spatial_axes[0], "(a)", 0.014), (ax, "(b)", 0.022)):
        fig.text(x_tag, min(ax_ref.get_position().y1 + dy, 0.995), tag,
                 fontsize=FS_TITLE + 1, va="bottom", ha="left")

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
