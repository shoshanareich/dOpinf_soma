#!/usr/bin/env python3
"""Journal-ready surface figure: 5 state variables (rows) x
MITgcm / OpInf ROM / Bias (columns), either at one time index or time-averaged.

Reads the split surface files written by export_surface_fields.py:
    <data-dir>/surface_fields/{fom,rom}_surface_<scenario>.nc
whose arrays are (time, y, x) = (time, latitude, longitude).

The third column is the SIGNED bias. In snapshot mode that is ROM - MITgcm; in
time-mean mode it is <ROM> - <FOM>, the difference of the two time means (which
equals the time mean of the signed difference). It is NOT <|ROM - FOM|>: over 10
years the signed bias is 2.4x (S) to 7.5x (V) smaller than the MAE, because most
of the instantaneous eddy error cancels in time.

Layout note: the colorbars get their own gridspec columns (width_ratios
[1, 1, 0.08, 1, 0.08]) rather than being carved out of the map axes with
make_axes_locatable, which shrank the OpInf and error panels relative to MITgcm.
Each colorbar axes is then repositioned to match the height of the map it
annotates, since equal-aspect maps do not fill their allocated row height.

Sized for JGR: ML/AI (7.5 in max width, 8-9 pt type, embedded TrueType fonts,
rasterized pcolormesh); writes PDF.

Usage:
    python3 src/plot_surface_snapshots.py --time 100 --out figs/surfacesnaps_day100.pdf
    python3 src/plot_surface_snapshots.py --time-mean --out figs/surfacesnaps_10yrmean.pdf
    python3 analysis/plot_surface_snapshots.py --year 10 --out figs/surfacesnaps_yr10mean.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))
import paths  # noqa: E402

VARS = ["T", "S", "Eta", "U", "V"]

# vmin/vmax/err_lim are the snapshot limits; t* are the time-mean limits.
# terr_lim starts from the 99.5th percentile of |<ROM> - <FOM>| over wet points,
# rounded up; Eta, U and V then share +/-0.1 so the three lower rows are directly
# comparable. tvmin/tvmax = None means derive the field limits from the data.
VAR_SETTINGS = {
    "T":   {"name": "Potential\nTemperature", "unit": r"[$^\circ$C]", "cmap": "thermal",
            "vmin": 3.5, "vmax": 22.5, "err_lim": 1.0, "cfmt": "%.1f", "efmt": "%.1f",
            "tvmin": None, "tvmax": None, "terr_lim": 0.5, "diverging": False},
    "S":   {"name": "Salinity", "unit": "[psu]", "cmap": "haline",
            "vmin": 35.2, "vmax": 35.7, "err_lim": 0.03, "cfmt": "%.2f", "efmt": "%.3f",
            "tvmin": None, "tvmax": None, "terr_lim": 0.02, "diverging": False},
    "Eta": {"name": "Sea Surface\nHeight", "unit": "[m]", "cmap": "RdBu_r",
            "vmin": -1.0, "vmax": 1.0, "err_lim": 0.3, "cfmt": "%.1f", "efmt": "%.2f",
            "tvmin": None, "tvmax": None, "terr_lim": 0.1, "diverging": True},
    "U":   {"name": "Zonal\nVelocity", "unit": "[m s$^{-1}$]", "cmap": "RdBu_r",
            "vmin": -0.8, "vmax": 0.8, "err_lim": 0.3, "cfmt": "%.1f", "efmt": "%.2f",
            "tvmin": None, "tvmax": None, "terr_lim": 0.1, "diverging": True},
    "V":   {"name": "Meridional\nVelocity", "unit": "[m s$^{-1}$]", "cmap": "RdBu_r",
            "vmin": -0.8, "vmax": 0.8, "err_lim": 0.4, "cfmt": "%.1f", "efmt": "%.1f",
            "tvmin": None, "tvmax": None, "terr_lim": 0.1, "diverging": True},
}

FIG_W = 7.0
FIG_H = 9.0
FS_TICK = 7.5
FS_LABEL = 8.5
FS_TITLE = 9.5
FS_ROW = 9.0
# the colorbar gridspec columns are wide enough to hold the bar plus its tick and
# unit labels; the bar itself is drawn as a narrow strip at the column's left edge
CBAR_RATIO = 0.34
CBAR_W = 0.012        # figure fraction
CBAR_PAD = 0.012      # figure fraction, gap between map and bar


def load_grid_1d(grid_dir, ny, nx):
    """Return (lon, lat) 1-D coordinate arrays, falling back to a uniform grid."""
    try:
        import xmitgcm

        g = xmitgcm.open_mdsdataset(grid_dir, iters=None)
        lon = np.asarray(g["XC"].values)
        lat = np.asarray(g["YC"].values)
        lon = lon[0, :] if lon.ndim == 2 else lon
        lat = lat[:, 0] if lat.ndim == 2 else lat
        if lon.size == nx and lat.size == ny:
            return lon, lat
        print(f"grid coords sized {lon.size},{lat.size} != {nx},{ny}; using uniform axes")
    except Exception as exc:
        print(f"Could not read grid from {grid_dir}: {exc}; using uniform axes")
    return np.linspace(0, 60, nx), np.linspace(15, 75, ny)


def time_mean(da, t0=0, t1=None, chunk=360):
    """Streaming mean over samples [t0, t1); the surface files are ~0.9 GB per variable."""
    t1 = da.shape[0] if t1 is None else t1
    acc = np.zeros(da.shape[1:], dtype="float64")
    for a in range(t0, t1, chunk):
        acc += da.isel(time=slice(a, min(a + chunk, t1))).values.sum(axis=0, dtype="float64")
    return acc / (t1 - t0)


def round_range(lo, hi, nticks=4):
    """Round a [lo, hi] window outward to a step derived from its span, not from
    the magnitude of its values (S sits near 35 with a range of 0.46)."""
    step = nice((hi - lo) / nticks)
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)


def widen_fmt(fmt, ticks):
    """Keep the tabulated format unless it collapses distinct ticks to the same
    string (the time-mean error limits are much smaller than the snapshot ones)
    or rounds a tick (+/-0.15 has ticks at +/-0.075, which %.2f prints as 0.07)."""
    tol = 1e-9 * max(abs(t) for t in ticks)
    for d in range(int(fmt[2]), 6):
        cand = f"%.{d}f"
        if (len({cand % t for t in ticks}) == len(ticks)
                and all(abs(float(cand % t) - t) <= tol for t in ticks)):
            return cand
    return fmt


def nice(x, up=True):
    """Round to 1-2 significant figures, away from zero."""
    if x <= 0:
        return x
    e = np.floor(np.log10(x))
    m = x / 10**e
    steps = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    m = steps[steps >= m][0] if up else steps[steps <= m][-1]
    return float(m * 10**e)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=paths.BASE_RUN)
    p.add_argument("--scenario", default="run_10yrspinup_tau0.3")
    p.add_argument("--time", type=int, default=100, help="time index (= day for n_days=1)")
    p.add_argument("--time-mean", action="store_true",
                   help="plot the time average over the whole record instead of one snapshot")
    p.add_argument("--year", type=int, default=None,
                   help="plot the time average over year N of the record (1-based: year 10 "
                        "= samples 3240:3600); uses the time-mean colour limits")
    p.add_argument("--spy", type=int, default=360, help="samples per year")
    p.add_argument("--grid-dir", default=paths.GRID_DIR)
    p.add_argument("--out", default="figs/surfacesnaps_day100.pdf")
    p.add_argument("--preview-png", default=None)
    p.add_argument("--err-pctl", type=float, default=None,
                   help="derive each error colour limit from this percentile of |ROM-MITgcm| "
                        "instead of the tabulated value (e.g. 98)")
    p.add_argument("--err-lims", default=None,
                   help="comma list of symmetric error colour limits, one per row "
                        f"({','.join(VARS)}), e.g. 0.6,0.04,0.15,0.15,0.15")
    p.add_argument("--field-pctl", type=float, default=0.5,
                   help="percentile tail trimmed when deriving time-mean field limits")
    p.add_argument("--no-cache", action="store_true")
    opts = p.parse_args()
    err_lims = None
    if opts.err_lims:
        err_lims = dict(zip(VARS, (float(v) for v in opts.err_lims.split(","))))
        if len(err_lims) != len(VARS) or len(opts.err_lims.split(",")) != len(VARS):
            raise SystemExit(f"--err-lims needs {len(VARS)} values ({','.join(VARS)})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, LinearLocator
    import cmocean as cmo

    sd = Path(opts.data_dir) / "surface_fields"
    ds_fom = xr.open_dataset(sd / f"fom_surface_{opts.scenario}.nc", engine="h5netcdf")
    ds_rom = xr.open_dataset(sd / f"rom_surface_{opts.scenario}.nc", engine="h5netcdf")
    ny, nx = ds_fom["T"].shape[1:]
    nt = ds_fom["T"].shape[0]
    lon, lat = load_grid_1d(opts.grid_dir, ny, nx)
    X, Y = np.meshgrid(lon, lat)
    lon_targets = np.array([0, 20, 40, 60])
    lat_targets = np.array([15, 35, 55, 75])
    # land mask from T (T = 0 only on land), so genuine zero crossings of Eta/U/V survive
    mask = ds_fom["T"].isel(time=0).values != 0

    t0, t1 = 0, nt
    tag = ""
    if opts.year is not None:
        t0, t1 = (opts.year - 1) * opts.spy, opts.year * opts.spy
        if t0 < 0 or t1 > nt:
            raise SystemExit(f"--year {opts.year} (samples {t0}:{t1}) outside the {nt}-sample record")
        tag = f"_yr{opts.year}"
        opts.time_mean = True
        print(f"averaging year {opts.year}: samples {t0}:{t1}", flush=True)
    cache = (Path(opts.data_dir) / "analysis" / "computed_fields"
             / f"surface_timemean{tag}_{opts.scenario}.npz")
    fields = {}
    if opts.time_mean:
        if cache.exists() and not opts.no_cache:
            z = np.load(cache)
            fields = {v: (z[f"fom_{v}"], z[f"rom_{v}"]) for v in VARS}
            print(f"loaded time means from {cache}", flush=True)
        else:
            for v in VARS:
                fields[v] = (time_mean(ds_fom[v], t0, t1), time_mean(ds_rom[v], t0, t1))
                print(f"  time-averaged {v} over {t1 - t0} steps", flush=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **{f"fom_{v}": fields[v][0] for v in VARS},
                     **{f"rom_{v}": fields[v][1] for v in VARS})
            print(f"cached time means -> {cache}", flush=True)
    else:
        for v in VARS:
            fields[v] = (ds_fom[v].isel(time=opts.time).values.astype(float),
                         ds_rom[v].isel(time=opts.time).values.astype(float))

    plt.rcParams.update({
        "font.size": FS_LABEL, "axes.titlesize": FS_TITLE, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    # dedicated gridspec columns for the two colorbars, so all three map panels
    # are allocated identical widths
    fig, axes = plt.subplots(len(VARS), 5, figsize=(FIG_W, FIG_H),
                             gridspec_kw=dict(width_ratios=[1, 1, CBAR_RATIO, 1, CBAR_RATIO]))
    fig.subplots_adjust(hspace=0.16, wspace=0.10, left=0.125, right=0.985,
                        top=0.965, bottom=0.045)

    for i, var in enumerate(VARS):
        s = VAR_SETTINGS[var]
        fom, rom = (np.array(a, dtype=float) for a in fields[var])
        fom[~mask] = np.nan
        rom[~mask] = np.nan
        err = rom - fom  # model minus reference

        if opts.time_mean:
            vmin, vmax = s["tvmin"], s["tvmax"]
            if vmin is None or vmax is None:
                q = opts.field_pctl
                if s["diverging"]:
                    lim = nice(np.nanpercentile(np.abs(fom), 100 - q))
                    vmin, vmax = -lim, lim
                else:
                    vmin, vmax = round_range(float(np.nanpercentile(fom, q)),
                                             float(np.nanpercentile(fom, 100 - q)))
        else:
            vmin, vmax = s["vmin"], s["vmax"]

        if err_lims:
            elim = err_lims[var]
        elif opts.err_pctl:
            elim = np.nanpercentile(np.abs(err), opts.err_pctl)
        elif opts.time_mean:
            elim = s["terr_lim"] or nice(np.nanpercentile(np.abs(err), 98))
        else:
            elim = s["err_lim"]

        ax_fom, ax_rom, cax_f, ax_err, cax_e = (axes[i, 0], axes[i, 1], axes[i, 2],
                                                axes[i, 3], axes[i, 4])
        cmap = getattr(cmo.cm, s["cmap"]) if s["cmap"] in dir(cmo.cm) else s["cmap"]
        kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
        ax_fom.pcolormesh(X, Y, fom, **kw)
        im_rom = ax_rom.pcolormesh(X, Y, rom, **kw)
        im_err = ax_err.pcolormesh(X, Y, err, cmap="RdBu_r", vmin=-elim, vmax=elim,
                                   rasterized=True)
        print(f"  {var}: field [{vmin:.4g}, {vmax:.4g}] ({100 * np.nanmean((fom < vmin) | (fom > vmax)):.2f}% "
              f"FOM saturates), error +/-{elim:.4g} "
              f"({100 * np.nanmean(np.abs(err) > elim):.2f}% saturates), "
              f"max|err| {np.nanmax(np.abs(err)):.4g}", flush=True)

        for cax, im, fmt, rng in ((cax_f, im_rom, s["cfmt"], (vmin, vmax)),
                                  (cax_e, im_err, s["efmt"], (-elim, elim))):
            cb = fig.colorbar(im, cax=cax)
            cb.set_label(s["unit"], fontsize=FS_LABEL, labelpad=3)
            cb.ax.tick_params(labelsize=FS_TICK, width=0.6, length=2.5)
            cb.ax.yaxis.set_major_locator(LinearLocator(numticks=5))
            cb.ax.yaxis.set_major_formatter(
                FormatStrFormatter(widen_fmt(fmt, np.linspace(rng[0], rng[1], 5))))
            cb.outline.set_linewidth(0.6)

        for j, ax in enumerate((ax_fom, ax_rom, ax_err)):
            ax.set_aspect("equal")
            ax.set_xlim(lon.min(), lon.max())      # identical limits on all three maps
            ax.set_ylim(lat.min(), lat.max())
            ax.set_xticks(lon_targets)
            ax.set_yticks(lat_targets)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if i == 0:
                ax.set_title(["MITgcm", "OpInf ROM", "Error"][j], pad=5)
            if j == 0:
                # no "Latitude" label: the degN tick labels identify the axis and
                # the row name occupies that margin
                ax.set_yticklabels([f"{v:.0f}°N" for v in lat_targets])
            if i == len(VARS) - 1:
                labs = [f"{v:.0f}°E" for v in lon_targets]
                if j > 0:
                    labs[0] = ""      # would collide with the 60degE of the panel to its left
                ax.set_xticklabels(labs)
                if j == 1:
                    ax.set_xlabel("Longitude", labelpad=3)

    # equal-aspect maps do not fill their allocated row height, so match each
    # colorbar's vertical extent to the map it annotates, and place the row
    # labels from the laid-out geometry
    fig.canvas.draw()
    for i, var in enumerate(VARS):
        pos_map = axes[i, 1].get_position()
        for jc, jm in ((2, 1), (4, 3)):
            pm = axes[i, jm].get_position()
            axes[i, jc].set_position([pm.x1 + CBAR_PAD, pm.y0, CBAR_W, pm.height])
        p0 = axes[i, 0].get_position()
        fig.text(max(p0.x0 - 0.075, 0.004), (p0.y0 + p0.y1) / 2, VAR_SETTINGS[var]["name"],
                 rotation=90, fontsize=FS_ROW, fontweight="bold", va="center", ha="center")
        del pos_map

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
