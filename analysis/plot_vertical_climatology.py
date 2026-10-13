#!/usr/bin/env python3
"""Journal-ready climatology figure for one state variable (T or S):
(a) time-mean vertical section -- MITgcm / OpInf ROM / Bias, and
(b) area-weighted horizontally-averaged time series at three model levels.

Level time series are computed here (area-weighted mean over wet points, same
reconstruction path as eval_rom_forcings) so the skill numbers are exact, and
cached to ``<field-dir>/<field>_level_series.npz``. Panel (a) reads
``<field>_vertical_<label>.nc`` (see export_vertical_fields.py).

The annotated R^2 is the ANOMALY R^2: both series taken about their own time
mean, so it scores variability rather than the offset. State this in the caption
and quote the bias separately -- the printed table has bias, RMSE and plain NSE.

CAVEAT to state in the caption: below the thermocline the variance of the series
is almost entirely the decadal trend (93% at 700 m, 99.9% at 2000 m for T at
tau0.3), so R^2 there measures agreement of the trend. At 700 m the ROM also
tracks the detrended residual (detrended R^2 = 0.865 vs anomaly R^2 = 0.870),
but at 2000 m it does not (detrended R^2 < 0) -- though the detrended signal
there is only ~0.6 mK, so there is nothing physically meaningful left to track.

Sized for JGR: ML/AI -- 7.5 in (190 mm) double-column width, 8-9 pt type,
embedded TrueType fonts, rasterized pcolormesh. Writes PDF (+ PNG preview).

Usage:
    python3 src/plot_vertical_climatology.py --var S \
        --config <run>/config.json --data-dir <run>
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))  # core ROM modules
import paths  # noqa: E402
import eval_rom_forcings as E  # noqa: E402
from config_utils import infer_tau_from_dir  # noqa: E402
from export_vertical_fields import resolve_args  # noqa: E402

# Per-variable plotting spec. ``cmap`` is resolved from cmocean at call time.
VAR_SPECS = {
    "S": {
        "field": "salinity", "cmap": "haline", "vmin": 35.0, "vmax": 35.65,
        "cbar": "Salinity [psu]", "bias_cbar": "Salinity Error [psu]",
        "err_lim": 0.02, "ylabel": "S [psu]", "yfmt": "%.3f",
        "cfmt": "%.2f", "bias_cfmt": "%.2f", "yticks": {},
    },
    "T": {
        "field": "temperature", "cmap": "thermal", "vmin": 0.0, "vmax": 22.0,
        "cbar": r"Temperature [$^\circ$C]", "bias_cbar": r"Temperature Error [$^\circ$C]",
        "err_lim": 0.4, "ylabel": r"T [$^\circ$C]", "yfmt": "%.2f",
        "cfmt": "%.0f", "bias_cfmt": "%.1f",
        # explicit ticks per model level: wide enough that the legend and the R^2
        # annotation clear the curves
        "yticks": {0: [12.5, 14.5, 16.5], 18: [7.4, 7.6, 7.8, 8.0],
                   26: [2.25, 2.30, 2.35, 2.40]},
    },
}

# --- JGR: ML/AI figure geometry (inches) and type sizes (pt) ------------------
FIG_W = 7.5          # AGU double-column max width (190 mm)
FIG_H = 7.2
FS_TICK = 7.5
FS_LABEL = 8.5
FS_TITLE = 9.0
FS_ANNOT = 8.0
FS_PANEL = 10.0      # (a) / (b)
LW = 1.0
# Zonal-mean panels get their own colour limits: zonal averaging cancels ~37% of the
# bias, so |bias| peaks near 0.25 degC vs >0.5 for a single longitude slice. Keeping
# the slice limits here would leave the zonal bias panel almost blank.
ZONAL_LIMITS = {"T": {"vmin": 0.0, "vmax": 21.0, "err_lim": 0.3},
                "S": {"vmin": 35.0, "vmax": 35.6, "err_lim": 0.01}}
DEPTH_TICKS = np.arange(0.0, -3501.0, -500.0)   # panel (a): label the 3500 m floor
DEPTH_LIM = (-3500.0, 0.0)


def h_mean(field: np.ndarray, mask: np.ndarray, area: np.ndarray | None) -> np.ndarray:
    """Area-weighted horizontal mean over wet points -> (nt,).

    ``field`` is (nt, ny, nx) with the same (j, i) ordering as rA, so no
    transpose is needed here (unlike the saved surface_fields/*.nc).
    """
    w = (mask.astype("float64") if area is None else np.where(mask, area, 0.0))
    flat = np.nan_to_num(field, nan=0.0).reshape(field.shape[0], -1)
    return (flat @ w.reshape(-1)) / w.sum()


def fmt_for(ticks, tol_frac=0.02):
    """Fewest decimals that keeps every tick label distinct and within
    tol_frac of a tick step of its true value (so an endpoint like 35.65 is not
    shown as 35.7)."""
    ticks = np.asarray(ticks, float)
    step = abs(ticks[1] - ticks[0]) if ticks.size > 1 else (abs(ticks[0]) or 1.0)
    for d in range(5):
        f = f"%.{d}f"
        vals = np.array([float(f % t) for t in ticks])
        if np.unique(vals).size == ticks.size and np.abs(vals - ticks).max() <= tol_frac * step:
            return f
    return "%.4f"


def endpoint_ticks(lo, hi, candidates=(4, 3, 2)):
    """Ticks including both endpoints, choosing the count whose labels are
    cleanest (fewest decimals; ties go to the larger count). For 0-22 this picks
    0 / 11 / 22 rather than 0.0 / 5.5 / 11.0 / 16.5 / 22.0."""
    best = None
    for n in candidates:
        t = np.linspace(lo, hi, n + 1)
        f = fmt_for(t)
        d = int(f[2])
        if best is None or d < best[0]:
            best = (d, t, f)
    return best[1], best[2]


def nice_yticks(lo_data, hi_data, pad_bot, pad_top, min_ticks=3, max_intervals=5):
    """Pick a 1-2-5 ladder step and y limits that are whole multiples of it.

    Ticks then land exactly on the top and bottom of the axes, the data stays
    strictly inside them, and there is room reserved above (legend) and below
    (R^2 annotation). Using the same ladder in every panel keeps the tick steps
    visually consistent across depths (e.g. 0.5 / 0.1 / 0.02).
    """
    dspan = hi_data - lo_data
    if not np.isfinite(dspan) or dspan <= 0:
        dspan = abs(hi_data) or 1.0
    span = dspan / max(1.0 - pad_bot - pad_top, 0.2)
    lo_t, hi_t = lo_data - pad_bot * span, hi_data + pad_top * span
    ladder = sorted(m * 10.0**e for e in range(-8, 5) for m in (1, 2, 5))
    for step in ladder:
        lo = np.floor(lo_t / step) * step
        hi = np.ceil(hi_t / step) * step
        n = int(round((hi - lo) / step))
        if n <= max_intervals and n + 1 >= min_ticks:
            return lo, hi, np.linspace(lo, hi, n + 1)
    return lo_t, hi_t, np.linspace(lo_t, hi_t, min_ticks)


def skill(fom: np.ndarray, rom: np.ndarray) -> dict:
    """Error decomposition for a scalar time series."""
    m = np.isfinite(fom) & np.isfinite(rom)
    fom, rom = np.asarray(fom, float)[m], np.asarray(rom, float)[m]
    err = rom - fom
    bias = err.mean()
    sst = ((fom - fom.mean()) ** 2).sum()
    t = np.arange(fom.size)
    trend = np.polyval(np.polyfit(t, fom, 1), t)
    fa, ra = fom - fom.mean(), rom - rom.mean()
    return {
        "std_fom": fom.std(),
        "std_fom_detrended": (fom - trend).std(),
        "drift_fom": trend[-1] - trend[0],
        "bias": bias,
        "rmse": np.sqrt((err**2).mean()),
        "rmse_debiased": (err - bias).std(),
        "err_t0": err[0],
        "corr": np.corrcoef(fom, rom)[0, 1],
        "r2_nse": 1 - (err**2).sum() / sst,
        "r2_debiased": 1 - ((err - bias) ** 2).sum() / sst,
        "r2_anom": 1 - ((ra - fa) ** 2).sum() / (fa**2).sum(),
    }


def load_fom_level(snapshot_dir, var, args, k) -> np.ndarray:
    """FOM level as (nt, ny, nx); space ordered (j, i) to match the ROM."""
    ds = xr.open_dataset(
        E.ensure_slash(snapshot_dir) + "states_20yrs.nc",
        engine="h5netcdf", decode_timedelta=False, decode_times=False, chunks={},
    )
    tsel = slice(0, 360 * (args.n_year_train + args.n_year_predict), args.n_days)
    arr = ds[var].isel(k=k, time=tsel).stack(space=("j", "i")).T.values
    ds.close()
    return arr.T.reshape(arr.shape[1], args.ny, args.nx)


def build_series(args, var, taus, levels, cache_path, recompute):
    """Return {(tau, k): {years, fom, rom}}, caching to npz."""
    cache = {}
    if cache_path.exists() and not recompute:
        with np.load(cache_path) as z:
            for key in z.files:
                tau_s, k_s, which = key.split("|")
                cache.setdefault((float(tau_s), int(k_s)), {})[which] = z[key]
    training_dirs = E.split_csv(args.training_dirs)
    test_dirs = E.split_csv(args.test_dirs) if args.test_dirs else []
    dir_by_tau, rom_by_tau = {}, {}
    for dirs, kind in ((training_dirs, "train"), (test_dirs, "test")):
        for d in dirs:
            tau = float(infer_tau_from_dir(d))
            dir_by_tau.setdefault(tau, d)
            rom_by_tau.setdefault(tau, f"Q_ROM_{kind}_{E.scenario_name_from_dir(d)}.npy")

    area = E.load_cell_area(args.grid_dir, args.ny, args.nx)
    print(f"area weighting: {'rA' if area is not None else 'UNWEIGHTED (rA unavailable)'}", flush=True)
    tr = np.load(Path(args.data_dir) / "Tr.npy")
    n_steps = (args.n_year_train + args.n_year_predict) * int(360 / args.n_days)
    proj_cache: dict = {}
    missing = [(tau, k) for k in levels for tau in taus
               if not {"years", "fom", "rom"} <= set(cache.get((tau, k), {}))]
    for tau, k in missing:
        q_rom = np.load(Path(args.data_dir) / rom_by_tau[tau])[:, :n_steps]
        _, rom = E.transform_and_project_k_forcings(
            training_dirs, var, args.n_year_train, args.n_days, args.center_opt,
            args.preproc_dir, tr, q_rom, args.nx, args.ny, k=k, projection_cache=proj_cache,
        )  # (nt, ny, nx)
        fom = load_fom_level(dir_by_tau[tau], var, args, k)
        nt = min(rom.shape[0], fom.shape[0])
        mask = np.isfinite(fom[0]) & (fom[0] != 0)
        cache[(tau, k)] = {
            "years": np.arange(nt) * args.n_days / 360.0,
            "fom": h_mean(fom[:nt], mask, area),
            "rom": h_mean(rom[:nt], mask, area),
        }
        del rom, fom
        print(f"  computed {var} tau={tau}, k={k}", flush=True)
    if missing:
        np.savez(cache_path, **{f"{tau}|{k}|{w}": v
                                for (tau, k), d in cache.items() for w, v in d.items()})
        print(f"cached series -> {cache_path}", flush=True)
    return cache


def zonal_mean_section(data_dir, scenario, var, field, cache_path, grid_dir, recompute=False):
    """Time-mean, zonally-averaged section (k, j) for FOM and ROM, plus the
    across-longitude spread of the time-mean bias.

    The zonal average at each (depth, latitude) is weighted by the wet fraction
    hFacC, so partial bottom cells and the basin outline are handled. Reads the
    3D fields from export_3d_fields.py; FOM T and S are not duplicated there, so
    they come from the original states_20yrs.nc.

    Returns dict with zF, zR, zB (= zR - zF), spread, Z, YC.
    """
    import netCDF4

    if cache_path.exists() and not recompute:
        z = np.load(cache_path)
        print(f"loaded zonal-mean section from {cache_path.name}", flush=True)
        return {k: z[k] for k in z.files}

    import xmitgcm

    g = xmitgcm.open_mdsdataset(grid_dir, iters=None)
    hf = np.asarray(g["hFacC"].values, dtype="float64")
    Z = np.asarray(g["Z"].values)
    YC = np.asarray(g["YC"].values)
    YC = YC[:, 0] if YC.ndim == 2 else YC
    den = hf.sum(axis=2)
    zm = lambda a: np.where(den > 0, (a * hf).sum(axis=2) / np.maximum(den, 1e-12), np.nan)

    rom_p = Path(data_dir) / "fields_3d" / f"rom_3d_{scenario}.nc"
    fom_p = Path(data_dir) / "fields_3d" / f"fom_3d_{scenario}.nc"
    dr = netCDF4.Dataset(rom_p)
    df = netCDF4.Dataset(fom_p) if fom_p.exists() else None
    tau = scenario.split("tau")[-1]
    src = xr.open_dataset(f"{paths.FOM_PREFIX}{tau}/states_20yrs.nc",
                          engine="h5netcdf", decode_timedelta=False, decode_times=False)
    nt = len(dr.dimensions["time"])
    accR = np.zeros(hf.shape); accF = np.zeros(hf.shape)
    t0 = time.time()
    for t in range(nt):
        accR += np.asarray(dr[var][t], dtype="float64")
        accF += (np.asarray(df[var][t], dtype="float64") if (df is not None and var in df.variables)
                 else src[var].isel(time=t).values.astype("float64"))
    dr.close(); src.close()
    if df is not None:
        df.close()
    mR, mF = accR / nt, accF / nt
    zF, zR = zm(mF), zm(mR)
    zB = zR - zF
    bias3 = np.where(hf > 0, mR - mF, np.nan)
    spread = np.sqrt(np.maximum(zm((np.nan_to_num(bias3) - zB[:, :, None]) ** 2), 0.0))
    print(f"zonal mean over {nt} steps in {(time.time() - t0) / 60:.1f} min; "
          f"RMS bias: zonal {np.sqrt(np.nanmean(zB**2)):.4f}, "
          f"3D pointwise {np.sqrt(np.nanmean(bias3[hf > 0]**2)):.4f}; "
          f"|zonal bias| > across-longitude spread in "
          f"{100 * np.nanmean(np.abs(zB) > spread):.0f}% of cells", flush=True)
    out = dict(zF=zF, zR=zR, zB=zB, spread=spread, Z=Z, YC=YC)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **out)
    print(f"cached -> {cache_path}", flush=True)
    return out


def make_figure(section, spec, series, levels, level_labels, series_tau, out_base,
                preview_png=None, zonal=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import cmocean as cmo

    plt.rcParams.update({
        "font.size": FS_LABEL, "axes.titlesize": FS_TITLE, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.fontsize": FS_ANNOT, "legend.frameon": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # AGU: embed TrueType fonts
    })
    field = spec["field"]
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs_master = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.35], hspace=0.30)

    # ---- (a) time-mean vertical section --------------------------------------
    # wide wspace: each of the 2 colorbars needs room for its label between panels
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_master[0], wspace=0.62)
    ax_fom, ax_rom, ax_err = [fig.add_subplot(gs_top[i]) for i in range(3)]
    if zonal is not None:
        X, Y = np.meshgrid(zonal["YC"], zonal["Z"])
        fom_data, rom_data = zonal["zF"], zonal["zR"]
        bias_data = zonal["zB"]
    else:
        X, Y = np.meshgrid(section["j"].values, section["k"].values)
        fom_data = section[f"{field}_fom"].mean(dim="time").values
        rom_data = section[f"{field}_rom"].mean(dim="time").values
        bias_data = rom_data - fom_data  # model minus reference

    cmap = getattr(cmo.cm, spec["cmap"])
    kw = dict(cmap=cmap, vmin=spec["vmin"], vmax=spec["vmax"], rasterized=True)
    for nm, arr in (("MITgcm", fom_data), ("ROM", rom_data)):
        a = arr[np.isfinite(arr) & (arr != 0)]
        print(f"    {nm} section: [{a.min():.3f}, {a.max():.3f}], "
              f"{100 * np.mean(a < spec['vmin']):.2f}% below vmin, "
              f"{100 * np.mean(a > spec['vmax']):.2f}% above vmax", flush=True)
    ax_fom.pcolormesh(X, Y, fom_data, **kw)
    im_rom = ax_rom.pcolormesh(X, Y, rom_data, **kw)
    cax_rom = make_axes_locatable(ax_rom).append_axes("right", size="5%", pad=0.06)
    cbar_rom = fig.colorbar(im_rom, cax=cax_rom)
    cbar_rom.set_label(spec["cbar"], fontsize=FS_LABEL, labelpad=3)
    cbar_rom.ax.tick_params(labelsize=FS_TICK, width=0.6)
    t_rom, f_rom = endpoint_ticks(spec["vmin"], spec["vmax"])
    cbar_rom.set_ticks(t_rom)                       # includes vmin and vmax
    cbar_rom.ax.yaxis.set_major_formatter(FormatStrFormatter(f_rom))
    cbar_rom.outline.set_linewidth(0.6)

    lim = spec["err_lim"]
    im_err = ax_err.pcolormesh(X, Y, bias_data, cmap="RdBu_r", vmin=-lim, vmax=lim, rasterized=True)
    cax_err = make_axes_locatable(ax_err).append_axes("right", size="5%", pad=0.06)
    cbar_err = fig.colorbar(im_err, cax=cax_err)
    cbar_err.set_label(spec["bias_cbar"], fontsize=FS_LABEL, labelpad=3)
    cbar_err.ax.tick_params(labelsize=FS_TICK, width=0.6)
    t_err, f_err = endpoint_ticks(-lim, lim, candidates=(4,))
    cbar_err.set_ticks(t_err)                       # includes -lim and +lim
    cbar_err.ax.yaxis.set_major_formatter(FormatStrFormatter(f_err))
    cbar_err.outline.set_linewidth(0.6)

    if zonal is not None:
        # Diagnostic only, not drawn: the fraction of cells where the zonal-mean bias
        # exceeds its across-longitude spread. Not a useful significance marker here --
        # in the quiescent abyss both quantities are ~1e-3 degC, so the ratio passes 1
        # where the bias is physically negligible, while the strong (and most
        # interesting) banded features have the largest longitude variability.
        ratio = np.abs(bias_data) / np.where(zonal["spread"] > 0, zonal["spread"], np.inf)
        print(f"    zonal-mean bias exceeds its across-longitude spread in "
              f"{100 * np.nanmean(np.nan_to_num(ratio) > 1.0):.0f}% of cells "
              f"(diagnostic only)", flush=True)

    lat_targets = np.array([15, 35, 55, 75])
    titles = (["MITgcm", "OpInf ROM", "Error"] if zonal is None else
              ["MITgcm", "OpInf ROM", "Error"])
    for i, (ax, title) in enumerate(zip((ax_fom, ax_rom, ax_err), titles)):
        ax.set_xticks(lat_targets)
        ax.set_xticklabels([f"{lat:.0f}°N" for lat in lat_targets])
        ax.set_xlabel("Latitude", labelpad=3)
        ax.set_title(title, pad=5)
        for depth in [0, -700, -2000]:
            ax.axhline(y=depth, color="#B0B0B0", ls="--", lw=0.7, alpha=0.85, zorder=3)
        ax.set_ylim(*DEPTH_LIM)
        ax.set_yticks(DEPTH_TICKS)
        if i == 0:
            ax.set_ylabel("Depth [m]", labelpad=3)
            ax.set_yticklabels([f"{d:.0f}" for d in DEPTH_TICKS])
        else:
            ax.set_yticklabels([])
    ax_fom.text(-0.34, 1.10, "(a)", transform=ax_fom.transAxes,
                fontsize=FS_PANEL, va="top", ha="left")

    # ---- (b) horizontally-averaged time series -------------------------------
    gs_bottom = gridspec.GridSpecFromSubplotSpec(len(levels), 1, subplot_spec=gs_master[1], hspace=0.45)
    axes_bottom = [fig.add_subplot(gs_bottom[i]) for i in range(len(levels))]
    c = "#0072B2"
    for idx, (ax, k) in enumerate(zip(axes_bottom, levels)):
        s = series[(series_tau, k)]
        fom_val, rom_val = np.asarray(s["fom"]), np.asarray(s["rom"])
        m = skill(fom_val, rom_val)
        ax.plot(s["years"], fom_val, ls=":", lw=LW, color=c, label="MITgcm")
        ax.plot(s["years"], rom_val, ls="-", lw=LW, color=c, label="OpInf ROM")
        ax.set_title(f"z = {level_labels[k]}", pad=4)
        # room above for the legend (first panel only) and below for the R^2 label;
        # limits land on whole tick steps so ticks mark the top and bottom and no
        # data falls outside them
        lo_d = min(fom_val.min(), rom_val.min())
        hi_d = max(fom_val.max(), rom_val.max())
        explicit = spec.get("yticks", {}).get(k)
        if explicit:
            yt = np.asarray(explicit, dtype=float)
            lo, hi = float(yt[0]), float(yt[-1])
            if lo_d < lo or hi_d > hi:
                print(f"    WARNING z={level_labels[k]}: data [{lo_d:.4g}, {hi_d:.4g}] "
                      f"falls outside the requested ticks [{lo:.4g}, {hi:.4g}]", flush=True)
        else:
            # max_intervals=2 with min_ticks=3 forces n == 2, i.e. exactly
            # three ticks: bottom, middle, top of the axes
            lo, hi, yt = nice_yticks(lo_d, hi_d, pad_bot=0.16,
                                     pad_top=0.30 if idx == 0 else 0.10,
                                     max_intervals=2)
        ax.set_ylim(lo, hi)
        ax.set_yticks(yt)
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt_for(yt)))
        print(f"    z={level_labels[k]}: data [{lo_d:.4g}, {hi_d:.4g}] -> ylim "
              f"[{lo:.4g}, {hi:.4g}], step {yt[1] - yt[0]:.4g}, {yt.size} ticks, "
              f"data fills {100 * (hi_d - lo_d) / (hi - lo):.0f}% of the axis"
              f"{'  (explicit)' if explicit else ''}", flush=True)
        # anomaly R^2: both series about their own time mean
        ax.text(0.99, 0.045, f"$R^2 = {m['r2_anom']:.3f}$", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=FS_ANNOT)
        ax.set_ylabel(spec["ylabel"], labelpad=3)
        ax.grid(alpha=0.3, lw=0.5)
        n_end = int(np.ceil(s["years"][-1]))         # tick at year 10, not 9
        ax.set_xlim(0, n_end)
        ax.set_xticks(np.arange(0, n_end + 1, 1))
        if idx == len(levels) - 1:
            ax.set_xlabel("Year", labelpad=3)
        else:
            ax.set_xticklabels([])
    axes_bottom[0].legend(ncol=2, loc="upper right", framealpha=0.9, borderpad=0.25,
                          handlelength=1.3, handletextpad=0.5, columnspacing=0.9,
                          labelspacing=0.3, fontsize=FS_ANNOT - 1.0)
    axes_bottom[0].text(-0.085, 1.32, "(b)", transform=axes_bottom[0].transAxes,
                        fontsize=FS_PANEL, va="top", ha="left")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    # not with_suffix(): scenario labels like "tau0.3-test" contain a dot
    out = out_base.parent / f"{out_base.name}.pdf"
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}", flush=True)
    if preview_png is not None:
        fig.savefig(preview_png, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote preview {preview_png}", flush=True)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--var", default="S", choices=sorted(VAR_SPECS), help="state variable")
    p.add_argument("--config", required=True)
    p.add_argument("--root-dir", default=paths.RESULTS_ROOT)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--field-dir", default=None,
                   help="default <data-dir>/analysis/computed_fields (where the section files live)")
    p.add_argument("--levels", default="0,18,26")
    p.add_argument("--level-labels", default="0m,700m,2000m",
                   help="nominal depth labels for the panel (b) titles, one per --levels entry")
    p.add_argument("--taus", default="0.2,0.3,0.4", help="scenarios to compute series for")
    p.add_argument("--series-tau", type=float, default=0.3, help="scenario plotted in panel (b)")
    p.add_argument("--section-label", default=None,
                   help="scenario label of the section file for panel (a); default matches --series-tau")
    p.add_argument("--vertical-i", type=int, default=124)
    p.add_argument("--vmin", type=float, default=None, help="override the field colour minimum")
    p.add_argument("--vmax", type=float, default=None, help="override the field colour maximum")
    p.add_argument("--err-lim", type=float, default=None,
                   help="override the symmetric bias colour limit")
    p.add_argument("--section-mode", default="slice", choices=("slice", "zonal"),
                   help="panel (a): single longitude slice (default) or zonal mean "
                        "computed from fields_3d/*_3d_<scenario>.nc")
    p.add_argument("--out-dir", default=None,
                   help="where to write the figure; default is <field-dir>")
    p.add_argument("--yticks", default=None,
                   help="override panel (b) ticks, e.g. \"0:12.5,14.5,16.5|18:7.4,7.6,7.8,8.0\"")
    p.add_argument("--out-base", default=None,
                   help="output path stem (no extension); default is "
                        "<field-dir>/<field>_climatology_timeseries_<label>")
    p.add_argument("--recompute", action="store_true", help="ignore the cached series npz")
    p.add_argument("--preview-png", default=None,
                   help="also write a raster preview here (the deliverable is the PDF)")
    opts = p.parse_args()

    spec = dict(VAR_SPECS[opts.var])
    if opts.section_mode == "zonal":
        spec.update(ZONAL_LIMITS.get(opts.var, {}))
    for key, val in (("vmin", opts.vmin), ("vmax", opts.vmax), ("err_lim", opts.err_lim)):
        if val is not None:
            spec[key] = val
    if opts.yticks:
        spec["yticks"] = {int(part.split(":")[0]): [float(v) for v in part.split(":")[1].split(",")]
                          for part in opts.yticks.split("|") if part.strip()}
    field = spec["field"]
    args = resolve_args(opts.config, opts.root_dir, opts.data_dir, opts.vertical_i)
    levels = [int(v) for v in opts.levels.split(",")]
    taus = [float(v) for v in opts.taus.split(",")]
    names = [v.strip() for v in opts.level_labels.split(",")]
    if len(names) != len(levels):
        p.error(f"--level-labels needs {len(levels)} entries to match --levels, got {len(names)}")
    field_dir = Path(opts.field_dir or (Path(args.data_dir) / "analysis" / "computed_fields"))

    series = build_series(args, opts.var, taus, levels,
                          field_dir / f"{field}_level_series.npz", opts.recompute)

    depths = E.load_depth_values(args.grid_dir, args.nz)
    print(f"\n=== area-weighted horizontal-mean {field}: error decomposition ===")
    hdr = ("tau", "k", "z[m]", "std_FOM", "detr", "drift", "bias", "RMSE", "r",
           "R2_NSE", "R2_debias", "R2_anom")
    print("{:>5} {:>3} {:>8} {:>9} {:>9} {:>10} {:>10} {:>9} {:>7} {:>8} {:>9} {:>8}".format(*hdr))
    for k in levels:
        for tau in taus:
            m = skill(np.asarray(series[(tau, k)]["fom"]), np.asarray(series[(tau, k)]["rom"]))
            print("{:>5} {:>3} {:>8.1f} {:>9.2e} {:>9.2e} {:>+10.2e} {:>+10.2e} {:>9.2e} "
                  "{:>7.3f} {:>8.3f} {:>9.3f} {:>8.3f}".format(
                      tau, k, depths[k], m["std_fom"], m["std_fom_detrended"], m["drift_fom"],
                      m["bias"], m["rmse"], m["corr"], m["r2_nse"], m["r2_debiased"], m["r2_anom"]))

    train_taus = [float(infer_tau_from_dir(d)) for d in E.split_csv(args.training_dirs)]
    label = opts.section_label or E.safe_stem(
        f"tau{opts.series_tau:g}" + ("" if opts.series_tau in train_taus else "-test"))
    section_path = field_dir / f"{field}_vertical_{label}.nc"
    zonal, section = None, None
    if opts.section_mode == "zonal":
        scenario = f"run_10yrspinup_tau{opts.series_tau:g}"
        zonal = zonal_mean_section(args.data_dir, scenario, opts.var, field,
                                   field_dir / f"{field}_zonalmean_{label}.npz",
                                   args.grid_dir, opts.recompute)
        print(f"\npanel (a): zonal mean over the whole basin ({scenario})")
        stem = f"{field}_climatology_zonalmean_{label}"
    else:
        section = xr.open_dataset(section_path, engine="h5netcdf")
        print(f"\npanel (a) section: {section_path.name}  "
              f"(i={section.attrs.get('vertical_i')})")
        stem = f"{field}_climatology_timeseries_{label}"
    if opts.out_base:
        out_base = Path(opts.out_base)
    else:
        out_base = (Path(opts.out_dir) if opts.out_dir else field_dir) / stem
    make_figure(section, spec, series, levels, dict(zip(levels, names)), opts.series_tau,
                out_base, zonal=zonal,
                preview_png=Path(opts.preview_png) if opts.preview_png else None)


if __name__ == "__main__":
    main()
