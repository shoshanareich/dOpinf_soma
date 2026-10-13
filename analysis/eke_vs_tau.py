#!/usr/bin/env python3
"""Surface eddy kinetic energy versus wind-stress amplitude: does the emulator get the
SENSITIVITY right even where it misses the amplitude?

EKE here is the area-weighted mean of 0.5 (u'^2 + v'^2) over wet surface points,
averaged over a post-adjustment window (default years 3-10, the same window the Psi_bt
figures use).  Primes are deviations from each model's OWN window time mean, so a
mean-state bias cannot leak into EKE; ``--anomaly deseason`` instead removes each
model's own day-of-year climatology, leaving eddy + interannual energy only.

Two panels, because the two statements are different and only one of them is the
paper's claim:

  (a) absolute EKE vs tau_max.  The ROM is badly low here -- roughly a factor of ten,
      the square of the ~0.3 velocity sigma ratio -- and the figure should say so.
  (b) EKE normalized by each model's own value at the reference amplitude.  This is
      the actual sensitivity claim: does d(EKE)/d(tau) have the right shape once the
      amplitude deficit is divided out?  A black-box emulator that regresses the
      forcing response from data is reported to flatten here; OpInf's forcing pathway
      is imposed, so this is the comparison that matters.

Training amplitudes are drawn as circles, held-out amplitudes as stars.  A rollout that
goes non-finite or unphysical inside the window is reported as a divergence and broken
out of the ROM curve rather than silently averaged (tau0.1 diverges at year 3.8).

Reads <data-dir>/surface_fields/{fom,rom,proj}_surface_<scenario>.nc.

Usage:
    python3 src/eke_vs_tau.py --taus 0.2,0.3,0.4,0.5 --train-taus 0.2,0.4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))  # core ROM modules
import paths  # noqa: E402
from surface_error_metrics import orient_area  # noqa: E402

FS_TICK, FS_LABEL = 7.5, 8.5
C_MAIN, C_PROJ = "#0072B2", "#777777"
KINDS = ["fom", "rom", "proj"]
STYLE = {"fom": dict(color=C_MAIN, ls=":", label="MITgcm"),
         "rom": dict(color=C_MAIN, ls="-", label="OpInf ROM"),
         "proj": dict(color=C_PROJ, ls="-", label="POD projection")}


def eke_series(ds, sl, w, spy, chunk=180):
    """Area-weighted mean surface EKE per snapshot over the window `sl`.

    Returns (eke_timemean, eke_deseason): anomalies about the window time mean, and
    about the window day-of-year climatology.  Both are m^2 s^-2.
    """
    n = sl.stop - sl.start
    acc = {v: np.zeros((spy,) + ds[v].shape[1:]) for v in ("U", "V")}
    for i0 in range(sl.start, sl.stop, chunk):
        for v in ("U", "V"):
            blk = ds[v].isel(time=slice(i0, min(i0 + chunk, sl.stop))).values.astype("float64")
            for j in range(blk.shape[0]):
                acc[v][(i0 + j) % spy] += blk[j]
    clim = {v: acc[v] / (n / spy) for v in acc}
    tmean = {v: clim[v].mean(axis=0) for v in clim}

    out = {"timemean": np.empty(n), "deseason": np.empty(n)}
    for i0 in range(sl.start, sl.stop, chunk):
        i1 = min(i0 + chunk, sl.stop)
        doy = np.arange(i0, i1) % spy
        blk = {v: ds[v].isel(time=slice(i0, i1)).values.astype("float64")
               for v in ("U", "V")}
        for key, ref in (("timemean", tmean), ("deseason", None)):
            e = np.zeros(i1 - i0)
            for v in ("U", "V"):
                a = blk[v] - (ref[v] if ref is not None else clim[v][doy])
                e += 0.5 * np.einsum("tij,ij->t", a ** 2, w) / w.sum()
            out[key][i0 - sl.start:i1 - sl.start] = e
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=paths.BASE_RUN)
    p.add_argument("--grid-dir", default=paths.GRID_DIR)
    p.add_argument("--taus", default="0.2,0.3,0.4,0.5")
    p.add_argument("--train-taus", default="0.2,0.4")
    p.add_argument("--scenario-prefix", default="run_10yrspinup_tau")
    p.add_argument("--year0", type=int, default=3)
    p.add_argument("--nyears", type=int, default=7)
    p.add_argument("--spy", type=int, default=360)
    p.add_argument("--anomaly", choices=("timemean", "deseason"), default="timemean")
    p.add_argument("--ref-tau", default=None, help="normalizing amplitude for panel (b); "
                                                   "default the smallest plotted tau")
    p.add_argument("--diverge-eke", type=float, default=10.0,
                   help="mean EKE above this (m2 s-2) counts the rollout as diverged")
    p.add_argument("--out", default="figs/variability/eke_vs_tau.pdf")
    p.add_argument("--preview-png", default=None)
    p.add_argument("--from-cache", action="store_true")
    opts = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    taus = [t.strip() for t in opts.taus.split(",") if t.strip()]
    train = {t.strip() for t in opts.train_taus.split(",")}
    sd = Path(opts.data_dir) / "surface_fields"
    adir = Path(opts.data_dir) / "analysis" / "variability"
    adir.mkdir(parents=True, exist_ok=True)
    cache = adir / "eke_vs_tau.npz"

    series, diverged = {}, {}
    if opts.from_cache:
        z = np.load(cache, allow_pickle=True)
        series = {k: z[k] for k in z.files if "|" in k}
        diverged = {k.split("!")[1]: True for k in z.files if k.startswith("diverged!")}
        print(f"replotting from {cache}", flush=True)
    else:
        sl = slice(opts.year0 * opts.spy, (opts.year0 + opts.nyears) * opts.spy)
        for tau in taus:
            scen = f"{opts.scenario_prefix}{tau}"
            files = {k: sd / f"{k}_surface_{scen}.nc" for k in KINDS}
            kinds = [k for k in KINDS if files[k].exists()]
            if "fom" not in kinds:
                print(f"tau {tau}: no FOM surface file; skipped", flush=True)
                continue
            ds = {k: xr.open_dataset(files[k], engine="h5netcdf") for k in kinds}
            mask = ds["fom"]["T"].isel(time=0).values != 0
            area = orient_area(opts.grid_dir, mask)
            w = mask.astype("float64") if area is None else np.where(mask, area, 0.0)
            for k in kinds:
                if ds[k]["U"].shape[0] < sl.stop:
                    print(f"  tau {tau} {k}: record shorter than the window; skipped",
                          flush=True)
                    continue
                e = eke_series(ds[k], sl, w, opts.spy)
                bad = (not np.isfinite(e[opts.anomaly]).all()
                       or np.nanmax(e[opts.anomaly]) > opts.diverge_eke)
                if bad and k != "fom":
                    diverged[f"{tau}|{k}"] = True
                    print(f"  tau {tau} {k}: DIVERGED inside the window "
                          f"(max EKE {np.nanmax(e[opts.anomaly]):.3g})", flush=True)
                for key, v in e.items():
                    series[f"{tau}|{k}|{key}"] = v.astype("float32")
            for d in ds.values():
                d.close()
            print(f"tau {tau}: done", flush=True)
        np.savez_compressed(cache, **series,
                            **{f"diverged!{k}": np.array(True) for k in diverged})
        print(f"wrote {cache}", flush=True)

    # ------------------------------------------------------------------ table
    key = opts.anomaly
    val = {}
    for tau in taus:
        for k in KINDS:
            s = series.get(f"{tau}|{k}|{key}")
            if s is not None and not diverged.get(f"{tau}|{k}"):
                val[(tau, k)] = float(np.mean(s))
    ref = opts.ref_tau or next(t for t in taus if (t, "fom") in val)
    print(f"\n=== area-weighted mean surface EKE (m2 s-2), yrs "
          f"{opts.year0}-{opts.year0 + opts.nyears}, anomaly about the "
          f"{'window time mean' if key == 'timemean' else 'day-of-year climatology'} ===")
    print(f"{'tau':>5} | " + " | ".join(f"{k:>11}" for k in KINDS) +
          " | " + " | ".join(f"{k + '/fom':>9}" for k in KINDS[1:]) +
          f" | normalized to tau {ref}")
    for tau in taus:
        cells = [f"{val[(tau, k)]:>11.4e}" if (tau, k) in val
                 else ("   DIVERGED" if diverged.get(f"{tau}|{k}") else " " * 11)
                 for k in KINDS]
        rats = [f"{val[(tau, k)] / val[(tau, 'fom')]:>9.3f}"
                if (tau, k) in val and (tau, "fom") in val else " " * 9 for k in KINDS[1:]]
        norm = [f"{k}={val[(tau, k)] / val[(ref, k)]:.2f}" for k in KINDS
                if (tau, k) in val and (ref, k) in val]
        print(f"{tau:>5} | " + " | ".join(cells) + " | " + " | ".join(rats) +
              " | " + "  ".join(norm))

    # ----------------------------------------------------------------- figure
    plt.rcParams.update({
        "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.fontsize": FS_TICK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.1), layout="constrained")
    x = {t: float(t) for t in taus}
    for k in KINDS:
        pts = [(x[t], val[(t, k)]) for t in taus if (t, k) in val]
        if not pts:
            continue
        st = {a: b for a, b in STYLE[k].items() if a != "label"}
        ax_a.plot(*zip(*pts), lw=1.4, **st)
        nb = [(x[t], val[(t, k)] / val[(ref, k)]) for t in taus
              if (t, k) in val and (ref, k) in val]
        ax_b.plot(*zip(*nb), lw=1.4, **st)
        for t in taus:
            if (t, k) not in val:
                continue
            m, ms = ("o", 4.0) if t in train else ("*", 7.0)
            for ax, yv in ((ax_a, val[(t, k)]),
                           (ax_b, val[(t, k)] / val[(ref, k)] if (ref, k) in val else None)):
                if yv is not None:
                    ax.plot(x[t], yv, m, ms=ms, color=st["color"],
                            mfc="white" if k == "proj" else st["color"], mew=0.9)
    for t in taus:
        if any(diverged.get(f"{t}|{k}") for k in KINDS):
            for ax in (ax_a, ax_b):
                ax.axvline(x[t], color="#999999", lw=0.6, ls="--", zorder=0)
            ax_a.text(x[t], 0.02, "ROM diverged", rotation=90, ha="right", va="bottom",
                      transform=ax_a.get_xaxis_transform(), fontsize=FS_TICK - 1,
                      color="#777777")
    # annotate the amplitude deficit at each tau -- it is the point of panel (a)
    for t in taus:
        if (t, "rom") in val and (t, "fom") in val:
            ax_a.annotate(f"{val[(t, 'rom')] / val[(t, 'fom')]:.0%}",
                          (x[t], val[(t, "rom")]), textcoords="offset points",
                          xytext=(0, -11), ha="center", va="top",
                          fontsize=FS_TICK - 1, color=C_MAIN)
    ax_a.set_yscale("log")
    ax_a.set_ylabel("Mean surface EKE (m$^2$ s$^{-2}$)")
    ax_a.set_title("(a) absolute", fontsize=FS_LABEL, loc="left")
    ax_b.set_ylabel(rf"EKE / EKE$(\tau_{{\max}}={ref})$")
    ax_b.set_title("(b) normalized — the sensitivity", fontsize=FS_LABEL, loc="left")
    for ax in (ax_a, ax_b):
        ax.set_xlabel(r"Wind-stress amplitude $\tau_{\max}$ (N m$^{-2}$)")
        ax.grid(alpha=0.25, lw=0.4, which="both")
        ax.set_xticks([float(t) for t in taus])
    handles = [Line2D([], [], lw=1.4, **STYLE[k]) for k in KINDS
               if any((t, k) in val for t in taus)]
    handles += [Line2D([], [], ls="none", marker="o", ms=4, color="k",
                       label="training amplitude"),
                Line2D([], [], ls="none", marker="*", ms=7, color="k",
                       label="held-out amplitude")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               frameon=False)
    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, facecolor="white")
    print(f"\nwrote {out}", flush=True)
    if opts.preview_png:
        fig.savefig(opts.preview_png, dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
