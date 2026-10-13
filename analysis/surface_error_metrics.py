#!/usr/bin/env python3
"""Surface-field error metrics, FOM vs OpInf ROM, for one scenario.

Reads the split surface files written by export_surface_fields.py
(<data-dir>/surface_fields/{fom,rom}_surface_<scenario>.nc, dims (time, y, x)).

Reports, per state variable:
  * snapshot metrics at --time (the time index used in the snapshot figure):
    MAE and max |bias|
  * full-record metrics over the whole rollout:
    MAE, RMSE, RMSE as % of the FOM spatio-temporal std, RMSE as % of the FOM
    range, and the anomaly variability-retention ratio sigma_ROM / sigma_FOM
    (anomalies = deviations from each point's own time mean)

Everything is computed over wet points only. Both area-weighted (MITgcm rA) and
unweighted variants are printed; the saved fields are (time, y, x) whereas rA is
(y, x) on the model grid, so the orientation of rA is checked against the land
mask before use.

Streams the record in time chunks (the files are ~0.9 GB per variable).

Usage:
    python3 src/surface_error_metrics.py --scenario run_10yrspinup_tau0.3 --time 100
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))
import paths  # noqa: E402

VARS = ["T", "S", "Eta", "U", "V"]
UNITS = {"T": "degC", "S": "psu", "Eta": "m", "U": "m/s", "V": "m/s"}


def orient_area(grid_dir, mask):
    """Return rA on the saved (y, x) orientation, or None if unavailable."""
    try:
        import xmitgcm

        g = xmitgcm.open_mdsdataset(grid_dir, iters=None)
        area = np.asarray(g["rA"].values, dtype="float64")
        hfac = np.asarray(g["hFacC"].isel(Z=0).values) > 0
        if hfac.shape == mask.shape:
            agree = (hfac == mask).mean()
            agree_T = (hfac.T == mask).mean()
            if agree_T > agree:
                print(f"  rA transposed to match the saved mask "
                      f"(agreement {agree_T:.4f} vs {agree:.4f} untransposed)")
                return area.T
            print(f"  rA used as-is (agreement {agree:.4f} vs {agree_T:.4f} transposed)")
        return area
    except Exception as exc:                       # noqa: BLE001
        print(f"  could not read rA from {grid_dir}: {exc}; unweighted only")
        return None


def wmean(x, w):
    """Weighted mean of x over the wet points defined by w (w already masked)."""
    return float((x * w).sum() / w.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=paths.BASE_RUN)
    p.add_argument("--scenario", default="run_10yrspinup_tau0.3")
    p.add_argument("--grid-dir", default=paths.GRID_DIR)
    p.add_argument("--time", type=int, default=100, help="snapshot time index for the figure metrics")
    p.add_argument("--chunk", type=int, default=180, help="time steps per streaming chunk")
    p.add_argument("--range-pctl", type=float, default=1.0,
                   help="percentile for the outlier-robust FOM range (1.0 -> p1-p99)")
    p.add_argument("--range-stride", type=int, default=10,
                   help="time stride used when estimating the robust range")
    p.add_argument("--out-csv", default=None)
    opts = p.parse_args()

    sd = Path(opts.data_dir) / "surface_fields"
    fom = xr.open_dataset(sd / f"fom_surface_{opts.scenario}.nc", engine="h5netcdf")
    rom = xr.open_dataset(sd / f"rom_surface_{opts.scenario}.nc", engine="h5netcdf")
    nt, ny, nx = fom["T"].shape
    mask = fom["T"].isel(time=0).values != 0            # T = 0 only on land
    print(f"{opts.scenario}: nt={nt}, grid {ny}x{nx}, {int(mask.sum())} wet points")
    area = orient_area(opts.grid_dir, mask)

    w_uni = mask.astype("float64")
    w_area = w_uni if area is None else np.where(mask, area, 0.0)
    weights = {"area-weighted": w_area, "unweighted": w_uni}

    rows = []
    for v in VARS:
        fa, ra = fom[v], rom[v]

        # ---- snapshot at --time -----------------------------------------
        f0 = fa.isel(time=opts.time).values.astype("float64")
        r0 = ra.isel(time=opts.time).values.astype("float64")
        d0 = np.where(mask, r0 - f0, np.nan)
        snap = {k: wmean(np.abs(np.nan_to_num(d0)), w) for k, w in weights.items()}
        snap_max = float(np.nanmax(np.abs(d0)))
        p99 = float(np.nanpercentile(np.abs(d0), 99))

        # ---- outlier-robust FOM range (sampled in time) ------------------
        samp = fa.isel(time=slice(0, nt, opts.range_stride)).values.astype("float64")[:, mask]
        q = opts.range_pctl
        rr = float(np.percentile(samp, 100 - q) - np.percentile(samp, q))
        del samp

        # ---- pass 1: time means, error moments, FOM moments, range ------
        acc = {k: dict(abs=0.0, sq=0.0, f=0.0, f2=0.0) for k in weights}
        sum_f = np.zeros((ny, nx)); sum_r = np.zeros((ny, nx))
        vmin, vmax = np.inf, -np.inf
        for i0 in range(0, nt, opts.chunk):
            F = fa.isel(time=slice(i0, i0 + opts.chunk)).values.astype("float64")
            R = ra.isel(time=slice(i0, i0 + opts.chunk)).values.astype("float64")
            D = R - F
            sum_f += F.sum(axis=0); sum_r += R.sum(axis=0)
            Fm = F[:, mask]
            vmin = min(vmin, float(Fm.min())); vmax = max(vmax, float(Fm.max()))
            for k, w in weights.items():
                acc[k]["abs"] += float((np.abs(D) * w).sum())
                acc[k]["sq"] += float((D**2 * w).sum())
                acc[k]["f"] += float((F * w).sum())
                acc[k]["f2"] += float((F**2 * w).sum())
            del F, R, D
        mean_f, mean_r = sum_f / nt, sum_r / nt

        # ---- pass 2: anomaly variance about each point's time mean ------
        anom = {k: dict(f=0.0, r=0.0) for k in weights}
        for i0 in range(0, nt, opts.chunk):
            F = fa.isel(time=slice(i0, i0 + opts.chunk)).values.astype("float64") - mean_f
            R = ra.isel(time=slice(i0, i0 + opts.chunk)).values.astype("float64") - mean_r
            for k, w in weights.items():
                anom[k]["f"] += float((F**2 * w).sum())
                anom[k]["r"] += float((R**2 * w).sum())
            del F, R

        for k, w in weights.items():
            wt = w.sum() * nt
            mae = acc[k]["abs"] / wt
            rmse = np.sqrt(acc[k]["sq"] / wt)
            std_f = np.sqrt(max(acc[k]["f2"] / wt - (acc[k]["f"] / wt) ** 2, 0.0))
            sig_f = np.sqrt(anom[k]["f"] / wt)
            sig_r = np.sqrt(anom[k]["r"] / wt)
            rows.append(dict(var=v, unit=UNITS[v], weighting=k,
                             snap_mae=snap[k], snap_maxbias=snap_max, snap_p99bias=p99,
                             mae=mae, rmse=rmse, fom_std=std_f,
                             nrmse_std_pct=100 * rmse / std_f,
                             fom_range=vmax - vmin,
                             nrmse_range_pct=100 * rmse / (vmax - vmin),
                             fom_range_robust=rr,
                             nrmse_rrange_pct=100 * rmse / rr,
                             sigma_fom=sig_f, sigma_rom=sig_r, sigma_ratio=sig_r / sig_f))
        print(f"  {v} done", flush=True)

    hdr = (f"{'var':>4} {'unit':>6} | {'snap MAE':>10} {'snap p99|b|':>11} {'snap max|b|':>11} "
           f"| {'MAE':>10} {'RMSE':>10} {'nRMSE/std':>10} {'nRMSE/rng':>10} {'nRMSE/rob':>10} "
           f"{'sig_ROM/FOM':>11}")
    for k in weights:
        print(f"\n=== {opts.scenario}, full {nt} steps ({nt / 360:.0f} yr), {k} ===")
        print(hdr)
        print("-" * len(hdr))
        for r in [r for r in rows if r["weighting"] == k]:
            print(f"{r['var']:>4} {r['unit']:>6} | {r['snap_mae']:>10.4g} "
                  f"{r['snap_p99bias']:>11.4g} {r['snap_maxbias']:>11.4g} | "
                  f"{r['mae']:>10.4g} {r['rmse']:>10.4g} "
                  f"{r['nrmse_std_pct']:>9.1f}% {r['nrmse_range_pct']:>9.1f}% "
                  f"{r['nrmse_rrange_pct']:>9.1f}% {r['sigma_ratio']:>11.3f}")
        print("(FOM std / full range / robust range: " + ", ".join(
            f"{r['var']} {r['fom_std']:.4g}/{r['fom_range']:.4g}/{r['fom_range_robust']:.4g}"
            for r in rows if r["weighting"] == k) + ")")

    if opts.out_csv:
        import csv

        out = Path(opts.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
