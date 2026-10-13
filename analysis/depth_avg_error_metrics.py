#!/usr/bin/env python3
"""Depth-averaged error metrics, FOM vs OpInf ROM, for one scenario.

Reads the 3D fields written by export_3d_fields.py plus (for FOM T and S, which
that script deliberately does not duplicate) the original states_20yrs.nc.

For each variable the volume-weighted column average is formed,

    xbar(j,i,t) = sum_k hFacC_k drF_k x(k,j,i,t) / sum_k hFacC_k drF_k

(partial cells handled by hFacC), and then the same metrics as
surface_error_metrics.py are reported over wet columns: MAE and max |bias| at one
snapshot, and MAE, RMSE, RMSE as % of the FOM spatio-temporal std, RMSE as % of
the FOM range (full and outlier-robust p1-p99), and the anomaly retention ratio
sigma_ROM / sigma_FOM over the whole record.

Anomaly variance is obtained in a single streaming pass from per-point sums and
sums of squares (var = <x^2> - <x>^2, accumulated in float64), so the record is
read only once.

Usage:
    python3 src/depth_avg_error_metrics.py --scenario run_10yrspinup_tau0.3
"""
import argparse
import time
from pathlib import Path

import numpy as np
import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))
import paths  # noqa: E402

NZ, NY, NX = 31, 248, 248
UNITS = {"T": "degC", "S": "psu", "speed": "m/s"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=paths.BASE_RUN)
    p.add_argument("--scenario", default="run_10yrspinup_tau0.3")
    p.add_argument("--fom-prefix", default=paths.FOM_PREFIX)
    p.add_argument("--grid-dir", default=paths.GRID_DIR)
    p.add_argument("--vars", default="T,S,speed")
    p.add_argument("--time", type=int, default=100, help="snapshot index for the snapshot columns")
    p.add_argument("--chunk", type=int, default=20)
    p.add_argument("--range-pctl", type=float, default=1.0)
    p.add_argument("--range-stride", type=int, default=10)
    p.add_argument("--out-csv", default=None)
    opts = p.parse_args()

    import netCDF4
    import xarray as xr
    import xmitgcm

    t_all = time.time()
    log = lambda m: print(f"[{(time.time() - t_all) / 60:5.1f} min] {m}", flush=True)

    g = xmitgcm.open_mdsdataset(opts.grid_dir, iters=None)
    hf = np.asarray(g["hFacC"].values, dtype="float64")
    drf = np.asarray(g["drF"].values, dtype="float64")
    W = hf * drf[:, None, None]                        # (k, j, i) metres of water per cell
    depth = W.sum(axis=0)                              # (j, i) column thickness
    mask = depth > 0
    area = np.asarray(g["rA"].values, dtype="float64")
    w_area = np.where(mask, area, 0.0)
    w_uni = mask.astype("float64")
    weights = {"area-weighted": w_area, "unweighted": w_uni}
    inv_depth = np.where(mask, 1.0 / np.where(depth > 0, depth, 1.0), 0.0)
    log(f"{int(mask.sum())} wet columns; depth {depth[mask].min():.0f}-{depth[mask].max():.0f} m")

    fom3d = Path(opts.data_dir) / "fields_3d" / f"fom_3d_{opts.scenario}.nc"
    rom3d = Path(opts.data_dir) / "fields_3d" / f"rom_3d_{opts.scenario}.nc"
    tau = opts.scenario.split("tau")[-1]
    src = Path(f"{opts.fom_prefix}{tau}") / "states_20yrs.nc"

    dr = netCDF4.Dataset(rom3d)
    nt = len(dr.dimensions["time"])
    df = netCDF4.Dataset(fom3d)
    ds_src = xr.open_dataset(src, engine="h5netcdf", decode_timedelta=False, decode_times=False)
    log(f"nt = {nt} ({nt / 360:.0f} yr); ROM {rom3d.name}, FOM {fom3d.name} + {src.name}")

    def davg(arr4):
        """(nc, k, j, i) -> (nc, j, i) volume-weighted column mean."""
        return np.einsum("tkji,kji->tji", arr4.astype("float64"), W) * inv_depth

    def fom_chunk(v, t0, t1):
        if v in df.variables:
            return np.asarray(df[v][t0:t1])
        return ds_src[v].isel(time=slice(t0, t1)).values

    rows = []
    for v in [s.strip() for s in opts.vars.split(",")]:
        t_v = time.time()
        sf = np.zeros((NY, NX)); sf2 = np.zeros((NY, NX))
        sr = np.zeros((NY, NX)); sr2 = np.zeros((NY, NX))
        sad = np.zeros((NY, NX)); ssd = np.zeros((NY, NX))
        vmin, vmax, samples = np.inf, -np.inf, []
        snap = {}
        for t0 in range(0, nt, opts.chunk):
            t1 = min(t0 + opts.chunk, nt)
            F = davg(fom_chunk(v, t0, t1))
            R = davg(np.asarray(dr[v][t0:t1]))
            D = R - F
            sf += F.sum(axis=0); sf2 += (F**2).sum(axis=0)
            sr += R.sum(axis=0); sr2 += (R**2).sum(axis=0)
            sad += np.abs(D).sum(axis=0); ssd += (D**2).sum(axis=0)
            Fm = F[:, mask]
            vmin = min(vmin, float(Fm.min())); vmax = max(vmax, float(Fm.max()))
            for i, t in enumerate(range(t0, t1)):
                if t % opts.range_stride == 0:
                    samples.append(F[i][mask].astype("float32"))
                if t == opts.time:
                    d0 = np.where(mask, D[i], np.nan)
                    snap = dict(maxbias=float(np.nanmax(np.abs(d0))),
                                p99bias=float(np.nanpercentile(np.abs(d0), 99)),
                                mae={k: float((np.abs(np.nan_to_num(d0)) * w).sum() / w.sum())
                                     for k, w in weights.items()})
            del F, R, D
        smp = np.concatenate(samples)
        q = opts.range_pctl
        rrange = float(np.percentile(smp, 100 - q) - np.percentile(smp, q))
        del smp, samples

        var_f_pt = np.maximum(sf2 / nt - (sf / nt) ** 2, 0.0)
        var_r_pt = np.maximum(sr2 / nt - (sr / nt) ** 2, 0.0)
        for k, w in weights.items():
            W_ = w.sum()
            mae = float((sad * w).sum() / (W_ * nt))
            rmse = float(np.sqrt((ssd * w).sum() / (W_ * nt)))
            mf = float((sf * w).sum() / (W_ * nt))
            std_f = float(np.sqrt(max((sf2 * w).sum() / (W_ * nt) - mf**2, 0.0)))
            sig_f = float(np.sqrt((var_f_pt * w).sum() / W_))
            sig_r = float(np.sqrt((var_r_pt * w).sum() / W_))
            rows.append(dict(var=v, unit=UNITS.get(v, ""), weighting=k,
                             snap_mae=snap["mae"][k], snap_p99bias=snap["p99bias"],
                             snap_maxbias=snap["maxbias"], mae=mae, rmse=rmse,
                             fom_std=std_f, nrmse_std_pct=100 * rmse / std_f,
                             fom_range=vmax - vmin, nrmse_range_pct=100 * rmse / (vmax - vmin),
                             fom_range_robust=rrange, nrmse_rrange_pct=100 * rmse / rrange,
                             sigma_fom=sig_f, sigma_rom=sig_r, sigma_ratio=sig_r / sig_f))
        log(f"  {v} done in {(time.time() - t_v) / 60:.1f} min")

    dr.close(); df.close(); ds_src.close()

    hdr = (f"{'var':>6} {'unit':>6} | {'snap MAE':>10} {'snap p99|b|':>11} {'snap max|b|':>11} "
           f"| {'MAE':>10} {'RMSE':>10} {'nRMSE/std':>10} {'nRMSE/rng':>10} {'nRMSE/rob':>10} "
           f"{'sig_ROM/FOM':>11}")
    for k in weights:
        print(f"\n=== depth-averaged, {opts.scenario}, {nt} steps ({nt / 360:.0f} yr), {k} ===")
        print(hdr); print("-" * len(hdr))
        for r in [r for r in rows if r["weighting"] == k]:
            print(f"{r['var']:>6} {r['unit']:>6} | {r['snap_mae']:>10.4g} "
                  f"{r['snap_p99bias']:>11.4g} {r['snap_maxbias']:>11.4g} | "
                  f"{r['mae']:>10.4g} {r['rmse']:>10.4g} {r['nrmse_std_pct']:>9.1f}% "
                  f"{r['nrmse_range_pct']:>9.1f}% {r['nrmse_rrange_pct']:>9.1f}% "
                  f"{r['sigma_ratio']:>11.3f}")
        print("(FOM std / full range / robust range: " + ", ".join(
            f"{r['var']} {r['fom_std']:.4g}/{r['fom_range']:.4g}/{r['fom_range_robust']:.4g}"
            for r in rows if r["weighting"] == k) + ")")

    if opts.out_csv:
        import csv
        out = Path(opts.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
