#!/usr/bin/env python3
"""Volume-averaged POINTWISE error metrics, FOM vs OpInf ROM, for one scenario.

This is the 3D counterpart of surface_error_metrics.py: the difference is taken
cell by cell and then averaged over the volume, so it answers the same question
the surface table answers at k = 0 ("how wrong is the field at a typical point?").

It is deliberately NOT the same as depth_avg_error_metrics.py, which averages the
field vertically *before* differencing and therefore measures the error of a
derived, column-integrated quantity (and is 5-28% smaller as a result).

Weights are cell volumes hFacC * drF * rA (partial cells included). The 1st-99th
percentile range used to normalize RMSE is taken over all wet 3D cells, matching
the surface convention of percentiles over wet points.

Reads fields_3d/{fom,rom}_3d_<scenario>.nc from export_3d_fields.py, falling back
to the original states_20yrs.nc for any FOM variable not duplicated there.

Usage:
    python3 src/volume_error_metrics.py --scenario run_10yrspinup_tau0.3
"""
import argparse
import time
from pathlib import Path

import numpy as np
import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))
import paths  # noqa: E402

UNITS = {"T": "degC", "S": "psu", "speed": "m/s"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=paths.BASE_RUN)
    p.add_argument("--scenario", default="run_10yrspinup_tau0.3")
    p.add_argument("--fom-prefix", default=paths.FOM_PREFIX)
    p.add_argument("--grid-dir", default=paths.GRID_DIR)
    p.add_argument("--vars", default="T,S,speed")
    p.add_argument("--range-pctl", type=float, default=1.0)
    p.add_argument("--range-stride", type=int, default=60,
                   help="time stride for the percentile-range estimate (memory bound)")
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
    rA = np.asarray(g["rA"].values, dtype="float64")
    vol = (hf * drf[:, None, None] * rA[None]).reshape(-1)     # (nspace,)
    wet = (hf > 0).reshape(-1)
    Vtot = vol.sum()
    log(f"{int(wet.sum())} wet cells of {wet.size}; total volume {Vtot:.4g} m^3")

    dd = Path(opts.data_dir)
    dr = netCDF4.Dataset(dd / "fields_3d" / f"rom_3d_{opts.scenario}.nc")
    df = netCDF4.Dataset(dd / "fields_3d" / f"fom_3d_{opts.scenario}.nc")
    nt = len(dr.dimensions["time"])
    tau = opts.scenario.split("tau")[-1]
    src = xr.open_dataset(Path(f"{opts.fom_prefix}{tau}") / "states_20yrs.nc",
                          engine="h5netcdf", decode_timedelta=False, decode_times=False)
    log(f"nt = {nt} ({nt / 360:.0f} yr)")

    rows = []
    for v in [s.strip() for s in opts.vars.split(",")]:
        t_v = time.time()
        s_ad = s_sd = s_f = s_f2 = 0.0
        sum_f = np.zeros(vol.size); sq_f = np.zeros(vol.size)
        sum_r = np.zeros(vol.size); sq_r = np.zeros(vol.size)
        vmin, vmax, samples = np.inf, -np.inf, []
        for t in range(nt):
            R = np.asarray(dr[v][t], dtype="float64").reshape(-1)
            F = (np.asarray(df[v][t], dtype="float64").reshape(-1) if v in df.variables
                 else src[v].isel(time=t).values.astype("float64").reshape(-1))
            d = R - F
            s_ad += float(np.abs(d) @ vol); s_sd += float((d * d) @ vol)
            s_f += float(F @ vol); s_f2 += float((F * F) @ vol)
            sum_f += F; sq_f += F * F
            sum_r += R; sq_r += R * R
            Fw = F[wet]
            vmin = min(vmin, float(Fw.min())); vmax = max(vmax, float(Fw.max()))
            if t % opts.range_stride == 0:
                samples.append(Fw.astype("float32"))
        smp = np.concatenate(samples)
        q = opts.range_pctl
        rrange = float(np.percentile(smp, 100 - q) - np.percentile(smp, q))
        del smp, samples

        mae = s_ad / (Vtot * nt)
        rmse = float(np.sqrt(s_sd / (Vtot * nt)))
        mean_f = s_f / (Vtot * nt)
        std_f = float(np.sqrt(max(s_f2 / (Vtot * nt) - mean_f**2, 0.0)))
        var_f = np.maximum(sq_f / nt - (sum_f / nt) ** 2, 0.0)
        var_r = np.maximum(sq_r / nt - (sum_r / nt) ** 2, 0.0)
        sig_f = float(np.sqrt((var_f @ vol) / Vtot))
        sig_r = float(np.sqrt((var_r @ vol) / Vtot))
        rows.append(dict(var=v, unit=UNITS.get(v, ""), mae=mae, rmse=rmse,
                         fom_mean=mean_f, fom_std=std_f,
                         nrmse_std_pct=100 * rmse / std_f,
                         fom_range=vmax - vmin, nrmse_range_pct=100 * rmse / (vmax - vmin),
                         fom_range_robust=rrange, nrmse_rrange_pct=100 * rmse / rrange,
                         sigma_fom=sig_f, sigma_rom=sig_r, sigma_ratio=sig_r / sig_f))
        log(f"  {v} done in {(time.time() - t_v) / 60:.1f} min")

    dr.close(); df.close(); src.close()
    hdr = (f"{'var':>6} {'unit':>6} | {'MAE':>10} {'RMSE':>10} {'nRMSE/std':>10} "
           f"{'nRMSE/rng':>10} {'nRMSE/rob':>10} {'sig_ROM/FOM':>11}")
    print(f"\n=== volume-averaged pointwise error, {opts.scenario}, {nt} steps "
          f"({nt / 360:.0f} yr) ===")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['var']:>6} {r['unit']:>6} | {r['mae']:>10.4g} {r['rmse']:>10.4g} "
              f"{r['nrmse_std_pct']:>9.1f}% {r['nrmse_range_pct']:>9.1f}% "
              f"{r['nrmse_rrange_pct']:>9.1f}% {r['sigma_ratio']:>11.3f}")
    print("(FOM std / full range / robust p1-p99 range: " + ", ".join(
        f"{r['var']} {r['fom_std']:.4g}/{r['fom_range']:.4g}/{r['fom_range_robust']:.4g}"
        for r in rows) + ")")

    if opts.out_csv:
        import csv
        out = Path(opts.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
