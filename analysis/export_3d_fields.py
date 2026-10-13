#!/usr/bin/env python3
"""Export full 3D (time, k, j, i) FOM and OpInf-ROM fields for one scenario.

Writes T, S and horizontal speed over the whole training/rollout horizon so that
depth-averaged and per-level diagnostics can be computed downstream.

Method
------
The ROM state is never stored in full dimension by the training driver: only the
temporal basis ``Tr`` (nt_train x r) and the reduced trajectory ``Q_ROM`` (r x nt)
are saved. A physical field is recovered per variable as

    rom = (X_train - center) @ Tr @ Q_ROM + center

where ``X_train`` are the *unscaled* centred training snapshots stacked across
training scenarios. The scaling alpha cancels, because the reduced coordinates
live in scaled space while the lifted basis is built from unscaled data (this is
the same identity used by utils.compute_barotropic_streamfunction and
eval_rom_forcings.transform_and_project_k_forcings).

Rather than reading one model level at a time (a strided read of the ~221 GB
snapshot file, measured at ~17 MB/s), this streams whole time chunks with all 31
levels at once (~150-260 MB/s) and accumulates

    A = X_train @ Tr          and     center = mean_t X_train

in a single pass, then forms the lifted basis as ``A - center * Tr.sum(axis=0)``.

Speed is nonlinear, so it is computed after reconstructing U and V. U and V live
on staggered faces (i_g / j_g); they are averaged to cell centres with a
wet-face-aware two-point average using hFacW / hFacS, and the same operation is
applied to FOM and ROM so the comparison is consistent.

Usage:
    python3 src/export_3d_fields.py --scenario run_10yrspinup_tau0.3
"""
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import xarray as xr
import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'src'))
import paths  # noqa: E402

NZ, NY, NX = 31, 248, 248
NXY = NY * NX
NSPACE = NZ * NXY
DIRECT_VARS = ("T", "S")          # reconstructed and written as-is
SPEED_PARTS = ("U", "V")          # reconstructed only to build speed


def open_states(d):
    return xr.open_dataset(Path(d) / "states_20yrs.nc", engine="h5netcdf",
                           decode_timedelta=False, decode_times=False)


def build_lifted(var, train_dirs, tr, nt_use, chunk, log):
    """Single streaming pass -> (lifted basis (NSPACE, r) float32, center (NSPACE,))."""
    r = tr.shape[1]
    A = np.zeros((NSPACE, r), dtype=np.float32)
    tmp = np.empty((NSPACE, r), dtype=np.float32)
    csum = np.zeros(NSPACE, dtype=np.float64)
    col, t_start, nbytes = 0, time.time(), 0
    for d in train_dirs:
        ds = open_states(d)
        for t0 in range(0, nt_use, chunk):
            t1 = min(t0 + chunk, nt_use)
            X = ds[var].isel(time=slice(t0, t1)).values          # (nc, k, j, i)
            nbytes += X.nbytes
            X2 = X.reshape(X.shape[0], -1)                       # (nc, NSPACE), C order
            np.matmul(X2.T, tr[col:col + X2.shape[0]], out=tmp)
            A += tmp
            csum += X2.sum(axis=0, dtype=np.float64)
            col += X2.shape[0]
            del X, X2
        ds.close()
    assert col == nt_use * len(train_dirs), f"{col} training columns, expected {nt_use * len(train_dirs)}"
    center = (csum / col).astype(np.float32)
    A -= center[:, None] * tr.sum(axis=0)[None, :]
    dt = time.time() - t_start
    log(f"  lifted basis for {var}: {col} cols, read {nbytes / 2**30:.1f} GB "
        f"in {dt / 60:.1f} min ({nbytes / 2**20 / dt:.0f} MB/s)")
    del tmp
    gc.collect()
    return A, center


def validate_against_section(lifted, center, q_rom, data_dir, field, scenario_label, log):
    """Cross-check every level against the saved i=124 vertical section, if present."""
    path = Path(data_dir) / "analysis" / "computed_fields" / f"{field}_vertical_{scenario_label}.nc"
    if not path.exists():
        log(f"  (no reference section {path.name}; skipping validation)")
        return
    ref = xr.open_dataset(path, engine="h5netcdf")
    name = f"{field}_rom"
    if name not in ref.data_vars:
        log(f"  (no {name} in {path.name}; skipping validation)")
        ref.close()
        return
    i_sec = int(ref.attrs.get("vertical_i", 124))
    ts = [0, min(100, q_rom.shape[1] - 1), q_rom.shape[1] - 1]
    rows = (np.arange(NZ)[:, None] * NXY + np.arange(NY)[None, :] * NX + i_sec).reshape(-1)
    got = (lifted[rows] @ q_rom[:, ts] + center[rows, None]).reshape(NZ, NY, len(ts))
    want = ref[name].isel(time=ts).values                        # (nt, k, j)
    ref.close()
    for n, t in enumerate(ts):
        a, b = got[:, :, n], want[n]
        m = np.isfinite(a) & np.isfinite(b) & (b != 0)
        log(f"  validate {field} section t={t}: max|diff|={np.abs(a[m] - b[m]).max():.3e} "
            f"(field rms {np.sqrt((b[m]**2).mean()):.4g})")


def center_speed(u, v, wu, wv):
    """Wet-face-aware average of staggered u, v to cell centres, then speed.

    u, v: (nc, k, j, i) on i_g / j_g. wu, wv: (k, j, i) wet-face indicators.
    """
    uc = np.zeros_like(u)
    num = u * wu
    den = wu.copy()
    uc[..., :-1] = num[..., :-1] + num[..., 1:]
    dl = den[..., :-1] + den[..., 1:]
    uc[..., :-1] /= np.where(dl > 0, dl, 1.0)
    uc[..., -1] = u[..., -1]
    vc = np.zeros_like(v)
    numv = v * wv
    denv = wv.copy()
    vc[:, :, :-1, :] = numv[:, :, :-1, :] + numv[:, :, 1:, :]
    dlv = denv[:, :-1, :] + denv[:, 1:, :]
    vc[:, :, :-1, :] /= np.where(dlv > 0, dlv, 1.0)
    vc[:, :, -1, :] = v[:, :, -1, :]
    return np.sqrt(uc**2 + vc**2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=paths.BASE_RUN)
    p.add_argument("--scenario", default="run_10yrspinup_tau0.3")
    p.add_argument("--train-taus", default="0.2,0.4")
    p.add_argument("--snapshot-prefix", default=paths.FOM_PREFIX)
    p.add_argument("--grid-dir", default=paths.GRID_DIR)
    p.add_argument("--n-year-train", type=int, default=10)
    p.add_argument("--n-days", type=int, default=1)
    p.add_argument("--read-chunk", type=int, default=20, help="time steps per streaming read")
    p.add_argument("--write-chunk", type=int, default=20, help="time steps per output write")
    p.add_argument("--speed-chunk", type=int, default=10, help="time steps per write for speed")
    p.add_argument("--out-subdir", default="fields_3d")
    p.add_argument("--skip-fom", action="store_true")
    p.add_argument("--fom-vars", default="speed",
                   help="which FOM variables to write. Default 'speed' only: T and S would be a "
                        "byte-for-byte subset copy of states_20yrs.nc, whose (time,k,j,i) layout is "
                        "already contiguous per timestep, so a re-chunked copy buys nothing. "
                        "Pass 'T,S,speed' to write a self-contained FOM file anyway.")
    p.add_argument("--rom-vars", default="T,S,speed")
    p.add_argument("--skip-rom", action="store_true")
    p.add_argument("--train-limit", type=int, default=None,
                   help="debug: cap the number of training columns per scenario")
    p.add_argument("--nt-limit", type=int, default=None,
                   help="debug: cap the number of output time steps")
    p.add_argument("--spike-limit", type=float, default=100.0,
                   help="|value| above this is set to NaN in the ROM, as in the surface export")
    opts = p.parse_args()

    import netCDF4

    t_all = time.time()

    def log(msg):
        print(f"[{(time.time() - t_all) / 60:6.1f} min] {msg}", flush=True)

    dd = Path(opts.data_dir)
    out_dir = dd / opts.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    nt_use = 360 * opts.n_year_train // opts.n_days
    if opts.train_limit:
        nt_use = min(nt_use, opts.train_limit)
    train_dirs = [f"{opts.snapshot_prefix}{float(t):g}" for t in opts.train_taus.split(",")]

    tr = np.load(dd / "Tr.npy").astype(np.float32)
    cands = sorted(dd.glob(f"Q_ROM_*_{opts.scenario}.npy"))
    if not cands:
        raise SystemExit(f"no Q_ROM_*_{opts.scenario}.npy in {dd}")
    q_rom = np.load(cands[0]).astype(np.float32)
    nt = q_rom.shape[1] if opts.nt_limit is None else min(opts.nt_limit, q_rom.shape[1])
    log(f"Tr {tr.shape}, q_rom {q_rom.shape} from {cands[0].name}")
    log(f"training dirs: {train_dirs} ({nt_use} cols each)")
    log(f"output -> {out_dir}  ({nt * NSPACE * 4 / 2**30:.1f} GB per 3D variable)")

    import xmitgcm

    g = xmitgcm.open_mdsdataset(opts.grid_dir, iters=None)
    wu = (np.asarray(g["hFacW"].values) > 0).astype(np.float32)
    wv = (np.asarray(g["hFacS"].values) > 0).astype(np.float32)
    coords = dict(Z=np.asarray(g["Z"].values), YC=np.asarray(g["YC"].values).reshape(-1)[:NY]
                  if np.asarray(g["YC"].values).ndim == 1 else np.asarray(g["YC"].values)[:, 0],
                  XC=np.asarray(g["XC"].values).reshape(-1)[:NX]
                  if np.asarray(g["XC"].values).ndim == 1 else np.asarray(g["XC"].values)[0, :])

    def make_file(path, varnames):
        nc = netCDF4.Dataset(path, "w", format="NETCDF4")
        nc.createDimension("time", nt)
        nc.createDimension("k", NZ)
        nc.createDimension("j", NY)
        nc.createDimension("i", NX)
        nc.createVariable("time", "i4", ("time",))[:] = np.arange(nt) * opts.n_days
        nc.createVariable("Z", "f4", ("k",))[:] = coords["Z"]
        nc.createVariable("YC", "f4", ("j",))[:] = coords["YC"]
        nc.createVariable("XC", "f4", ("i",))[:] = coords["XC"]
        for v, units in (("T", "degC"), ("S", "psu"), ("speed", "m/s")):
            if v not in varnames:
                continue
            var = nc.createVariable(v, "f4", ("time", "k", "j", "i"),
                                    chunksizes=(1, NZ, NY, NX), zlib=False)
            var.units = units
        nc.scenario = opts.scenario
        nc.note = ("speed = |horizontal velocity| at cell centres, from a wet-face-aware "
                   "two-point average of U (i_g) and V (j_g)")
        return nc

    # ------------------------------------------------------------------ FOM
    if not opts.skip_fom:
        fom_vars = tuple(v.strip() for v in opts.fom_vars.split(",") if v.strip())
        path = out_dir / f"fom_3d_{opts.scenario}.nc"
        nc = make_file(path, fom_vars)
        ds = open_states(f"{opts.snapshot_prefix}{opts.scenario.split('tau')[-1]}")
        nc.source_note = ("T and S, if absent here, are an unmodified subset (first "
                          f"{nt} steps) of states_20yrs.nc and should be read from there.")
        for v in [v for v in DIRECT_VARS if v in fom_vars]:
            t0w = time.time()
            for t0 in range(0, nt, opts.write_chunk):
                t1 = min(t0 + opts.write_chunk, nt)
                nc[v][t0:t1] = ds[v].isel(time=slice(t0, t1)).values
            log(f"  FOM {v} written in {(time.time() - t0w) / 60:.1f} min")
        if "speed" in fom_vars:
            t0w = time.time()
            for t0 in range(0, nt, opts.speed_chunk):
                t1 = min(t0 + opts.speed_chunk, nt)
                u = ds["U"].isel(time=slice(t0, t1)).values
                v_ = ds["V"].isel(time=slice(t0, t1)).values
                nc["speed"][t0:t1] = center_speed(u, v_, wu, wv)
                del u, v_
            log(f"  FOM speed written in {(time.time() - t0w) / 60:.1f} min")
        ds.close()
        nc.close()
        log(f"wrote {path} ({path.stat().st_size / 2**30:.1f} GB)")

    # ------------------------------------------------------------------ ROM
    if not opts.skip_rom:
        rom_vars = tuple(v.strip() for v in opts.rom_vars.split(",") if v.strip())
        path = out_dir / f"rom_3d_{opts.scenario}.nc"
        nc = make_file(path, rom_vars)
        label = opts.scenario.split("run_10yrspinup_")[-1]
        fields = {"T": "temperature", "S": "salinity"}
        for v in [v for v in DIRECT_VARS if v in rom_vars]:
            lifted, center = build_lifted(v, train_dirs, tr, nt_use, opts.read_chunk, log)
            validate_against_section(lifted, center, q_rom, dd, fields[v],
                                     f"{label}-test" if "0.3" in label else label, log)
            t0w = time.time()
            for t0 in range(0, nt, opts.write_chunk):
                t1 = min(t0 + opts.write_chunk, nt)
                blk = lifted @ q_rom[:, t0:t1]
                blk += center[:, None]
                blk[np.abs(blk) > opts.spike_limit] = np.nan
                nc[v][t0:t1] = blk.T.reshape(t1 - t0, NZ, NY, NX)
                del blk
            log(f"  ROM {v} written in {(time.time() - t0w) / 60:.1f} min")
            del lifted, center
            gc.collect()

        if "speed" not in rom_vars:
            nc.close()
            log(f"wrote {path}")
            return
        lu, cu = build_lifted("U", train_dirs, tr, nt_use, opts.read_chunk, log)
        lv, cv = build_lifted("V", train_dirs, tr, nt_use, opts.read_chunk, log)
        t0w = time.time()
        for t0 in range(0, nt, opts.speed_chunk):
            t1 = min(t0 + opts.speed_chunk, nt)
            qc = q_rom[:, t0:t1]
            u = (lu @ qc + cu[:, None]).T.reshape(t1 - t0, NZ, NY, NX)
            v_ = (lv @ qc + cv[:, None]).T.reshape(t1 - t0, NZ, NY, NX)
            nc["speed"][t0:t1] = center_speed(u, v_, wu, wv)
            del u, v_
        log(f"  ROM speed written in {(time.time() - t0w) / 60:.1f} min")
        del lu, lv
        gc.collect()
        nc.close()
        log(f"wrote {path} ({path.stat().st_size / 2**30:.1f} GB)")

    log("done")


if __name__ == "__main__":
    main()
