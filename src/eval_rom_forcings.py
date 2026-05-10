#!/usr/bin/env python3
"""Evaluate multi-forcing OpInf ROMs against MITgcm SOMA runs.

This script is the cleaned-up, scriptable version of
``notebooks/eval_rom_forcings-2.ipynb``.  It assumes the multi-forcing ROM was
trained with a kernel/SVD representation where ``Tr.npy`` maps training
snapshots to the reduced spatial basis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import cmocean.cm as cmo


VAR_META = {
    "Eta": {
        "short": "SSH",
        "label": "Sea surface height [m]",
        "units": "m",
        "cmap": "RdBu_r",
        "vmin": -1.0,
        "vmax": 1.0,
    },
    "T": {
        "short": "SST",
        "label": "Sea Surface Temperature [deg C]",
        "units": r"$^\circ$C",
        "cmap": "inferno",
        "vmin": -2.0,
        "vmax": 30.0,
    },
    "speed": {
        "short": "Speed",
        "label": "Surface speed [m/s]",
        "units": "m/s",
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 1.0,
    },
}


@dataclass
class Scenario:
    label: str
    fom_dir: str
    rom_file: str
    is_training: bool = False


@dataclass
class SurfaceFields:
    sst_fom: xr.DataArray
    sst_rom: np.ndarray
    ssh_fom: xr.DataArray
    ssh_rom: np.ndarray
    speed_fom: xr.DataArray
    speed_rom: np.ndarray


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ensure_slash(path: str) -> str:
    return path if path.endswith("/") else path + "/"


def list_to_csv(values: list[str]) -> str:
    return ",".join(str(value) for value in values)


def scenario_name_from_dir(path: str) -> str:
    return Path(path.rstrip("/")).name


def scenario_label_from_dir(path: str, is_test: bool = False) -> str:
    name = scenario_name_from_dir(path)
    match = re.search(r"tau\d+(?:\.\d+)?", name)
    label = match.group(0) if match else name
    return f"{label}-test" if is_test else label


def infer_r_from_data_dir(data_dir: str | None) -> int | None:
    if data_dir is None:
        return None
    match = re.search(r"_r(\d+)(?:/)?$", data_dir.rstrip("/"))
    return int(match.group(1)) if match else None


def infer_r_from_tr(data_dir: str | None) -> int | None:
    if data_dir is None:
        return None
    tr_path = Path(data_dir) / "Tr.npy"
    if not tr_path.exists():
        return None
    return int(np.load(tr_path, mmap_mode="r").shape[1])


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill evaluator options from the ROM training JSON unless CLI overrides them."""
    if args.config is None:
        return args

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    args.target_ret_energy = config.get("target_ret_energy")

    scalar_keys = {
        "n_year_train": "n_year_train",
        "n_year_predict": "n_year_predict",
        "n_days": "n_days",
        "center_opt": "center_opt",
        "scale": "scale",
        "r": "r",
    }
    for attr, key in scalar_keys.items():
        if getattr(args, attr) is None and config.get(key) is not None:
            setattr(args, attr, config[key])

    snapshot_dirs = config.get("snapshot_dirs") or []
    test_dir = config.get("test_dir")
    if args.training_dirs is None and snapshot_dirs:
        args.training_dirs = list_to_csv(snapshot_dirs)
    if args.test_dir is None and test_dir:
        args.test_dir = test_dir

    fom_dirs = snapshot_dirs + ([test_dir] if test_dir else [])
    if args.scenario_labels is None and fom_dirs:
        labels = config.get("scenario_labels") or [
            scenario_label_from_dir(path, is_test=(idx == len(fom_dirs) - 1))
            for idx, path in enumerate(fom_dirs)
        ]
        args.scenario_labels = list_to_csv(labels)

    if args.rom_files is None and snapshot_dirs and test_dir:
        train_files = [f"Q_ROM_train_{scenario_name_from_dir(path)}.npy" for path in snapshot_dirs]
        test_file = f"Q_ROM_test_{scenario_name_from_dir(test_dir)}.npy"
        args.rom_files = list_to_csv(train_files + [test_file])

    if args.preproc_dir is None and config.get("dir_extension"):
        args.preproc_dir = str(Path(args.root_dir) / "save_roms" / config["dir_extension"])

    return args


def load_center(preproc_dir: str, var: str, center_opt: str, n_days: int, n_year_train: int) -> xr.DataArray:
    path = Path(preproc_dir) / f"center{var}_{center_opt}_{n_days}days_{n_year_train}yrs.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing center file for {var}: {path}")
    return xr.open_dataset(path, engine="netcdf4").center


def load_training_snapshots_at_k(
    snapshot_dirs: list[str],
    var: str,
    n_year_train: int,
    n_days: int,
    k: int,
) -> np.ndarray:
    """Return stacked training snapshots as (space, time across scenarios)."""
    blocks = []
    for snapshot_dir in snapshot_dirs:
        ds = xr.open_dataset(
            ensure_slash(snapshot_dir) + "states_20yrs.nc",
            engine="h5netcdf",
            decode_timedelta=False,
            decode_times=False,
            chunks={},
        )
        tsel = slice(0, 360 * n_year_train, n_days)
        if var == "Eta":
            block = ds[var].isel(time=tsel).stack(space=("j", "i")).T.values
        elif var == "U":
            block = ds[var].isel(k=k, time=tsel).stack(space=("j", "i_g")).T.values
        elif var == "V":
            block = ds[var].isel(k=k, time=tsel).stack(space=("j_g", "i")).T.values
        else:
            block = ds[var].isel(k=k, time=tsel).stack(space=("j", "i")).T.values
        blocks.append(block)
        ds.close()
    return np.hstack(blocks)


def load_var_fom_k(
    snapshot_dir: str,
    var: str,
    n_year_train: int,
    n_year_predict: int,
    n_days: int,
    preproc_dir: str,
    center_opt: str,
    scale: str,
    nx: int = 248,
    ny: int = 248,
    k: int = 0,
    anom: bool = False,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Serial copy of the FOM loader, kept local to avoid importing MPI utils."""
    del scale
    ds = xr.open_dataset(
        ensure_slash(snapshot_dir) + "states_20yrs.nc",
        engine="netcdf4",
        decode_timedelta=False,
        decode_times=False,
        chunks={},
    )
    tsel = slice(0, 360 * (n_year_train + n_year_predict), n_days)
    if var == "Eta":
        var_fom = ds[var].isel(time=tsel).stack(space=("j", "i")).T
    elif var == "U":
        var_fom = ds[var].isel(k=k, time=tsel).stack(space=("j", "i_g")).T
    elif var == "V":
        var_fom = ds[var].isel(k=k, time=tsel).stack(space=("j_g", "i")).T
    else:
        var_fom = ds[var].isel(k=k, time=tsel).stack(space=("j", "i")).T

    if anom:
        center_da = load_center(preproc_dir, var, center_opt, n_days, n_year_train)
        if center_opt == "global_mean":
            var_fom = var_fom - center_da.values
        else:
            center = center_da[0:nx * ny] if var == "Eta" else center_da[k * nx * ny:(k + 1) * nx * ny]
            var_fom = var_fom - center.values[:, None]

    return var_fom, var_fom.unstack("space")


def load_var_fom_lon(
    snapshot_dir: str,
    var: str,
    n_year_train: int,
    n_year_predict: int,
    n_days: int,
    preproc_dir: str,
    center_opt: str,
    scale: str,
    nx: int = 248,
    ny: int = 248,
    nz: int = 31,
    i: int = 124,
    anom: bool = False,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Serial longitude-section FOM loader."""
    del scale, nx, ny, nz
    ds = xr.open_dataset(
        ensure_slash(snapshot_dir) + "states_20yrs.nc",
        engine="netcdf4",
        decode_timedelta=False,
        decode_times=False,
        chunks={},
    )
    tsel = slice(0, 360 * (n_year_train + n_year_predict), n_days)
    if var == "U":
        var_fom = ds[var].isel(i_g=i, time=tsel).stack(space=("k", "j")).T
    elif var == "V":
        var_fom = ds[var].isel(i=i, time=tsel).stack(space=("k", "j_g")).T
    else:
        var_fom = ds[var].isel(i=i, time=tsel).stack(space=("k", "j")).T

    if anom:
        center_da = load_center(preproc_dir, var, center_opt, n_days, n_year_train)
        if center_opt == "global_mean":
            var_fom = var_fom - center_da.values
        else:
            raise NotImplementedError("anom=True for longitude sections is not needed by this evaluator.")

    return var_fom, var_fom.unstack("space")


def transform_and_project_k_forcings(
    snapshot_dirs: list[str],
    var: str,
    n_year_train: int,
    n_days: int,
    center_opt: str,
    preproc_dir: str,
    tr: np.ndarray,
    q_rom: np.ndarray,
    nx: int,
    ny: int,
    k: int = 0,
    anom: bool = False,
    projection_cache: dict[tuple[str, int, bool], tuple[np.ndarray, np.ndarray | float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a ROM trajectory from reduced coordinates to one model level."""
    cache_key = (var, k, anom)
    if projection_cache is not None and cache_key in projection_cache:
        lifted_basis, center_for_add = projection_cache[cache_key]
    else:
        train = load_training_snapshots_at_k(snapshot_dirs, var, n_year_train, n_days, k)
        center_da = load_center(preproc_dir, var, center_opt, n_days, n_year_train)
        if center_opt == "global_mean":
            center = np.asarray(center_da.values).reshape(-1)
            center_for_add = center.item() if center.size == 1 else center
            centered = train - center[:, None] if center.size > 1 else train - center.item()
        else:
            start = 0 if var == "Eta" else k * nx * ny
            stop = nx * ny if var == "Eta" else (k + 1) * nx * ny
            center = np.asarray(center_da[start:stop].values)
            center_for_add = center
            centered = train - center[:, None]
        lifted_basis = centered @ tr
        if projection_cache is not None:
            projection_cache[cache_key] = (lifted_basis, center_for_add)

    rom = lifted_basis @ q_rom
    if not anom:
        if np.isscalar(center_for_add):
            rom = rom + center_for_add
        else:
            rom = rom + center_for_add[:, None]
    rom[np.abs(rom) > 100] = np.nan
    return rom, rom.T.reshape(rom.shape[1], ny, nx)


def as_space_time(field: xr.DataArray | np.ndarray) -> np.ndarray:
    arr = field.values if isinstance(field, xr.DataArray) else np.asarray(field)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1).T


def calc_metrics(fom: xr.DataArray | np.ndarray, rom: xr.DataArray | np.ndarray) -> dict[str, float]:
    fom_st = as_space_time(fom)
    rom_st = as_space_time(rom)
    nt = min(fom_st.shape[1], rom_st.shape[1])
    fom_st = fom_st[:, :nt]
    rom_st = rom_st[:, :nt]
    err = rom_st - fom_st

    rmse_t = np.sqrt(np.nanmean(err**2, axis=0))
    rel_l2_t = np.linalg.norm(np.nan_to_num(err), axis=0) / np.maximum(
        np.linalg.norm(np.nan_to_num(fom_st), axis=0), 1e-14
    )
    clim = np.nanmean(fom_st, axis=1)
    pers = fom_st[:, 0]
    clim_rmse_t = np.sqrt(np.nanmean((fom_st - clim[:, None]) ** 2, axis=0))
    pers_rmse_t = np.sqrt(np.nanmean((fom_st - pers[:, None]) ** 2, axis=0))
    corr_t = np.array(
        [
            np.corrcoef(fom_st[:, i], rom_st[:, i])[0, 1]
            if np.nanstd(fom_st[:, i]) > 0 and np.nanstd(rom_st[:, i]) > 0
            else np.nan
            for i in range(nt)
        ]
    )

    return {
        "rmse_mean": float(np.nanmean(rmse_t)),
        "rmse_final": float(rmse_t[-1]),
        "rel_l2_mean": float(np.nanmean(rel_l2_t)),
        "rel_l2_final": float(rel_l2_t[-1]),
        "corr_mean": float(np.nanmean(corr_t)),
        "corr_final": float(corr_t[-1]),
        "skill_vs_climatology": float(1.0 - np.nanmean(rmse_t) / np.nanmean(clim_rmse_t)),
        "skill_vs_persistence": float(1.0 - np.nanmean(rmse_t) / np.nanmean(pers_rmse_t)),
    }


def save_metrics_csv(rows: list[dict[str, str | float]], save_path: Path) -> None:
    if not rows:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def set_geo_ticks(ax: plt.Axes, nx: int, ny: int) -> None:
    ax.set_xticks(np.linspace(4, nx - 4, 4), [r"$0^\circ$E", r"$20^\circ$E", r"$40^\circ$E", r"$60^\circ$E"])
    ax.set_yticks(np.linspace(4, ny - 4, 4), [r"$15^\circ$N", r"$35^\circ$N", r"$55^\circ$N", r"$75^\circ$N"])


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_depth_values(grid_dir: str, nz: int) -> np.ndarray:
    try:
        import xmitgcm

        grid = xmitgcm.open_mdsdataset(grid_dir, iters=None)
        if "Z" in grid:
            depth = np.asarray(grid["Z"].values)
            if depth.size == nz:
                return depth
    except Exception as exc:
        print(f"Could not load MITgcm depth coordinates from {grid_dir}: {exc}")
    return -np.arange(nz)


def load_lat_values(grid_dir: str, ny: int) -> np.ndarray:
    try:
        import xmitgcm

        grid = xmitgcm.open_mdsdataset(grid_dir, iters=None)
        if "YC" in grid:
            lat = np.asarray(grid["YC"].values)
            lat = lat[:, 0] if lat.ndim == 2 else lat
            if lat.size == ny:
                return lat
    except Exception as exc:
        print(f"Could not load MITgcm latitude coordinates from {grid_dir}: {exc}")
    return np.linspace(15, 75, ny)


def assign_depth_coord(da: xr.DataArray, depth: np.ndarray) -> tuple[xr.DataArray, str]:
    depth_dim = next((dim for dim in ("Z", "k") if dim in da.dims), da.dims[1])
    if da.sizes[depth_dim] == depth.size:
        da = da.assign_coords({depth_dim: depth})
    return da, depth_dim


def assign_lat_coord(da: xr.DataArray, lat: np.ndarray, depth_dim: str) -> tuple[xr.DataArray, str]:
    horizontal_dim = next(dim for dim in da.dims if dim not in ("time", depth_dim))
    if da.sizes[horizontal_dim] == lat.size:
        da = da.assign_coords({horizontal_dim: lat})
    return da, horizontal_dim


def plot_rmse_summary(
    fields_by_scenario: dict[str, SurfaceFields],
    n_days: int,
    outdir: Path,
) -> None:
    variables = [
        ("Eta", "ssh_fom", "ssh_rom"),
        ("T", "sst_fom", "sst_rom"),
        ("speed", "speed_fom", "speed_rom"),
    ]
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 8), sharex=True)
    for ax, (var, fom_attr, rom_attr) in zip(axes, variables):
        for label, fields in fields_by_scenario.items():
            fom = as_space_time(getattr(fields, fom_attr))
            rom = as_space_time(getattr(fields, rom_attr))
            nt = min(fom.shape[1], rom.shape[1])
            rmse = np.sqrt(np.nanmean((rom[:, :nt] - fom[:, :nt]) ** 2, axis=0))
            ax.plot(np.arange(nt) * n_days / 360.0, rmse, lw=1.8, label=label)
        ax.set_ylabel(f"{VAR_META[var]['short']} RMSE [{VAR_META[var]['units']}]")
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("Year")
    fig.suptitle("Surface RMSE by Forcing Scenario")
    savefig(fig, outdir / "surface_rmse_summary.png")


def plot_relative_error_summary(
    fields_by_scenario: dict[str, SurfaceFields],
    n_days: int,
    outdir: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    variables = [
        ("Eta", "ssh_fom", "ssh_rom"),
        ("T", "sst_fom", "sst_rom"),
        ("speed", "speed_fom", "speed_rom"),
    ]
    for ax, (var, fom_attr, rom_attr) in zip(axes, variables):
        for label, fields in fields_by_scenario.items():
            fom = as_space_time(getattr(fields, fom_attr))
            rom = as_space_time(getattr(fields, rom_attr))
            nt = min(fom.shape[1], rom.shape[1])
            err = rom[:, :nt] - fom[:, :nt]
            rel = np.linalg.norm(np.nan_to_num(err), axis=0) / np.maximum(
                np.linalg.norm(np.nan_to_num(fom[:, :nt]), axis=0), 1e-14
            )
            ax.plot(np.arange(nt) * n_days / 360.0, rel, lw=1.6, label=label)
        ax.set_ylabel(f"{VAR_META[var]['short']} rel. L2")
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("Year")
    fig.suptitle("Surface Relative L2 Error by Forcing Scenario")
    savefig(fig, outdir / "relative_l2_summary.png")


def plot_variability_maps(
    fields_by_scenario: dict[str, SurfaceFields],
    var: str,
    fom_attr: str,
    rom_attr: str,
    nx: int,
    ny: int,
    outdir: Path,
) -> None:
    n = len(fields_by_scenario)
    fig, axes = plt.subplots(3, n, figsize=(4.2 * n, 9), squeeze=False)
    variances = []
    for fields in fields_by_scenario.values():
        fom_var = np.nanvar(np.asarray(getattr(fields, fom_attr)), axis=0)
        rom_var = np.nanvar(np.asarray(getattr(fields, rom_attr)), axis=0)
        variances.append((fom_var, rom_var, rom_var - fom_var))
    vmax = np.nanpercentile([v for pair in variances for v in pair[:2]], 98)
    if var == "Eta":
        vmax = 0.02
    diff_lim = np.nanpercentile(np.abs([pair[2] for pair in variances]), 98)
    field_mesh = None
    error_mesh = None

    for col, (label, (fom_var, rom_var, diff)) in enumerate(zip(fields_by_scenario, variances)):
        for row, data, cmap, vmin, vmax_i in [
            (0, fom_var, "viridis", 0, vmax),
            (1, rom_var, "viridis", 0, vmax),
            (2, diff, "RdBu_r", -diff_lim, diff_lim),
        ]:
            mesh = axes[row, col].pcolormesh(data, cmap=cmap, vmin=vmin, vmax=vmax_i)
            axes[row, col].set_title(label if row == 0 else "")
            set_geo_ticks(axes[row, col], nx, ny)
            if row == 2:
                error_mesh = mesh
            else:
                field_mesh = mesh
    for ax, row_label in zip(axes[:, 0], ["MITgcm", "ROM", "ROM - MITgcm"]):
        ax.set_ylabel(row_label, fontsize=12)
    fig.colorbar(field_mesh, ax=axes[:2, :].ravel().tolist(), shrink=0.75, label=f"{VAR_META[var]['short']} variance")
    fig.colorbar(error_mesh, ax=axes[2, :].ravel().tolist(), shrink=0.75, label="Variance error")
    fig.suptitle(f"{VAR_META[var]['short']} Temporal Variance")
    savefig(fig, outdir / f"{var.lower()}_variance_maps.png")


def plot_timeseries_locations(
    fields_by_scenario: dict[str, SurfaceFields],
    var: str,
    fom_attr: str,
    rom_attr: str,
    n_days: int,
    outdir: Path,
    locs: tuple[tuple[int, int], ...] = ((232, 124), (136, 124), (80, 124)),
) -> None:
    rows = len(locs)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 2.4 * rows), sharex=True)
    axes = np.atleast_1d(axes)
    loc_labels = ["High latitude", "Mid latitude", "Low latitude"]
    for ax, (j, i), loc_label in zip(axes, locs, loc_labels):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for idx, (label, fields) in enumerate(fields_by_scenario.items()):
            fom = np.asarray(getattr(fields, fom_attr))
            rom = np.asarray(getattr(fields, rom_attr))
            nt = min(fom.shape[0], rom.shape[0])
            years = np.arange(nt) * n_days / 360.0
            color = colors[idx % len(colors)]
            ax.plot(years, fom[:nt, j, i], color=color, ls="--", lw=1.5, label=f"MITgcm {label}")
            ax.plot(years, rom[:nt, j, i], color=color, ls="-", lw=1.5, label=f"ROM {label}")
        ax.text(0.01, 0.87, loc_label, transform=ax.transAxes, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Year")
    fig.supylabel(VAR_META[var]["label"])
    savefig(fig, outdir / f"{var.lower()}_location_timeseries.png")


def plot_spatial_mean_timeseries(
    fields_by_scenario: dict[str, SurfaceFields],
    var: str,
    fom_attr: str,
    rom_attr: str,
    n_days: int,
    outdir: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (label, fields) in enumerate(fields_by_scenario.items()):
        fom = np.asarray(getattr(fields, fom_attr), dtype=float)
        rom = np.asarray(getattr(fields, rom_attr), dtype=float)
        nt = min(fom.shape[0], rom.shape[0])
        years = np.arange(nt) * n_days / 360.0
        color = colors[idx % len(colors)]

        fom_mean = np.nanmean(np.where(fom[:nt] == 0, np.nan, fom[:nt]), axis=(1, 2))
        rom_mean = np.nanmean(np.where(rom[:nt] == 0, np.nan, rom[:nt]), axis=(1, 2))
        ax.plot(years, fom_mean, label=f"MITgcm {label}", ls="--", color=color, lw=1.6)
        ax.plot(years, rom_mean, label=f"ROM {label}", color=color, lw=1.6)

    ax.set_title(f"Mean {VAR_META[var]['short']}", fontsize=18)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(VAR_META[var]["label"], fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    savefig(fig, outdir / f"{var.lower()}_spatial_mean_timeseries.png")


def plot_monthly_snapshots(
    fields_by_scenario: dict[str, SurfaceFields],
    month_idx: int,
    n_days: int,
    nx: int,
    ny: int,
    outdir: Path,
) -> None:
    snapshots_per_month = int(30 / n_days)
    if snapshots_per_month < 1:
        return
    n = len(fields_by_scenario)
    fig, axes = plt.subplots(3, n, figsize=(4.2 * n, 9), squeeze=False)
    last_meshes = [None, None, None]
    for col, (label, fields) in enumerate(fields_by_scenario.items()):
        fom = np.asarray(fields.ssh_fom)
        rom = np.asarray(fields.ssh_rom)
        nmonths = min(fom.shape[0], rom.shape[0]) // snapshots_per_month
        if nmonths == 0:
            continue
        idx = min(month_idx, nmonths - 1)
        fom_m = fom[: nmonths * snapshots_per_month].reshape(nmonths, snapshots_per_month, ny, nx).mean(axis=1)[idx]
        rom_m = rom[: nmonths * snapshots_per_month].reshape(nmonths, snapshots_per_month, ny, nx).mean(axis=1)[idx]
        err = rom_m - fom_m
        for row, ax, data, cmap, vmin, vmax in [
            (0, axes[0, col], fom_m, "RdBu_r", -1, 1),
            (1, axes[1, col], rom_m, "RdBu_r", -1, 1),
            (2, axes[2, col], err, "RdBu_r", -0.5, 0.5),
        ]:
            mesh = ax.pcolormesh(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(label if row == 0 else "")
            set_geo_ticks(ax, nx, ny)
            last_meshes[0 if vmin == -1 else 2] = mesh
    for ax, row_label in zip(axes[:, 0], ["MITgcm", "ROM", "ROM - MITgcm"]):
        ax.set_ylabel(row_label, fontsize=12)
    fig.colorbar(last_meshes[0], ax=axes[:2, :].ravel().tolist(), shrink=0.75, label="SSH [m]")
    fig.colorbar(last_meshes[2], ax=axes[2, :].ravel().tolist(), shrink=0.75, label="Error [m]")
    fig.suptitle(f"SSH Monthly Mean, Month {month_idx + 1}")
    savefig(fig, outdir / f"ssh_monthly_mean_month{month_idx + 1:02d}.png")


def plot_vertical_temperature_profiles(
    scenarios: list[Scenario],
    training_dirs: list[str],
    q_roms: dict[str, np.ndarray],
    tr: np.ndarray,
    args: argparse.Namespace,
    outdir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    cmap = cmo.thermal if cmo is not None else "inferno"
    lon_projection_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray | float]] = {}
    depth = load_depth_values(args.grid_dir, args.nz)
    lat = load_lat_values(args.grid_dir, args.ny)
    section_data = []
    profile_windows = [
        ("All-time mean", None),
        ("February mean", 1),
        ("August mean", 7),
    ]

    for color, scenario in zip(colors, scenarios):
        t_rom, t_rom_3d = transform_and_project_lon_forcings(
            training_dirs,
            "T",
            args.n_year_train,
            args.n_days,
            args.center_opt,
            args.preproc_dir,
            tr,
            q_roms[scenario.label],
            args.nx,
            args.ny,
            args.nz,
            i=args.vertical_i,
            projection_cache=lon_projection_cache,
        )
        _, t_fom_3d = load_var_fom_lon(
            ensure_slash(scenario.fom_dir),
            "T",
            args.n_year_train,
            args.n_year_predict,
            args.n_days,
            args.preproc_dir,
            args.center_opt,
            args.scale,
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            i=args.vertical_i,
            anom=False,
        )
        t_fom_3d = t_fom_3d.where(t_fom_3d != 0)
        t_fom_3d, depth_dim = assign_depth_coord(t_fom_3d, depth)
        t_fom_3d, horizontal_dim = assign_lat_coord(t_fom_3d, lat, depth_dim)
        t_rom_da = xr.DataArray(t_rom_3d, coords=t_fom_3d.coords, dims=t_fom_3d.dims).where(t_rom_3d != 0)
        for ax, (window_label, month_idx) in zip(axes, profile_windows):
            fom_window = select_month(t_fom_3d, month_idx, args.n_days)
            rom_window = select_month(t_rom_da, month_idx, args.n_days)
            mean_dims = [dim for dim in fom_window.dims if dim != depth_dim]
            profile_fom = fom_window.mean(dim=mean_dims)
            profile_rom = rom_window.mean(dim=mean_dims)
            depth_plot = profile_fom[depth_dim].values
            ax.plot(profile_fom.values, depth_plot, ls="--", color=color, label=f"MITgcm {scenario.label}")
            ax.plot(profile_rom.values, depth_plot, ls="-", color=color, label=f"ROM {scenario.label}")
            ax.set_title(window_label, fontsize=14, pad=15)
            ax.set_xlabel(r"Temperature ($^\circ$C)", fontsize=12, fontweight="bold")
            ax.set_xlim(5, 14)
            ax.set_ylim(-1000, 0)
            ax.grid(True, linestyle="--", alpha=0.5)
        section_data.append((scenario.label, t_fom_3d, t_rom_da, depth_dim, horizontal_dim))

    axes[0].set_ylabel("Depth (m)", fontsize=12, fontweight="bold")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Ocean Temperature Profile", fontsize=16)
    savefig(fig, outdir / "temperature_vertical_profiles.png")
    plot_vertical_sections(section_data, cmap, outdir)


def select_month(da: xr.DataArray, month_idx: int | None, n_days: int) -> xr.DataArray:
    if month_idx is None:
        return da
    snapshots_per_month = int(30 / n_days)
    snapshots_per_year = int(360 / n_days)
    if snapshots_per_month < 1:
        return da
    indices = []
    nt = da.sizes["time"]
    for year_start in range(0, nt, snapshots_per_year):
        start = year_start + month_idx * snapshots_per_month
        stop = start + snapshots_per_month
        if start < nt:
            indices.extend(range(start, min(stop, nt)))
    return da.isel(time=indices)


def transform_and_project_lon_forcings(
    snapshot_dirs: list[str],
    var: str,
    n_year_train: int,
    n_days: int,
    center_opt: str,
    preproc_dir: str,
    tr: np.ndarray,
    q_rom: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    i: int,
    projection_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray | float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cache_key = (var, i)
    if projection_cache is not None and cache_key in projection_cache:
        lifted_basis, center_i = projection_cache[cache_key]
    else:
        blocks = []
        for snapshot_dir in snapshot_dirs:
            ds = xr.open_dataset(
                ensure_slash(snapshot_dir) + "states_20yrs.nc",
                engine="h5netcdf",
                decode_timedelta=False,
                decode_times=False,
                chunks={},
            )
            tsel = slice(0, 360 * n_year_train, n_days)
            if var == "V":
                block = ds[var].isel(i=i, time=tsel).stack(space=("k", "j_g")).T.values
            else:
                block = ds[var].isel(i=i, time=tsel).stack(space=("k", "j")).T.values
            blocks.append(block)
            ds.close()
        train = np.hstack(blocks)
        center = load_center(preproc_dir, var, center_opt, n_days, n_year_train)
        if center_opt == "global_mean":
            center_i = np.asarray(center.values).reshape(-1)
            centered = train - center_i[:, None] if center_i.size > 1 else train - center_i.item()
        else:
            center_i = np.asarray(center.values).reshape(nz, ny, nx)[:, :, i].ravel()
            centered = train - center_i[:, None]
        center_i = center_i.item() if center_i.size == 1 else center_i
        lifted_basis = centered @ tr
        if projection_cache is not None:
            projection_cache[cache_key] = (lifted_basis, center_i)

    rom = lifted_basis @ q_rom + (center_i if np.isscalar(center_i) else center_i[:, None])
    rom[np.abs(rom) > 100] = np.nan
    return rom, rom.T.reshape(rom.shape[1], nz, ny)


def plot_vertical_sections(
    section_data: list[tuple[str, xr.DataArray, xr.DataArray, str, str]],
    cmap: str,
    outdir: Path,
) -> None:
    n = len(section_data)
    fig, axes = plt.subplots(3, n, figsize=(4.4 * n, 9), squeeze=False, sharey=True)
    last_field_mesh = None
    last_error_mesh = None

    for col, (label, t_fom, t_rom, depth_dim, horizontal_dim) in enumerate(section_data):
        fom_mean = t_fom.mean(dim="time")
        rom_mean = t_rom.mean(dim="time")
        err = rom_mean - fom_mean
        for row, data, cmap_i, vmin, vmax in [
            (0, fom_mean, cmap, 0, 22),
            (1, rom_mean, cmap, 0, 22),
            (2, err, "RdBu_r", -0.5, 0.5),
        ]:
            mesh = data.plot(
                ax=axes[row, col],
                x=horizontal_dim,
                y=depth_dim,
                cmap=cmap_i,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
            )
            axes[row, col].set_title(label if row == 0 else "")
            axes[row, col].set_ylabel("Depth (m)" if col == 0 else "")
            axes[row, col].set_xlabel("Latitude (degrees north)" if row == 2 else "")
            if row == 2:
                last_error_mesh = mesh
            else:
                last_field_mesh = mesh
    for ax, row_label in zip(axes[:, 0], ["MITgcm", "ROM", "ROM - MITgcm"]):
        ax.set_ylabel(row_label + "\nDepth (m)", fontsize=12)
    fig.colorbar(last_field_mesh, ax=axes[:2, :].ravel().tolist(), shrink=0.75, label=r"Temperature ($^\circ$C)")
    fig.colorbar(last_error_mesh, ax=axes[2, :].ravel().tolist(), shrink=0.75, label=r"Error ($^\circ$C)")
    fig.suptitle("Vertical Profile of Time-Mean Temperature")
    savefig(fig, outdir / "temperature_vertical_sections_timemean.png")


def find_psi_rom_path(scenario: Scenario, args: argparse.Namespace) -> Path | None:
    """Find a ROM psi file for one scenario, preferring scenario-specific outputs."""
    data_dir = Path(args.data_dir)
    base_suffix = f"{args.center_opt}_{args.scale}_r{args.r}_{args.n_year_train}trainingyrs.nc"
    candidates = [
        data_dir / f"psi_rom_{scenario.label}_{base_suffix}",
        data_dir / f"psi_rom_{Path(scenario.rom_file).stem}_{base_suffix}",
        data_dir / f"{Path(scenario.rom_file).stem}_psi.nc",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    legacy_path = data_dir / f"psi_rom_{base_suffix}"
    if not legacy_path.exists():
        return None

    # The legacy mixed-forcing writer saves this single file from Q_ROM_ after
    # the training loop, so it corresponds to the held-out/test trajectory.
    if "test" in scenario.rom_file.lower() or "test" in scenario.label.lower():
        print(f"Using legacy single ROM psi file for {scenario.label}: {legacy_path}")
        return legacy_path

    print(
        f"Skipping ROM psi for {scenario.label}; found only the legacy single ROM psi file "
        f"({legacy_path}), which is generated from the test trajectory."
    )
    return None


def maybe_plot_psi(scenarios: list[Scenario], args: argparse.Namespace, outdir: Path) -> None:
    plot_data = []
    for scenario in scenarios:
        psi_fom_path = Path(scenario.fom_dir) / "psi_20yrs.nc"
        if not psi_fom_path.exists():
            continue
        psi_rom_path = find_psi_rom_path(scenario, args)
        if psi_rom_path is None:
            continue
        psi = xr.open_dataset(psi_fom_path).isel(
            time=slice(0, 360 * (args.n_year_train + args.n_year_predict), args.n_days)
        )
        psi_rom = xr.open_dataset(psi_rom_path).isel(
            time=slice(0, 360 * (args.n_year_train + args.n_year_predict), args.n_days)
        )
        fom_var = "psi_bt" if "psi_bt" in psi else list(psi.data_vars)[0]
        rom_var = "psi" if "psi" in psi_rom else list(psi_rom.data_vars)[0]
        snapshots_per_month = int(30 / args.n_days)
        fom_monthly = psi[fom_var].coarsen(time=snapshots_per_month, boundary="trim").mean()
        rom_monthly = psi_rom[rom_var].coarsen(time=snapshots_per_month, boundary="trim").mean()
        if np.nanmax(np.abs(fom_monthly.values)) > 1e4:
            fom_monthly = fom_monthly / 1e6
        month_idx = min(args.month_idx, fom_monthly.sizes["time"] - 1, rom_monthly.sizes["time"] - 1)
        plot_data.append((scenario.label, fom_monthly.isel(time=month_idx), rom_monthly.isel(time=month_idx)))

    if not plot_data:
        print("Skipping psi plots; no matching FOM/ROM psi files found.")
        return

    n = len(plot_data)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 6), squeeze=False, sharey=True)
    field_levels = np.arange(-90, 95, 10)
    last_field_mesh = None
    for col, (label, fom, rom) in enumerate(plot_data):
        for row, data, levels in [
            (0, fom, field_levels),
            (1, rom, field_levels),
        ]:
            mesh = axes[row, col].contourf(data, cmap="RdBu_r", levels=levels, extend="both")
            axes[row, col].set_title(label if row == 0 else "")
            last_field_mesh = mesh
    for ax, row_label in zip(axes[:, 0], ["MITgcm", "ROM"]):
        ax.set_ylabel(row_label, fontsize=12)
    fig.colorbar(last_field_mesh, ax=axes.ravel().tolist(), shrink=0.75, label="Streamfunction [Sv]")
    fig.suptitle(f"Barotropic Streamfunction, Month {args.month_idx + 1}")
    savefig(fig, outdir / f"psi_month{args.month_idx + 1:02d}.png")


def load_surface_fields(
    scenario: Scenario,
    training_dirs: list[str],
    q_rom: np.ndarray,
    tr: np.ndarray,
    args: argparse.Namespace,
    projection_cache: dict[tuple[str, int, bool], tuple[np.ndarray, np.ndarray | float]],
) -> SurfaceFields:
    _, sst_rom = transform_and_project_k_forcings(
        training_dirs, "T", args.n_year_train, args.n_days, args.center_opt,
        args.preproc_dir, tr, q_rom, args.nx, args.ny, k=0, projection_cache=projection_cache
    )
    _, ssh_rom = transform_and_project_k_forcings(
        training_dirs, "Eta", args.n_year_train, args.n_days, args.center_opt,
        args.preproc_dir, tr, q_rom, args.nx, args.ny, k=0, projection_cache=projection_cache
    )
    u_rom, _ = transform_and_project_k_forcings(
        training_dirs, "U", args.n_year_train, args.n_days, args.center_opt,
        args.preproc_dir, tr, q_rom, args.nx, args.ny, k=0, projection_cache=projection_cache
    )
    v_rom, _ = transform_and_project_k_forcings(
        training_dirs, "V", args.n_year_train, args.n_days, args.center_opt,
        args.preproc_dir, tr, q_rom, args.nx, args.ny, k=0, projection_cache=projection_cache
    )
    speed_rom = np.sqrt(u_rom**2 + v_rom**2).T.reshape(q_rom.shape[1], args.ny, args.nx)

    _, sst_fom = load_var_fom_k(
        ensure_slash(scenario.fom_dir), "T", args.n_year_train, args.n_year_predict,
        args.n_days, args.preproc_dir, args.center_opt, args.scale, nx=args.nx, ny=args.ny, k=0
    )
    _, ssh_fom = load_var_fom_k(
        ensure_slash(scenario.fom_dir), "Eta", args.n_year_train, args.n_year_predict,
        args.n_days, args.preproc_dir, args.center_opt, args.scale, nx=args.nx, ny=args.ny, k=0
    )
    u_fom, _ = load_var_fom_k(
        ensure_slash(scenario.fom_dir), "U", args.n_year_train, args.n_year_predict,
        args.n_days, args.preproc_dir, args.center_opt, args.scale, nx=args.nx, ny=args.ny, k=0
    )
    v_fom, _ = load_var_fom_k(
        ensure_slash(scenario.fom_dir), "V", args.n_year_train, args.n_year_predict,
        args.n_days, args.preproc_dir, args.center_opt, args.scale, nx=args.nx, ny=args.ny, k=0
    )
    speed_fom = xr.DataArray(
        np.sqrt(u_fom.values**2 + v_fom.values**2),
        coords=u_fom.coords,
        dims=u_fom.dims,
        name="Surface_Speed",
    ).unstack("space")

    return SurfaceFields(sst_fom, sst_rom, ssh_fom, ssh_rom, speed_fom, speed_rom)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="ROM training JSON file to use for evaluator defaults")
    parser.add_argument("--root-dir", default="/scratch/shoshi/soma4/dOpInf_results/")
    parser.add_argument("--data-dir", default=None, help="Directory containing Tr.npy and Q_ROM*.npy")
    parser.add_argument("--preproc-dir", default=None, help="Directory containing center*.nc files")
    parser.add_argument(
        "--training-dirs",
        default=None,
    )
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--scenario-labels", default=None)
    parser.add_argument("--rom-files", default=None)
    parser.add_argument("--n-year-train", type=int, default=None)
    parser.add_argument("--n-year-predict", type=int, default=None)
    parser.add_argument("--n-days", type=int, default=None)
    parser.add_argument("--center-opt", default=None)
    parser.add_argument("--scale", default=None)
    parser.add_argument("--r", type=int, default=None)
    parser.set_defaults(target_ret_energy=None)
    parser.add_argument("--nx", type=int, default=248)
    parser.add_argument("--ny", type=int, default=248)
    parser.add_argument("--nz", type=int, default=31)
    parser.add_argument("--vertical-i", type=int, default=124)
    parser.add_argument("--grid-dir", default="/scratch/shoshi/soma4/grid/")
    parser.add_argument("--month-idx", type=int, default=12, help="Zero-based monthly mean index to plot")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--skip-vertical", action="store_true")
    parser.add_argument("--skip-psi", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.root_dir = ensure_slash(args.root_dir)
    args = apply_config_defaults(args)
    args.n_year_train = 2 if args.n_year_train is None else args.n_year_train
    args.n_year_predict = 0 if args.n_year_predict is None else args.n_year_predict
    args.n_days = 1 if args.n_days is None else args.n_days
    args.center_opt = args.center_opt or "mean"
    args.scale = args.scale or "maxabs"
    args.r = args.r or infer_r_from_data_dir(args.data_dir)
    if args.data_dir is None and args.r is None and args.target_ret_energy is not None:
        raise ValueError("Config uses target_ret_energy, so pass --data-dir with the completed ROM results directory.")
    args.r = args.r or infer_r_from_tr(args.data_dir) or 26
    args.training_dirs = (
        args.training_dirs
        or "/scratch/shoshi/soma4/run_20yrs_tau0.1/,/scratch/shoshi/soma4/run_20yrs_cdscheme/training_snapshots/"
    )
    args.test_dir = args.test_dir or "/scratch/shoshi/soma4/run_20yrs_tau0.3/"
    args.scenario_labels = args.scenario_labels or "tau0.1,tau0.4,tau0.3-test"
    args.rom_files = args.rom_files or "Q_ROM_train_scenario_0.npy,Q_ROM_train_scenario_1.npy,Q_ROM_test3.npy"

    default_group = Path(args.root_dir) / "save_roms" / "taus_1_4"
    args.data_dir = ensure_slash(
        args.data_dir
        or str(default_group / f"{args.center_opt}_{args.scale}_{args.n_days}days_{args.n_year_train}yrs_r{args.r}")
    )
    args.preproc_dir = ensure_slash(args.preproc_dir or str(default_group))
    outdir = Path(args.outdir or args.data_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    training_dirs = split_csv(args.training_dirs)
    labels = split_csv(args.scenario_labels)
    rom_files = split_csv(args.rom_files)
    fom_dirs = training_dirs + [args.test_dir]
    if len(labels) != len(rom_files) or len(labels) != len(fom_dirs):
        raise ValueError(
            "Expected equal counts for scenario labels, ROM files, and FOM directories "
            f"({len(labels)}, {len(rom_files)}, {len(fom_dirs)})."
        )
    scenarios = [
        Scenario(label=label, fom_dir=fom_dir, rom_file=rom_file, is_training=i < len(training_dirs))
        for i, (label, fom_dir, rom_file) in enumerate(zip(labels, fom_dirs, rom_files))
    ]

    print(f"Writing evaluation plots to {outdir}")
    print(f"Using ROM/preprocessing center files from: {args.preproc_dir}")
    print(f"Using ROM trajectory files from: {args.data_dir}")
    tr = np.load(Path(args.data_dir) / "Tr.npy")
    q_roms = {
        scenario.label: np.load(Path(args.data_dir) / scenario.rom_file)[:, : args.n_year_train * int(360 / args.n_days)]
        for scenario in scenarios
    }

    fields_by_scenario = {}
    metric_rows = []
    projection_cache: dict[tuple[str, int, bool], tuple[np.ndarray, np.ndarray | float]] = {}
    for scenario in scenarios:
        print(f"Loading and reconstructing surface fields for {scenario.label}")
        fields = load_surface_fields(scenario, training_dirs, q_roms[scenario.label], tr, args, projection_cache)
        fields_by_scenario[scenario.label] = fields
        for var, fom_attr, rom_attr in [
            ("Eta", "ssh_fom", "ssh_rom"),
            ("T", "sst_fom", "sst_rom"),
            ("speed", "speed_fom", "speed_rom"),
        ]:
            row = {
                "scenario": scenario.label,
                "split": "train" if scenario.is_training else "test",
                "variable": VAR_META[var]["short"],
            }
            row.update(calc_metrics(getattr(fields, fom_attr), getattr(fields, rom_attr)))
            metric_rows.append(row)

    save_metrics_csv(metric_rows, outdir / "surface_metrics.csv")
    plot_rmse_summary(fields_by_scenario, args.n_days, outdir)
    plot_relative_error_summary(fields_by_scenario, args.n_days, outdir)
    plot_variability_maps(fields_by_scenario, "Eta", "ssh_fom", "ssh_rom", args.nx, args.ny, outdir)
    plot_variability_maps(fields_by_scenario, "T", "sst_fom", "sst_rom", args.nx, args.ny, outdir)
    plot_timeseries_locations(fields_by_scenario, "Eta", "ssh_fom", "ssh_rom", args.n_days, outdir)
    plot_spatial_mean_timeseries(fields_by_scenario, "Eta", "ssh_fom", "ssh_rom", args.n_days, outdir)
    plot_timeseries_locations(fields_by_scenario, "T", "sst_fom", "sst_rom", args.n_days, outdir)
    plot_monthly_snapshots(fields_by_scenario, args.month_idx, args.n_days, args.nx, args.ny, outdir)

    if not args.skip_vertical:
        print("Creating vertical temperature diagnostics")
        plot_vertical_temperature_profiles(scenarios, training_dirs, q_roms, tr, args, outdir)

    if not args.skip_psi:
        maybe_plot_psi(scenarios, args, outdir)

    print(f"Done. Metrics: {outdir / 'surface_metrics.csv'}")


if __name__ == "__main__":
    main()
