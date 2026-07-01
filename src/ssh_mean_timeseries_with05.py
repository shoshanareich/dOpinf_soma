"""Recreate the SSH domain-mean timeseries figure, including the tau0.5
extrapolation scenario.

The eval driver (eval_rom_forcings.py) only builds the SSH domain-mean
timeseries for the training taus + first test tau. This standalone script reads
the already-saved surface_fields/{fom,rom}_surface_*.nc reconstructions (which
DO include tau0.5) and reproduces plot_spatial_mean_timeseries for any set of
taus, using the same land-masking + area-weighting as the eval code.

Run inside the opinf_mixed conda env.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

RUN_DIR = Path(
    "/scratch2/shoshi/soma4/dOpInf_results/save_roms/"
    "taus_2_4_sameIC/mean_var_1days_10yrs_r105"
)
GRID_DIR = "/scratch/shoshi/soma4/grid/"
NY = NX = 248


def load_cell_area(grid_dir: str, ny: int, nx: int) -> np.ndarray | None:
    """Tracer-point cell area rA (ny, nx); None if unreadable (unweighted mean)."""
    try:
        import xmitgcm

        grid = xmitgcm.open_mdsdataset(grid_dir, iters=None)
        if "rA" in grid:
            area = np.asarray(grid["rA"].values)
            if area.shape == (ny, nx):
                return area
            print(f"rA shape {area.shape} != ({ny}, {nx}); skipping area weighting.")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not load cell area rA from {grid_dir}: {exc}")
    return None


def spatial_mean(field: np.ndarray, area: np.ndarray | None) -> np.ndarray:
    """Area-weighted mean over space (axes 1,2), ignoring land (==0) / non-finite.

    Matches eval_rom_forcings._spatial_mean exactly.
    """
    field = np.asarray(field, dtype=np.float64)
    valid = np.isfinite(field) & (field != 0)
    if area is None:
        return np.nanmean(np.where(valid, field, np.nan), axis=(1, 2))
    # rA is stored as big-endian float32; float32 accumulation over ~5e8-magnitude
    # weights cancels to ~0, so force float64 before summing.
    area = np.asarray(area, dtype=np.float64)
    w = np.where(valid, area[None, :, :], 0.0)
    num = np.nansum(np.where(valid, field, 0.0) * w, axis=(1, 2))
    den = np.sum(w, axis=(1, 2))
    return num / np.where(den == 0, np.nan, den)


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean (low-pass), NaN-aware, same length as input."""
    if window <= 1:
        return x
    s = pd.Series(x)
    return s.rolling(window, center=True, min_periods=1).mean().to_numpy()


def load_eta(run_dir: Path, tau: float, kind: str) -> np.ndarray:
    path = run_dir / "surface_fields" / f"{kind}_surface_run_10yrspinup_tau{tau}.nc"
    with xr.open_dataset(path) as ds:
        return np.asarray(ds["Eta"].values, dtype=float)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=RUN_DIR)
    p.add_argument("--grid-dir", default=GRID_DIR)
    p.add_argument("--n-days", type=int, default=1)
    p.add_argument(
        "--taus", default="0.2,0.3,0.4,0.5",
        help="comma-separated taus to plot",
    )
    p.add_argument("--train-taus", default="0.2,0.4")
    p.add_argument(
        "--smooth-days", type=int, default=30,
        help="low-pass rolling-mean window in days (0/1 = no smoothing)",
    )
    p.add_argument("--no-area", action="store_true",
                   help="disable area weighting (plain unweighted domain mean)")
    p.add_argument("--outfile", default=None)
    args = p.parse_args()

    window = max(1, round(args.smooth_days / args.n_days))

    taus = [t.strip() for t in args.taus.split(",")]
    train = {t.strip() for t in args.train_taus.split(",")}
    if args.no_area:
        area = None
    else:
        area = load_cell_area(args.grid_dir, NY, NX)
        # The saved surface_fields Eta is stored (time, y, x), transposed relative
        # to MITgcm's rA (j, i). Without this .T the weights pair mismatched cells
        # and the area-weighted mean collapses to ~0 (see area vs area.T check).
        if area is not None:
            area = area.T
    print(f"Area weighting: {'on' if area is not None else 'OFF (unweighted)'}")

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, tau in enumerate(taus):
        fom = load_eta(args.run_dir, tau, "fom")
        rom = load_eta(args.run_dir, tau, "rom")
        nt = min(fom.shape[0], rom.shape[0])
        years = np.arange(nt) * args.n_days / 360.0
        color = colors[idx % len(colors)]

        split = "train" if tau in train else "test"
        lbl = f"tau{tau} ({split})"
        fom_mean = rolling_mean(spatial_mean(fom[:nt], area), window)
        rom_mean = rolling_mean(spatial_mean(rom[:nt], area), window)
        ax.plot(years, fom_mean, ls="--", color=color, lw=1.6, label=f"MITgcm {lbl}")
        ax.plot(years, rom_mean, ls="-", color=color, lw=1.6, label=f"ROM {lbl}")

    smooth_note = f"  ({args.smooth_days}-day rolling mean)" if window > 1 else ""
    ax.set_title(f"Mean SSH{smooth_note}", fontsize=18)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Sea surface height [m]", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    out = args.outfile or (args.run_dir / "analysis" / "eta_spatial_mean_timeseries_with05.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
