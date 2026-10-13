"""Projection-only diagnostic: basis (truncation) error vs dynamics (ROM) error.

For each TRAINING scenario this compares three SSH curves:
  - FOM          : the full-order surface SSH (MITgcm).
  - projection   : FOM projected onto the POD basis with NO dynamics, reconstructed
                   from the saved Qhat.npy (Qhat = V^T Q is exactly the POD
                   projection of the training snapshots).
  - ROM          : the OpInf rollout reconstructed from Q_ROM_train_*.npy.

Reading the result:
  - FOM  vs projection  -> POD truncation / basis error (controlled by r, energy).
  - projection vs ROM   -> dynamics / regularization error (the overdamping).
If the projection curve already damps relative to the FOM, the problem is the basis
(raise r / retained energy). If the projection stays vibrant but the ROM rollout
damps, the problem is the learned operators / regularization.

Reuses eval_rom_forcings reconstruction so the projection is identical to how the
evaluator builds ssh_fom / ssh_rom. Held-out (test) scenarios are NOT handled here
(their projection needs the full-state MPI projection).

Run (serial, in the opinf_mixed env), e.g.:
  python3 src/projection_only_diag.py \
      --config configs/config_forcings_9.json \
      --data-dir   /scratch2/shoshi/soma4/dOpInf_results/save_roms/taus_1_5_sameIC/mean_var_2days_10yrs_r65 \
      --preproc-dir /scratch2/shoshi/soma4/dOpInf_results/save_roms/taus_1_5_sameIC
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_utils import resolve_forcing_config
from eval_rom_forcings import (
    ensure_slash,
    load_var_fom_k,
    transform_and_project_k_forcings,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="ROM training JSON used for the run")
    p.add_argument("--data-dir", required=True, help="Dir with Tr.npy, Qhat.npy, Q_ROM_train_*.npy")
    p.add_argument("--preproc-dir", required=True, help="Dir with center*.nc files")
    p.add_argument("--outdir", default=None, help="Defaults to <data-dir>/analysis/projection_only")
    p.add_argument("--nx", type=int, default=248)
    p.add_argument("--ny", type=int, default=248)
    return p.parse_args()


def domain_reduce(field: np.ndarray, land_mask: np.ndarray):
    """Return (spatial mean, spatial std) over time, masking land points."""
    masked = np.where(land_mask[:, None], np.nan, field)
    return np.nanmean(masked, axis=0), np.nanstd(masked, axis=0)


def main() -> None:
    args = parse_args()
    data_dir = ensure_slash(args.data_dir)
    preproc_dir = ensure_slash(args.preproc_dir)
    outdir = Path(args.outdir or (Path(data_dir) / "analysis" / "projection_only"))
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)
    n_year_train = config["n_year_train"]
    n_days = config["n_days"]
    center_opt = config["center_opt"]
    scale = config["scale"]
    fc = resolve_forcing_config(config)
    snapshot_dirs = [ensure_slash(d) for d in fc["snapshot_dirs"]]
    train_taus = fc["train_taus"]

    nt = n_year_train * int(360 / n_days)
    years = np.arange(nt) * n_days / 360.0

    tr = np.load(Path(data_dir) / "Tr.npy")
    qhat = np.load(Path(data_dir) / "Qhat.npy")  # (r, nt * n_scenarios), = V^T Q
    n_scen = qhat.shape[1] // nt
    if n_scen != len(snapshot_dirs):
        print(
            f"[warn] Qhat has {n_scen} scenario blocks but config lists "
            f"{len(snapshot_dirs)} training dirs; using config order."
        )

    projection_cache: dict = {}
    summary_rows = []

    for i, snap_dir in enumerate(snapshot_dirs):
        label = f"tau{train_taus[i]:g}" if train_taus[i] is not None else Path(snap_dir.rstrip("/")).name
        scen_name = Path(snap_dir.rstrip("/")).name
        rom_path = Path(data_dir) / f"Q_ROM_train_{scen_name}.npy"
        if not rom_path.exists():
            print(f"[skip] {label}: ROM file not found ({rom_path})")
            continue
        print(f"Processing training scenario {i}: {label}")

        # --- FOM surface SSH (raw, physical units) ---
        ssh_fom, _ = load_var_fom_k(
            snap_dir, "Eta", n_year_train, 0, n_days, preproc_dir,
            center_opt, scale, nx=args.nx, ny=args.ny, k=0, anom=False,
        )
        ssh_fom = np.asarray(ssh_fom.values)  # (space, time)
        ssh_fom = ssh_fom[:, :nt]
        land_mask = np.all(np.abs(ssh_fom) < 1e-12, axis=1)  # land = identically zero

        # --- projection-only: reconstruct from Qhat (no dynamics) ---
        q_proj = qhat[:, i * nt:(i + 1) * nt]
        ssh_proj, _ = transform_and_project_k_forcings(
            snapshot_dirs, "Eta", n_year_train, n_days, center_opt, preproc_dir,
            tr, q_proj, args.nx, args.ny, k=0, anom=False, projection_cache=projection_cache,
        )

        # --- ROM rollout reconstruction ---
        q_rom = np.load(rom_path)[:, :nt]
        ssh_rom, _ = transform_and_project_k_forcings(
            snapshot_dirs, "Eta", n_year_train, n_days, center_opt, preproc_dir,
            tr, q_rom, args.nx, args.ny, k=0, anom=False, projection_cache=projection_cache,
        )

        fom_mean, fom_std = domain_reduce(ssh_fom, land_mask)
        proj_mean, proj_std = domain_reduce(ssh_proj, land_mask)
        rom_mean, rom_std = domain_reduce(ssh_rom, land_mask)

        # --- plot: domain-mean SSH (top) + spatial std of SSH (bottom) ---
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(years, fom_mean, color="black", lw=1.0, label="FOM")
        axes[0].plot(years, proj_mean, color="tab:blue", lw=1.2, label="projection-only")
        axes[0].plot(years, rom_mean, color="tab:orange", lw=1.2, label="ROM")
        axes[0].set_ylabel("domain-mean SSH [m]")
        axes[0].set_title(f"Projection vs ROM — training {label}")
        axes[0].legend(loc="best")

        axes[1].plot(years, fom_std, color="black", lw=1.0, label="FOM")
        axes[1].plot(years, proj_std, color="tab:blue", lw=1.2, label="projection-only")
        axes[1].plot(years, rom_std, color="tab:orange", lw=1.2, label="ROM")
        axes[1].set_ylabel("spatial std of SSH [m]\n(eddy/structure amplitude)")
        axes[1].set_xlabel("Year")
        axes[1].legend(loc="best")

        fig.tight_layout()
        out_png = outdir / f"projection_only_{label}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  saved {out_png}")

        # --- quantitative damping summary (last year of the window) ---
        last = years >= (years[-1] - 1.0)

        def ratio(a, b):
            db = np.nanmean(b[last])
            return float(np.nanmean(a[last]) / db) if db > 0 else np.nan

        summary_rows.append({
            "scenario": label,
            "fom_std_lastyr": float(np.nanmean(fom_std[last])),
            "proj_std_lastyr": float(np.nanmean(proj_std[last])),
            "rom_std_lastyr": float(np.nanmean(rom_std[last])),
            "proj/fom (basis)": ratio(proj_std, fom_std),
            "rom/proj (dynamics)": ratio(rom_std, proj_std),
            "rom/fom (total)": ratio(rom_std, fom_std),
        })

    if summary_rows:
        cols = list(summary_rows[0].keys())
        csv_path = outdir / "projection_only_summary.csv"
        with open(csv_path, "w") as f:
            f.write(",".join(cols) + "\n")
            for row in summary_rows:
                f.write(",".join(f"{row[c]:.4g}" if isinstance(row[c], float) else str(row[c]) for c in cols) + "\n")
        print(f"\nSpatial-std ratios over the final year (variability retained = closer to 1):")
        header = f"{'scenario':<12}{'proj/fom':>12}{'rom/proj':>12}{'rom/fom':>12}"
        print(header)
        print("-" * len(header))
        for row in summary_rows:
            print(f"{row['scenario']:<12}{row['proj/fom (basis)']:>12.3f}"
                  f"{row['rom/proj (dynamics)']:>12.3f}{row['rom/fom (total)']:>12.3f}")
        print(f"\nInterpretation: proj/fom << 1 -> basis/truncation limits variability (raise r);")
        print(f"                rom/proj << 1 -> dynamics/regularization damps it (operator/reg fix).")
        print(f"saved {csv_path}")
    else:
        print("No training scenarios processed.")


if __name__ == "__main__":
    main()
