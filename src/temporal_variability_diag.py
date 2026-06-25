"""Temporal-variability diagnostic in reduced (POD) space.

The projection-only diagnostic showed the basis reproduces SSH spatial amplitude
(proj/fom ~ 1), so the observed damping is TEMPORAL, not a loss of spatial amplitude.
This script measures it where it actually happens: the reduced coordinates.

For each training scenario, Qhat (the saved POD coefficients) IS the exact reduced-
space ground truth, and Q_ROM is the OpInf rollout. Comparing them per mode and over
time isolates the learned dynamics with no reconstruction / no standing-pattern
contamination:

  - per-mode temporal std (Q_ROM vs Qhat): which modes lose temporal variability.
    Low modes (high energy) = large-scale / seasonal; high modes = faster eddies.
    Damping shows up as ratio rom/true dropping for higher modes.
  - rolling 1-year std by mode band, over time: WHEN the collapse sets in
    (your "year 1-2" vs "year 3").

Cheap: operates only on Qhat.npy and Q_ROM_train_*.npy.

Run (serial, opinf_mixed env):
  python3 src/temporal_variability_diag.py \
      --config   <data-dir>/config.json \
      --data-dir <data-dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_utils import resolve_forcing_config, ensure_slash


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True, help="Dir with Qhat.npy and Q_ROM_train_*.npy")
    p.add_argument("--outdir", default=None, help="Defaults to <data-dir>/analysis/temporal_variability")
    return p.parse_args()


def rolling_std(X: np.ndarray, w: int) -> np.ndarray:
    """Rolling std along time (axis 1) over a window of w samples, via cumulative sums.
    X: (r, T) -> (r, T-w+1)."""
    z = np.zeros((X.shape[0], 1))
    c1 = np.cumsum(np.concatenate([z, X], axis=1), axis=1)
    c2 = np.cumsum(np.concatenate([z, X ** 2], axis=1), axis=1)
    s = c1[:, w:] - c1[:, :-w]
    s2 = c2[:, w:] - c2[:, :-w]
    var = np.clip(s2 / w - (s / w) ** 2, 0, None)
    return np.sqrt(var)


def main() -> None:
    args = parse_args()
    data_dir = Path(ensure_slash(args.data_dir))
    outdir = Path(args.outdir or (data_dir / "analysis" / "temporal_variability"))
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)
    n_year_train = config["n_year_train"]
    n_days = config["n_days"]
    fc = resolve_forcing_config(config)
    snapshot_dirs = [ensure_slash(d) for d in fc["snapshot_dirs"]]
    train_taus = fc["train_taus"]

    steps_per_year = int(360 / n_days)
    nt = n_year_train * steps_per_year

    qhat = np.load(data_dir / "Qhat.npy")  # (r, nt * n_scenarios)
    r = qhat.shape[0]
    bands = {"low (0:r/3)": (0, r // 3), "mid (r/3:2r/3)": (r // 3, 2 * r // 3),
             "high (2r/3:r)": (2 * r // 3, r)}

    print(f"r = {r}, nt per scenario = {nt}, scenarios = {len(snapshot_dirs)}\n")
    summary = []

    for i, snap_dir in enumerate(snapshot_dirs):
        label = f"tau{train_taus[i]:g}" if train_taus[i] is not None else Path(snap_dir.rstrip("/")).name
        scen_name = Path(snap_dir.rstrip("/")).name
        rom_path = data_dir / f"Q_ROM_train_{scen_name}.npy"
        if not rom_path.exists():
            print(f"[skip] {label}: {rom_path} not found")
            continue

        q_true = qhat[:, i * nt:(i + 1) * nt]
        q_rom = np.load(rom_path)[:, :nt]
        T = min(q_true.shape[1], q_rom.shape[1])
        q_true, q_rom = q_true[:, :T], q_rom[:, :T]

        std_true = q_true.std(axis=1)
        std_rom = q_rom.std(axis=1)
        ratio = std_rom / np.maximum(std_true, 1e-30)

        # rolling 1-year std per band, over time
        w = steps_per_year
        rs_true = rolling_std(q_true, w)  # (r, T-w+1)
        rs_rom = rolling_std(q_rom, w)
        centers = (np.arange(rs_true.shape[1]) + w // 2) * n_days / 360.0

        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

        # (A) per-mode temporal std
        ax = axes[0]
        ax.semilogy(np.arange(r), std_true, color="tab:blue", lw=1.2, label="projection (truth)")
        ax.semilogy(np.arange(r), std_rom, color="tab:orange", lw=1.2, label="ROM")
        ax.set_xlabel("POD mode index"); ax.set_ylabel("temporal std of coeff")
        ax.set_title(f"{label}: per-mode temporal variability"); ax.legend()

        # (B) per-mode ratio
        ax = axes[1]
        ax.plot(np.arange(r), ratio, color="tab:red", lw=1.0)
        ax.axhline(1.0, color="k", ls="--", lw=0.7)
        ax.set_ylim(0, 1.4)
        ax.set_xlabel("POD mode index"); ax.set_ylabel("std ratio ROM / truth")
        ax.set_title("variability retained per mode")

        # (C) rolling 1-yr std by band, over time
        ax = axes[2]
        colors = {"low (0:r/3)": "tab:green", "mid (r/3:2r/3)": "tab:purple", "high (2r/3:r)": "tab:brown"}
        for name, (a, b) in bands.items():
            if b <= a:
                continue
            ax.plot(centers, rs_true[a:b].mean(axis=0), color=colors[name], lw=1.3, ls="--")
            ax.plot(centers, rs_rom[a:b].mean(axis=0), color=colors[name], lw=1.3, ls="-",
                    label=f"{name}")
        ax.set_xlabel("Year (window center)"); ax.set_ylabel("rolling 1-yr std (band mean)")
        ax.set_title("dashed = truth, solid = ROM"); ax.legend(fontsize=8)

        fig.tight_layout()
        out_png = outdir / f"temporal_variability_{label}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Processed {label} -> {out_png}")

        row = {"scenario": label}
        for name, (a, b) in bands.items():
            if b <= a:
                row[name] = np.nan
                continue
            row[name] = float(np.mean(std_rom[a:b]) / max(np.mean(std_true[a:b]), 1e-30))
        # late/early variability ratio for ROM (high band) -> shows temporal collapse
        first2 = centers <= (centers[0] + 2.0)
        last2 = centers >= (centers[-1] - 2.0)
        hi_a, hi_b = bands["high (2r/3:r)"]
        hi_rom = rs_rom[hi_a:hi_b].mean(axis=0)
        hi_true = rs_true[hi_a:hi_b].mean(axis=0)
        row["ROM high late/early"] = float(np.mean(hi_rom[last2]) / max(np.mean(hi_rom[first2]), 1e-30))
        row["truth high late/early"] = float(np.mean(hi_true[last2]) / max(np.mean(hi_true[first2]), 1e-30))
        summary.append(row)

    if summary:
        cols = list(summary[0].keys())
        print("\nVariability retained ROM/truth (per mode band) + temporal collapse (late/early):")
        widths = {c: max(len(c) + 2, 10) for c in cols}
        print("".join(f"{c:>{widths[c]}}" for c in cols))
        for row in summary:
            print("".join(
                (f"{row[c]:>{widths[c]}.3f}" if isinstance(row[c], float) else f"{row[c]:>{widths[c]}}")
                for c in cols
            ))
        csv = outdir / "temporal_variability_summary.csv"
        with open(csv, "w") as f:
            f.write(",".join(cols) + "\n")
            for row in summary:
                f.write(",".join(f"{row[c]:.5g}" if isinstance(row[c], float) else str(row[c]) for c in cols) + "\n")
        print(f"\nsaved {csv}")
        print("\nRead: band ratio << 1 (esp. high band) = those modes lose temporal variability;")
        print("      'ROM high late/early' << 1 = eddy variability collapses over the rollout")
        print("      (truth's should stay ~1 if equilibrium/stationary).")
    else:
        print("No scenarios processed.")


if __name__ == "__main__":
    main()
