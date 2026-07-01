#!/usr/bin/env python3
"""Compare the variability-experiment options (baseline / bilinearN / varloss /
windows) in reduced (POD) space, against the saved Qhat truth.

For each run and each TRAINING scenario (Qhat truth exists only for training taus):
  - band-retained ratio std_rom/std_true for low / mid / high mode bands
    (high band = eddies; << 1 means damped),
  - high-band late/early ratio (rolling 1-yr std; << 1 means variability collapses
    over the rollout),
  - relative trajectory RMSE ||q_rom - q_true|| / ||q_true|| (fidelity, for the
    variability-vs-accuracy tradeoff).

Higher high-retained and late/early nearer 1 = more eddy variability kept.

Run:
  python3 src/compare_variability_options.py \
      --group-dir /scratch2/.../save_roms/taus_2_4_sameIC_testvar
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_utils import resolve_forcing_config, ensure_slash

OPTIONS = ["baseline", "bilinearN", "varloss", "varloss5", "windows"]
SUBDIRS = ["IC_var_1days_10yrs_r90", "mean_var_1days_10yrs_r105"]


def rolling_std(X: np.ndarray, w: int) -> np.ndarray:
    z = np.zeros((X.shape[0], 1))
    c1 = np.cumsum(np.concatenate([z, X], axis=1), axis=1)
    c2 = np.cumsum(np.concatenate([z, X ** 2], axis=1), axis=1)
    s = c1[:, w:] - c1[:, :-w]
    s2 = c2[:, w:] - c2[:, :-w]
    var = np.clip(s2 / w - (s / w) ** 2, 0, None)
    return np.sqrt(var)


def analyze_run(data_dir: Path):
    """Return list of per-(training)scenario metric rows for one run dir."""
    cfg_path = data_dir / "config.json"
    qhat_path = data_dir / "Qhat.npy"
    if not (cfg_path.exists() and qhat_path.exists()):
        return None
    with open(cfg_path) as f:
        config = json.load(f)
    n_year_train, n_days = config["n_year_train"], config["n_days"]
    fc = resolve_forcing_config(config)
    snapshot_dirs = [ensure_slash(d) for d in fc["snapshot_dirs"]]
    train_taus = fc["train_taus"]
    spy = int(360 / n_days)
    nt = n_year_train * spy

    qhat = np.load(qhat_path)
    r = qhat.shape[0]
    bands = {"low": (0, r // 3), "mid": (r // 3, 2 * r // 3), "high": (2 * r // 3, r)}

    rows = []
    for i, snap_dir in enumerate(snapshot_dirs):
        scen = Path(snap_dir.rstrip("/")).name
        tau = train_taus[i]
        rom_path = data_dir / f"Q_ROM_train_{scen}.npy"
        if not rom_path.exists():
            continue
        q_true = qhat[:, i * nt:(i + 1) * nt]
        q_rom = np.load(rom_path)[:, :nt]
        T = min(q_true.shape[1], q_rom.shape[1])
        q_true, q_rom = q_true[:, :T], q_rom[:, :T]

        std_true, std_rom = q_true.std(axis=1), q_rom.std(axis=1)
        row = {"tau": f"{tau:g}" if tau is not None else scen, "r": r}
        for name, (a, b) in bands.items():
            row[f"{name}_ret"] = float(np.mean(std_rom[a:b]) / max(np.mean(std_true[a:b]), 1e-30))

        w = spy
        rs_rom = rolling_std(q_rom, w)
        rs_true = rolling_std(q_true, w)
        centers = (np.arange(rs_rom.shape[1]) + w // 2) * n_days / 360.0
        first2, last2 = centers <= centers[0] + 2.0, centers >= centers[-1] - 2.0
        ha, hb = bands["high"]
        hi_rom = rs_rom[ha:hb].mean(axis=0)
        hi_true = rs_true[ha:hb].mean(axis=0)
        row["hi_late/early"] = float(np.mean(hi_rom[last2]) / max(np.mean(hi_rom[first2]), 1e-30))
        row["truth_late/early"] = float(np.mean(hi_true[last2]) / max(np.mean(hi_true[first2]), 1e-30))
        row["traj_relRMSE"] = float(np.linalg.norm(q_rom - q_true) / max(np.linalg.norm(q_true), 1e-30))
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group-dir", required=True, help="parent dir holding <option>/<subdir>/ runs")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()
    group = Path(ensure_slash(args.group_dir))
    outdir = Path(args.outdir or (group / "analysis_compare"))
    outdir.mkdir(parents=True, exist_ok=True)

    records = []
    for sub in SUBDIRS:
        center = "IC" if sub.startswith("IC") else "mean"
        for opt in OPTIONS:
            d = group / opt / sub
            rows = analyze_run(d)
            if rows is None:
                print(f"[skip] {opt}/{sub} (no Qhat/config yet)")
                continue
            for row in rows:
                records.append({"center": center, "option": opt, **row})

    if not records:
        print("No runs found.")
        return

    cols = ["center", "option", "tau", "r", "low_ret", "mid_ret", "high_ret",
            "hi_late/early", "truth_late/early", "traj_relRMSE"]
    # print grouped by center then tau so baseline vs options line up
    print("\n=== variability comparison (reduced space, training scenarios) ===")
    print("high_ret / hi_late-early closer to 1 = more eddy variability retained\n")
    w = {c: max(len(c) + 2, 9) for c in cols}
    def fmt(v): return f"{v:.3f}" if isinstance(v, float) else str(v)
    for center in ["IC", "mean"]:
        for tau in sorted({r["tau"] for r in records if r["center"] == center}):
            print(f"--- {center} center, tau {tau} ---")
            print("".join(f"{c:>{w[c]}}" for c in cols))
            base = None
            for opt in OPTIONS:
                rr = [r for r in records if r["center"] == center and r["tau"] == tau and r["option"] == opt]
                if not rr:
                    continue
                row = rr[0]
                print("".join(f"{fmt(row.get(c,'')):>{w[c]}}" for c in cols))
                if opt == "baseline":
                    base = row
            # delta vs baseline for the key metrics
            if base:
                for opt in OPTIONS:
                    if opt == "baseline":
                        continue
                    rr = [r for r in records if r["center"] == center and r["tau"] == tau and r["option"] == opt]
                    if not rr:
                        continue
                    row = rr[0]
                    dh = row["high_ret"] - base["high_ret"]
                    dle = row["hi_late/early"] - base["hi_late/early"]
                    dt = row["traj_relRMSE"] - base["traj_relRMSE"]
                    print(f"    {opt:9s} vs baseline:  high_ret {dh:+.3f}   hi_late/early {dle:+.3f}   traj_relRMSE {dt:+.3f}")
            print()

    csv_path = outdir / "variability_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=cols)
        wri.writeheader()
        for r in records:
            wri.writerow({c: r.get(c, "") for c in cols})
    print(f"saved {csv_path}")

    # bar charts: high_ret and hi_late/early, grouped by option, per (center, tau)
    keys = sorted({(r["center"], r["tau"]) for r in records})
    for metric in ["high_ret", "hi_late/early"]:
        fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(keys)), 4.2))
        xlab = [f"{c}\n{t}" for c, t in keys]
        xpos = np.arange(len(keys))
        nopt = len(OPTIONS)
        width = 0.8 / nopt
        for k, opt in enumerate(OPTIONS):
            vals = []
            for c, t in keys:
                rr = [r for r in records if r["center"] == c and r["tau"] == t and r["option"] == opt]
                vals.append(rr[0][metric] if rr else np.nan)
            ax.bar(xpos + (k - (nopt - 1) / 2) * width, vals, width, label=opt)
        ax.set_xticks(xpos); ax.set_xticklabels(xlab)
        ax.axhline(1.0, color="k", ls="--", lw=0.7)
        ax.set_ylabel(metric); ax.set_title(f"{metric} by option (closer to 1 = better)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = outdir / f"compare_{metric.replace('/', '_')}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
