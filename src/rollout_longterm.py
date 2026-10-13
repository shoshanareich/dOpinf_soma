"""Extend a trained OpInf ROM to a long (default 60-yr) autonomous rollout for the
long-term-stability figure.

The saved Q_ROM_*.npy in a result dir only span the training horizon (here 10 yr,
n_year_predict=0). This re-loads model.h5 and re-rolls each scenario from its ORIGINAL
initial condition (the first column of the saved Q_ROM_*), so the first 10 yr are
bit-identical to what was saved and the remainder is the extrapolated rollout.

Inputs u_k = [Tmax(t), tau4(t)] are analytic (two 12-month cosines), grid-independent,
and match forcing.build_rom_inputs / OceanForcing exactly. No input standardization is
applied (this run has no bilinear N and no input_mean.npy/input_scale.npy); if those
files are present they are used.

Outputs Q_ROM_{n_years}yr_tau{tau}.npy (shape (r, n_years*360/n_days)) per scenario.

Run (serial, opinf_mixed env):
  python3 src/rollout_longterm.py --data-dir <dir> [--n-years 60]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import opinf


def temp_forcing(Tmax_low=25.0, Tmax_high=30.0):
    """12-month seasonal SST-relaxation max; matches OceanForcing.compute_temp_forcing."""
    mo = np.linspace(0, 2 * np.pi, 12)
    return -(Tmax_high - Tmax_low) / 2 * np.cos(mo) + (Tmax_high + Tmax_low) / 2


def wind_amplitude(tau, min_val=0.1):
    """12-month seasonal wind amplitude; matches OceanForcing.compute_wind_seasonal_amplitude."""
    mo = np.linspace(0, 2 * np.pi, 12)
    k = (tau - min_val) / 2
    return k * np.cos(mo) + (min_val + k)


def build_inputs(tau, n_years, n_days):
    """Matches forcing.build_rom_inputs (Tmax, tau4, n_years, n_days) -> (2, n_years*360/n_days)."""
    reps = int(30 / n_days)
    Tmax = np.tile(np.repeat(temp_forcing(), reps), n_years)
    tau4 = np.tile(np.repeat(wind_amplitude(tau), reps), n_years)
    return np.vstack((Tmax, tau4))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-years", type=int, default=60)
    args = p.parse_args()

    d = Path(args.data_dir)
    with open(d / "config.json") as f:
        cfg = json.load(f)
    n_days = cfg["n_days"]

    model = opinf.models.DiscreteModel.load(str(d / "model.h5"))

    # optional input standardization (identity when files absent / no bilinear N)
    im_path, is_path = d / "input_mean.npy", d / "input_scale.npy"
    input_mean = np.load(im_path) if im_path.exists() else None
    input_scale = np.load(is_path) if is_path.exists() else None

    def standardize(u):
        if input_mean is None:
            return u
        return (u - input_mean[:, None]) / input_scale[:, None]

    # map each scenario tau -> the saved Q_ROM file that holds its original IC
    scenarios = {}
    for f in sorted(d.glob("Q_ROM_train_*.npy")) + sorted(d.glob("Q_ROM_test_*.npy")):
        tau = float(f.stem.split("tau")[-1])
        scenarios[tau] = f

    niters = args.n_years * int(360 / n_days)
    for tau, f in scenarios.items():
        q0 = np.load(f)[:, 0]
        u = standardize(build_inputs(tau, args.n_years, n_days))
        Q = model.predict(q0, niters=niters, inputs=u)
        finite = np.isfinite(Q).all()
        out = d / f"Q_ROM_{args.n_years}yr_tau{tau:g}.npy"
        np.save(out, Q)
        print(f"tau{tau:g}: {Q.shape} finite={finite} max|q|={np.abs(Q).max():.3e} -> {out.name}")


if __name__ == "__main__":
    main()
