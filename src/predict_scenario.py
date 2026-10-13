#!/usr/bin/env python3
"""Predict-only ROM rollout from a trained model.h5 for a chosen forcing tau,
reusing the imposed/learned operators (no retraining). Saves Q_ROM_test_<name>.npy
into the run dir.

IC options:
  --ic training     q0 = first column of Qhat.npy (the training-scenario IC)
  --ic <file.npy>   q0 = np.load(file)  (e.g. a projected test-scenario IC)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opinf  # noqa: E402
from forcing import OceanForcing, build_rom_inputs  # noqa: E402
import paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--ic", default="training", help="'training' or path to a q0 .npy")
    p.add_argument("--scenario-name", required=True, help="e.g. run_20yrs_tau0.3")
    p.add_argument("--grid-dir", default=paths.GRID_DIR)
    p.add_argument("--nx", type=int, default=248)
    p.add_argument("--ny", type=int, default=248)
    a = p.parse_args()

    cfg = json.load(open(a.config))
    nyt = cfg["n_year_train"]
    nyp = cfg.get("n_year_predict", 0)
    nd = cfg["n_days"]
    D = Path(a.data_dir)

    model = opinf.models.DiscreteModel.load(str(D / "model.h5"))

    F = OceanForcing(grid_dir=a.grid_dir, nx=a.nx, ny=a.ny,
                     xo=0, yo=14, xc=31, yc=45, dx=0.25, dy=0.25)
    Tmax, _, _ = F.compute_temp_forcing()
    tau = F.compute_wind_seasonal_amplitude(max_val=a.tau)
    inputs = build_rom_inputs(Tmax, tau, nyt + nyp, nd)

    # If the run learned N, its B/C/N assume standardized inputs -> apply the same transform.
    if cfg.get("learn_N") and (D / "input_mean.npy").exists():
        mean = np.load(D / "input_mean.npy")
        scale = np.load(D / "input_scale.npy")
        inputs = (inputs - mean[:, None]) / scale[:, None]
        print("applied input standardization (learn_N run)", flush=True)

    if a.ic == "training":
        q0 = np.load(D / "Qhat.npy")[:, 0]
        ic_desc = "training IC (Qhat[:,0])"
    else:
        q0 = np.load(a.ic)
        ic_desc = f"projected IC from {a.ic}"

    niters = int((nyt + nyp) * 360 / nd)
    print(f"predict tau={a.tau} niters={niters} inputs{inputs.shape} q0{q0.shape} | {ic_desc}", flush=True)
    Q = model.predict(q0, niters=niters, inputs=inputs)
    out = D / f"Q_ROM_test_{a.scenario_name}.npy"
    np.save(out, Q)
    print(f"saved {out}  shape={Q.shape}  all_finite={bool(np.isfinite(Q).all())}", flush=True)


if __name__ == "__main__":
    main()
