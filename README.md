# dOpInf SOMA ocean emulator

Code for *Data-Driven Ocean Emulation: Using Operator Inference to Emulate a Double Gyre
Ocean System* (TODO: authors, journal, DOI).

We learn a reduced-order model (ROM) of a 1/4° MITgcm double-gyre ocean (SOMA; Wolfram
et al. 2015) with distributed Operator Inference (dOpInf; Farcas et al. 2025). The model is
a discrete quadratic system with inputs,

```
q_{k+1} = c + A q_k + H (q_k ⊗ q_k) + B u_k + C,      u_k = [Tmax(t), tau(t)]
```

where `q` is the POD-reduced state of the full 3D ocean (U, V, SSH, T, S). The forcing
operators `B` and `C` are not learned: they are exact projections of the known surface
wind-stress, temperature- and salinity-restoring fields. The internal dynamics `c`, `A`, `H`
are learned by Tikhonov-regularized least squares. In the paper ROM, two rows of
`[c A H]` are additionally constrained so that the ROM conserves heat and salt exactly.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Core ROM code: driver `mixed_rom_forcings.py`, parallel POD (`svd.py`), centering/scaling (`shiftscale.py`), forcing projection (`forcing.py`), operator learning and regularization sweep (`learn_rom.py`), evaluation (`eval_rom_forcings.py`), rollouts at any forcing amplitude (`predict_scenario.py`), projection of a held-out scenario onto the basis (`project_scenario.py`), default paths (`paths.py`), config handling (`config_utils.py`) |
| `analysis/` | Conservation-constrained refit (`refit_conservation.py`), heat/salt budgets, field exports, error metrics, and the paper figure scripts |
| `configs/config_11.json` | Configuration of the base run (train τ 0.2 and 0.4, 10 years, daily snapshots, r = 105) |
| `job_scripts/run_roms.sh` | Example SLURM script: train and evaluate one ROM from a config |
| `job_scripts/run_conserve_norm10.sh` | Example SLURM script: the conservation-constrained refit that gives the paper ROM |

## Installation

```bash
conda env create -f environment.yml
conda activate opinf_mixed
```

`mpi4py` must be built against the MPI library of your machine. The paper runs used
62 MPI ranks (Intel MPI 2021.9) on 3 nodes.

## Data and paths

The MITgcm output is not in this repository (TODO: data DOI). The code expects

```
$SOMA_DATA_ROOT/grid/                                  MITgcm grid files
$SOMA_DATA_ROOT/run_10yrspinup_tau{X}/states_20yrs.nc  daily snapshots, X = 0.1 ... 0.5
$SOMA_INPUT_DIR/                                       MITgcm input binaries (forcing files)
```

and writes ROM results to `$SOMA_RESULTS_ROOT/save_roms/`. Set these three environment
variables before running anything; the defaults in `src/paths.py` are the paths on the
authors' cluster. Every script also accepts explicit `--grid-dir`, `--data-dir`, etc.

## Reproducing the paper

The two SLURM scripts are examples written for the authors' cluster (62 MPI ranks on 3 nodes).
Adjust the `#SBATCH` lines for your system, and submit from the repository root, since
log paths are relative to it.

```bash
mkdir -p logs figs

# 1. Base ROM (r = 105): POD, imposed B/C, Tikhonov sweep, evaluation
sbatch job_scripts/run_roms.sh configs/config_11.json

# 2. Paper ROM: conservation-constrained refit of the base run (reuses its POD basis)
sbatch job_scripts/run_conserve_norm10.sh
```

Step 1 is the only one that reads the full snapshot set. Step 2 also writes the extra
rollouts, the barotropic streamfunction, and the gyre-transport figure. The unconstrained
comparison ROM uses the same gate and grid: run step 2's `refit_conservation.py` command
with `--fix none --suffix unconstrained_norm10`.

### Figures

Set `B` to the base run and `R` to the paper ROM:

```bash
B=$SOMA_RESULTS_ROOT/save_roms/taus_2_4_sameIC/mean_var_1days_10yrs_r105
R=${B}_conserve_constrained_norm10
```

`mpiexec` marks the steps that run under MPI. All other commands are serial.

```bash
# POD spectrum and retained energy (property of the shared basis)
python analysis/plot_pod_spectrum.py --data-dir $B/ --r 105 --out figs/svd_decay.pdf

# Field exports for the held-out tau0.3 (needed by the surface, SSH and climatology figures)
python analysis/export_surface_fields.py  --config $R/config.json --data-dir $R --scenarios tau0.3
python analysis/export_vertical_fields.py --config $R/config.json --data-dir $R --vars T,S --scenarios tau0.3
python analysis/export_3d_fields.py --data-dir $R --scenario run_10yrspinup_tau0.3 --skip-fom --rom-vars T,S

# Surface fields, year-10 mean: MITgcm / ROM / error (T, S, SSH, U, V)
python analysis/plot_surface_snapshots.py --data-dir $R --year 10 \
    --err-lims 0.8,0.04,0.15,0.15,0.15 --out figs/surfacesnaps_yr10mean.pdf
# SSH snapshots and domain-mean time series
python analysis/plot_snapshots_timeseries.py --data-dir $R --var Eta --times 100,300,1000,3000 \
    --out figs/ssh_10yrs.pdf
# Zonal-mean temperature and salinity climatology
python analysis/plot_vertical_climatology.py --var T --config $R/config.json --data-dir $R \
    --section-mode zonal --err-lim 0.25 --out-base figs/temperature_climatology_zonalmean_tau0.3-test
python analysis/plot_vertical_climatology.py --var S --config $R/config.json --data-dir $R \
    --section-mode zonal --out-base figs/salinity_climatology_zonalmean_tau0.3-test

# Overturning: deep subpolar cell index at 62N
mpiexec -np 62 python analysis/compute_moc.py --data-dir $R --taus 0.3
python analysis/plot_moc_figure.py --moc-dir $R/analysis/moc --tau 0.3 --lat 62 --min-depth 200 \
    --roll-which fom --lat-line - --out figs/amoc_62N.pdf

# Barotropic streamfunction by tau, January climatology, years 3-10 (psi from step 2)
python analysis/plot_psi_wind_panels.py --data-dir $R --taus 0.2,0.3,0.4,0.5 --error \
    --err-vmax 90 --single-cbar --month 1 --year0 3 --nyears 7 --out figs/psi_jan_avg.pdf

# EKE vs tau (needs surface exports for every tau; the projection-only surfaces
# come from analysis/export_proj_surface.py)
python analysis/export_surface_fields.py --config $R/config.json --data-dir $R --scenarios tau0.2,tau0.4,tau0.5
python analysis/eke_vs_tau.py --data-dir $R --taus 0.2,0.3,0.4,0.5 --train-taus 0.2,0.4 \
    --anomaly timemean --out figs/eke_vs_tau.pdf

# Heat and salt budgets: MITgcm budgets, held-out projection, then the figures
mpiexec -np 62 python analysis/fom_heat_budget.py --taus 0.2 0.3 0.4 0.5
mpiexec -np 62 python analysis/fom_salt_budget.py --taus 0.2 0.3 0.4 0.5
mpiexec -np 62 python src/project_scenario.py --data-dir $B --config $B/config.json \
    --scenario-dir $SOMA_DATA_ROOT/run_10yrspinup_tau0.5/
ln -sf $B/Q_PROJ_run_10yrspinup_tau0.5.npy $R/    # the paper ROM shares the base run's basis
python analysis/heat_budget_depth.py --taus 0.2 0.3 0.4 0.5 --run-dir $R/ --fom-color k \
    --rows full sflux resid --resid-units J --consistent-units \
    --out figs/heat_content_sflux_residual_taus.pdf
python analysis/salt_volume_budget.py --taus 0.2 0.3 0.4 0.5 --run-dir $R/ --fom-color k \
    --consistent-units --out-salt figs/salt_content_sflux_residual_taus.pdf \
    --out-vol figs/volume_eta_residual_taus.pdf

# Compute cost
python analysis/plot_compute_cost.py --out figs/compute_cost.pdf

# Error tables (tau0.3)
python analysis/surface_error_metrics.py   --data-dir $R/ --scenario run_10yrspinup_tau0.3
python analysis/volume_error_metrics.py    --data-dir $R/ --scenario run_10yrspinup_tau0.3
python analysis/depth_avg_error_metrics.py --data-dir $R/ --scenario run_10yrspinup_tau0.3
```

## Citation

TODO

## License

TODO
