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
| `src/` | Core ROM code: driver `mixed_rom_forcings.py`, parallel POD (`svd.py`), centering/scaling (`shiftscale.py`), forcing projection (`forcing.py`), operator learning and regularization sweep (`learn_rom.py`), evaluation (`eval_rom_forcings.py`), rollouts (`predict_scenario.py`, `rollout_longterm.py`), default paths (`paths.py`) |
| `analysis/` | Conservation-constrained refit (`refit_conservation.py`), heat/salt budgets, field exports, error metrics, and the paper figure scripts |
| `configs_forcings/taus_2_4/config_11.json` | Configuration of the base run (train τ 0.2 and 0.4, 10 years, daily snapshots, r = 105) |
| `run_roms_taus24.sh` | SLURM script that trains the base run |
| `job_scripts/` | SLURM scripts for the constrained refit, the unconstrained comparison, and the figures |

The other scripts in `src/` and the repository root are earlier experiments, kept for
reference. They are not needed to reproduce the paper.

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

Submit all job scripts from the repository root (their log paths are relative to it):

```bash
mkdir -p logs figs

# 1. Base ROM: POD, imposed B/C, Tikhonov sweep, evaluation  (~62 ranks)
TAUS24_CONFIGS=11 sbatch --array=0 run_roms_taus24.sh

# 2. Paper ROM: conservation-constrained refit of the base run, selected with a
#    10-year norm-drift gate; also writes the extra rollouts and the gyre-transport figure
sbatch job_scripts/run_conserve_norm10.sh

# 3. Unconstrained comparison ROMs, selected with the same gate and grid
sbatch job_scripts/run_unconstrained_norm10.sh

# 4. Figures for the paper ROM
R=$SOMA_RESULTS_ROOT/save_roms/taus_2_4_sameIC/mean_var_1days_10yrs_r105_conserve_constrained_norm10
sbatch job_scripts/run_paperfigs_physscale_stab.sh "$R" _conserve_norm10
sbatch job_scripts/run_eke_norm10.sh
sbatch job_scripts/run_budget_figs_v2.sh
```

Step 2 reuses the base run's POD basis and imposed operators, so only step 1 reads the
full snapshot set.

### Figures

| Figure | Script |
|---|---|
| POD spectrum / retained energy | `analysis/plot_pod_spectrum.py` |
| Surface fields, year-10 mean (MITgcm / ROM / error) | `analysis/plot_surface_snapshots.py --year 10` |
| SSH snapshots + domain-mean time series | `analysis/plot_snapshots_timeseries.py --var Eta` |
| Zonal-mean T and S climatology | `analysis/plot_vertical_climatology.py --var T\|S --section-mode zonal` |
| Overturning (62°N deep-cell index) | `analysis/compute_moc.py`, `analysis/plot_moc_figure.py` |
| Barotropic streamfunction by τ | `analysis/compute_psi_scenario.py`, `analysis/plot_psi_wind_panels.py` |
| Gyre transport vs τ | `analysis/plot_gyre_extremes.py` |
| EKE vs τ | `analysis/eke_vs_tau.py` |
| Heat and salt budgets | `analysis/fom_heat_budget.py`, `analysis/fom_salt_budget.py`, `analysis/heat_budget_depth.py`, `analysis/salt_volume_budget.py`, `analysis/plot_conservation_figs.py` |
| Long-term stability | `analysis/plot_longterm_stability.py` |
| Compute cost | `analysis/plot_compute_cost.py` |
| Error tables | `analysis/surface_error_metrics.py`, `analysis/volume_error_metrics.py`, `analysis/depth_avg_error_metrics.py` |

The exact arguments for each figure are in the job scripts above.

## Citation

TODO

## License

TODO
