"""
Default file-system locations for the SOMA dOpInf pipeline.

Every script takes its default paths from here, so the code runs on another
machine by setting environment variables instead of editing scripts:

    SOMA_DATA_ROOT     MITgcm output: grid/ and run_10yrspinup_tau{X}/states_20yrs.nc
    SOMA_RESULTS_ROOT  ROM output (save_roms/, heat_budget/)
    SOMA_INPUT_DIR     MITgcm input binaries (SSS relaxation, wind stress)

Command-line arguments (--grid-dir, --data-dir, ...) still override these.
"""
import os


def _dir(env, default):
    p = os.environ.get(env, default)
    return p if p.endswith('/') else p + '/'


DATA_ROOT = _dir('SOMA_DATA_ROOT', '/scratch/shoshi/soma4/')
RESULTS_ROOT = _dir('SOMA_RESULTS_ROOT', '/scratch2/shoshi/soma4/dOpInf_results/')
INPUT_DIR = _dir('SOMA_INPUT_DIR', '/home/shoshi/MITgcm_c68q/soma4/input_binaries/')

GRID_DIR = DATA_ROOT + 'grid/'
FOM_PREFIX = DATA_ROOT + 'run_10yrspinup_tau'   # + '{tau}/states_20yrs.nc'
SAVE_ROMS = RESULTS_ROOT + 'save_roms/'
BUDGET_DIR = RESULTS_ROOT + 'heat_budget/'

# Unconstrained r=105 run (configs/config_11.json). The paper
# ROM (_conserve_constrained_norm10) is refit from it by refit_conservation.py.
BASE_RUN = SAVE_ROMS + 'taus_2_4_sameIC/mean_var_1days_10yrs_r105/'
