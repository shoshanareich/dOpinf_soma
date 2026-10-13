#!/bin/bash
#SBATCH --job-name=budget_v2
#SBATCH --output=logs/budget_figs_v2_%A.out
#SBATCH --error=logs/budget_figs_v2_%A.err
#SBATCH --nodes=3
#SBATCH --ntasks=62
#SBATCH --time=06:00:00

# Heat and salt budget figures with consistent units (--consistent-units: row (b) is the
# rate of row (a)'s quantity, conventional W m^-2 / mm yr^-1 on a right-hand axis; row (c)
# is the cumulative residual in row (a)'s unit), for tau 0.2 0.3 0.4 0.5, for
#   the paper run         -> figs/heat_budget/*_taus_v2.pdf  (originals kept)
#   conserve_stab (stab125) -> figs/heat_budget/conserve_constrained_stab125/*_taus.pdf
#
# tau0.5 first needs: MITgcm heat and salt budget passes (heat/salt_budget_fom_tau0.5.npz)
# and the projection-only coefficients Q_PROJ_run_10yrspinup_tau0.5.npy. The projection is
# a property of the POD basis, which both runs share: it is written once into the paper-run
# dir (a NEW file; nothing existing is touched) and symlinked into the stab125 dir, as the
# refit tool does for tau0.2/0.3. Steps whose output exists are skipped.
#
#   sbatch job_scripts/run_budget_figs_v2.sh

set -u
source activate opinf_mixed

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export HDF5_USE_FILE_LOCKING=FALSE
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
# Force (not default) the transport: the UCX UD transport crashes on this cluster.
export UCX_TLS=rc,sm,self

root=${SLURM_SUBMIT_DIR:-$PWD}  # submit from the repo root
P=${SOMA_RESULTS_ROOT:-/scratch2/shoshi/soma4/dOpInf_results}/save_roms/taus_2_4_sameIC/mean_var_1days_10yrs_r105
S=${P}_conserve_constrained_stab125
BD=${SOMA_RESULTS_ROOT:-/scratch2/shoshi/soma4/dOpInf_results}/heat_budget
FH=$root/figs/heat_budget
TAUS=(0.2 0.3 0.4 0.5)
cd "$root" || exit 1
step() { echo "=== $* ($(date '+%F %T'))"; }
# mpiexec with retries: MPI_Init intermittently dies in the UCX UD wire-up (job 31495)
mpirun_retry() {
    local i
    for i in 1 2 3; do
        mpiexec -np 62 "$@" && return 0
        echo "mpiexec attempt $i failed; retrying in 30 s"; sleep 30
    done
    return 1
}

# --- 1. MITgcm budgets, tau0.5 ----------------------------------------------------------
step "FOM heat budget tau0.5"
[ -f "$BD/heat_budget_fom_tau0.5.npz" ] && echo "exists, skipping" || \
    mpirun_retry python3 analysis/fom_heat_budget.py --taus 0.5 || exit 1
step "FOM salt budget tau0.5"
[ -f "$BD/salt_budget_fom_tau0.5.npz" ] && echo "exists, skipping" || \
    mpirun_retry python3 analysis/fom_salt_budget.py --taus 0.5 || exit 1

# --- 2. projection-only coefficients, tau0.5 --------------------------------------------
step "project tau0.5 onto the r105 basis"
Q=$P/Q_PROJ_run_10yrspinup_tau0.5.npy
[ -f "$Q" ] && echo "exists, skipping" || \
    mpirun_retry python3 src/project_scenario.py --data-dir "$P" --config "$P/config.json" \
        --scenario-dir ${SOMA_DATA_ROOT:-/scratch/shoshi/soma4}/run_10yrspinup_tau0.5/ || exit 1
[ -e "$S/Q_PROJ_run_10yrspinup_tau0.5.npy" ] || ln -s "$Q" "$S/Q_PROJ_run_10yrspinup_tau0.5.npy"

# --- 3. figures -------------------------------------------------------------------------
step "figures: paper run (_v2)"
python3 analysis/salt_volume_budget.py --taus "${TAUS[@]}" --run-dir "$P/" --fom-color k \
    --consistent-units --out-salt "$FH/salt_content_sflux_residual_taus_v2.pdf" \
    --out-vol "$FH/volume_eta_residual_taus_v2.pdf"
python3 analysis/heat_budget_depth.py --taus "${TAUS[@]}" --run-dir "$P/" --fom-color k \
    --rows full sflux resid --resid-units J --consistent-units \
    --out "$FH/heat_content_sflux_residual_taus_v2.pdf"

step "figures: conserve_stab"
mkdir -p "$FH/conserve_constrained_stab125"
python3 analysis/salt_volume_budget.py --taus "${TAUS[@]}" --run-dir "$S/" --fom-color k \
    --consistent-units --out-salt "$FH/conserve_constrained_stab125/salt_content_sflux_residual_taus.pdf" \
    --out-vol "$FH/conserve_constrained_stab125/volume_eta_residual_taus.pdf"
python3 analysis/heat_budget_depth.py --taus "${TAUS[@]}" --run-dir "$S/" --fom-color k \
    --rows full sflux resid --resid-units J --consistent-units \
    --out "$FH/conserve_constrained_stab125/heat_content_sflux_residual_taus.pdf"
step "done"
