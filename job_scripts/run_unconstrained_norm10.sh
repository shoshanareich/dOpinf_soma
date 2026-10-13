#!/bin/bash
#SBATCH --job-name=uncons_norm10
#SBATCH --output=logs/unconstrained_norm10_%A.out
#SBATCH --error=logs/unconstrained_norm10_%A.err
#SBATCH --nodes=3
#SBATCH --ntasks=62
#SBATCH --time=06:00:00

# UNCONSTRAINED twins of the norm10 paper model (_conserve_constrained_norm10, job 31505),
# selected exactly as norm10 was: same refit_conservation.py sweep, same grid
# (reg_A = logspace(-10, 0, 11), reg_H = logspace(4.3, 6, 18)), same yearly-mean-norm drift gate
# (rollouts from Qhat[:,0] at tau 0.2 / 0.4 / 0.5 over 10 yr, yearly-mean ||q|| <= 1.5 x training
# level, no rho(A) gate). Only the hard heat/salt constraints (and, for "none", the B/C) differ,
# so the heat/salt budget comparison against norm10 is apples to apples.
#
#   tag unconstrained_norm10         --fix none: paper-run imposed B/C, plain Tikhonov.
#                                    Partner for the "constrained vs paper" budget figures.
#   tag unconstrained_physwn_norm10  --fix physical_windneutral: the constrained ROM's physical,
#                                    wind-neutral B/C without the constraints (norm10 also sets the
#                                    Tmax column heat to physical). Isolates the constraints.
#
# Each writes <paper run>_<tag>/ (model.h5, Q_ROM train 0.2/0.4 + test 0.3, then the tau0.5 test
# rollout from the training IC, as for norm10) and prints a 20-yr drift check. A variant whose
# gate passes no candidate fails on its own; the other still runs. The paper run, norm10 and all
# figures are only read.
#
# Afterwards (login node, not part of this job), e.g. for the heat figure:
#   python3 analysis/heat_budget_depth.py --taus 0.2 0.3 0.4 0.5 \
#       --run-dir ${base}_conserve_constrained_norm10/ --label "constrained ROM" \
#       --run-dir2 ${base}_unconstrained_norm10/ --label2 "unconstrained ROM" \
#       --fom-color k --consistent-units --no-projection --exact-resid --rows full sflux resid \
#       --resid-units J --resid-ylim -30 30 --out figs/heat_budget/<name>.pdf
# (salt_volume_budget.py needs a --run-dir2 option added for the salt counterpart)
#
#   sbatch job_scripts/run_unconstrained_norm10.sh            # both variants
#   sbatch job_scripts/run_unconstrained_norm10.sh none       # one variant (none | physwn)

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
src_run=${SOMA_RESULTS_ROOT:-/scratch2/shoshi/soma4/dOpInf_results}/save_roms/taus_2_4_sameIC/mean_var_1days_10yrs_r105
cd "$root" || exit 1
step() { echo "=== $* ($(date '+%F %T'))"; }

variant() {  # $1 tag (output suffix), $2 --fix value
    local tag=$1 fix=$2
    local D=${src_run}_$tag
    step "[$tag] refit (--fix $fix), drift gate over 10 yr"
    mpiexec -np 62 python3 analysis/refit_conservation.py --src-run "$src_run/" --fix "$fix" \
        --suffix "$tag" --stability-taus 0.2 0.4 0.5 --stability-years 10 \
        --stability-norm-factor 1.5 --grid-A -10 0 11 --grid-H 4.3 6 18 \
        || { echo "[$tag] refit FAILED (see above)"; return 1; }

    step "[$tag] tau0.5 rollout"
    python3 src/predict_scenario.py --data-dir "$D" --config "$D/config.json" --tau 0.5 \
        --ic training --scenario-name run_10yrspinup_tau0.5 || return 1

    step "[$tag] 20-yr drift check (yearly-mean ||q||, training level ~101.7)"
    python3 - "$D" <<'EOF'
import sys
import numpy as np
import opinf
sys.path.insert(0, 'src')
from rollout_longterm import build_inputs
D = sys.argv[1] + '/'
Qhat = np.load(D + 'Qhat.npy')
m = opinf.models.DiscreteModel.load(D + 'model.h5')
for tau in [0.2, 0.3, 0.4, 0.45, 0.5]:
    with np.errstate(all='ignore'):
        n = np.linalg.norm(m.predict(Qhat[:, 0], niters=20 * 360, inputs=build_inputs(tau, 20, 1)), axis=0)
    n[~np.isfinite(n)] = np.inf
    y = n.reshape(20, 360).mean(axis=1)
    print(f'  tau{tau}: yr 1/5/10/15/20 = ' + ' / '.join(f'{y[k]:.0f}' for k in (0, 4, 9, 14, 19)))
EOF
    step "[$tag] done"
    ls -la "$D"/Q_ROM_*.npy
}

which=${1:-all}
case "$which" in all|none) variant unconstrained_norm10 none;; esac
case "$which" in all|physwn) variant unconstrained_physwn_norm10 physical_windneutral;; esac
step "all done"
