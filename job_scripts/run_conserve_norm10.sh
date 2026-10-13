#!/bin/bash
#SBATCH --job-name=cons_norm10
#SBATCH --output=logs/conserve_norm10_%A.out
#SBATCH --error=logs/conserve_norm10_%A.err
#SBATCH --nodes=3
#SBATCH --ntasks=62
#SBATCH --time=06:00:00


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

variant() {  # $1 tag, $2 gate horizon in years
    local tag=$1 years=$2
    local suffix=conserve_constrained_$tag
    local D=${src_run}_$suffix
    step "[$tag] refit, drift gate over $years yr"
    mpiexec -np 62 python3 analysis/refit_conservation.py --src-run "$src_run/" --fix constrained \
        --suffix "$suffix" --stability-taus 0.2 0.4 0.5 --stability-years "$years" \
        --stability-norm-factor 1.5 --grid-A -10 0 11 --grid-H 4.3 6 18 \
        || { echo "[$tag] refit FAILED (see above); skipping its figures"; return 1; }

    step "[$tag] extra rollouts"
    for T in 0.5 0.25 0.35 0.45; do
        python3 src/predict_scenario.py --data-dir "$D" --config "$D/config.json" --tau "$T" \
            --ic training --scenario-name "run_10yrspinup_tau$T" || return 1
    done

    step "[$tag] psi"
    local args=()
    for s in train:0.2 train:0.4 test:0.3 test:0.5 test:0.25 test:0.35 test:0.45; do
        local k=${s%%:*} T=${s##*:}
        args+=(--rom-file "Q_ROM_${k}_run_10yrspinup_tau$T.npy"
               --output-label "Q_ROM_${k}_run_10yrspinup_tau$T")
    done
    mpiexec -np 62 python3 analysis/compute_psi_scenario.py --data-dir "$D" --config "$D/config.json" \
        "${args[@]}" || return 1

    step "[$tag] figures"
    python3 analysis/plot_gyre_extremes.py --data-dir "$D" --taus 0.2,0.3,0.4,0.5 \
        --rom-only-taus 0.25,0.35,0.45 --train-taus 0.2,0.4 --year0 3 --nyears 7 --ylim 50,200 \
        --out "figs/gyre_extremes_interp_conserve_$tag.pdf"
    mkdir -p "figs/heat_budget/$suffix"
    python3 analysis/plot_conservation_figs.py --run-dir "$D/" \
        --out-resid "figs/heat_budget/$suffix/conservation_residual_heat_salt.pdf" \
        --out-ops "figs/heat_budget/$suffix/conservation_operators_heat_salt.pdf"

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
}

variant norm10 10
step "all done"
ls -la figs/gyre_extremes_interp_conserve_norm10.pdf
