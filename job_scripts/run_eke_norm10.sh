#!/bin/bash
#SBATCH --job-name=eke_n10
#SBATCH --output=logs/eke_norm10_%A.out
#SBATCH --error=logs/eke_norm10_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Surface EKE vs tau for the drift-gated constrained ROM (_conserve_norm10, job 31505).
# 1. export ROM (+FOM) surface fields at tau0.2 / 0.4 / 0.5 (tau0.3 already exported by the
#    paper-figures job 31506; ~12 min per scenario, serial). tau0.1 has no rollout -> skipped.
# 2. link the POD-projection surfaces from the paper run (same basis: Tr/Qhat are symlinks).
# 3. analysis/eke_vs_tau.py for tau 0.2-0.5, both anomaly definitions, cache in
#    <run>/analysis/variability/. Figures get the _conserve_norm10 suffix; nothing overwritten.
#
#   sbatch job_scripts/run_eke_norm10.sh

set -u
source activate opinf_mixed
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HDF5_USE_FILE_LOCKING=FALSE

root=${SLURM_SUBMIT_DIR:-$PWD}  # submit from the repo root
paper=${SOMA_RESULTS_ROOT:-/scratch2/shoshi/soma4/dOpInf_results}/save_roms/taus_2_4_sameIC/mean_var_1days_10yrs_r105
D=${paper}_conserve_constrained_norm10
cd "$root" || exit 1
step() { echo "=== $* ($(date '+%F %T'))"; }

step "export surface fields tau0.2, tau0.4, tau0.5"
python3 -u analysis/export_surface_fields.py --config "$D/config.json" --data-dir "$D" \
    --scenarios tau0.2,tau0.4,tau0.5 || exit 1

step "link projection surfaces from the paper run"
for f in "$paper"/surface_fields/proj_surface_*.nc; do
    [ -e "$D/surface_fields/$(basename "$f")" ] || ln -s "$f" "$D/surface_fields/"
done
ls -la "$D/surface_fields/"

step "EKE vs tau (timemean; computes the cache)"
python3 analysis/eke_vs_tau.py --data-dir "$D" --taus 0.2,0.3,0.4,0.5 --train-taus 0.2,0.4 \
    --anomaly timemean --out figs/variability/eke_vs_tau_timemean_conserve_norm10.pdf || exit 1
step "EKE vs tau (deseason, from the cache)"
python3 analysis/eke_vs_tau.py --data-dir "$D" --taus 0.2,0.3,0.4,0.5 --train-taus 0.2,0.4 \
    --anomaly deseason --from-cache --out figs/variability/eke_vs_tau_deseason_conserve_norm10.pdf
step "all done"
