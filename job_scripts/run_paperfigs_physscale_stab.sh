#!/bin/bash
#SBATCH --job-name=figs_ps_stab
#SBATCH --output=logs/paperfigs_physscale_stab_%A.out
#SBATCH --error=logs/paperfigs_physscale_stab_%A.err
#SBATCH --nodes=3
#SBATCH --ntasks=62
#SBATCH --time=10:00:00

# Paper figures for two refits, written next to the paper versions with a suffix:
#
#   _physscale     mean_var_1days_10yrs_r105_physscale  (physically scaled imposed B/C,
#                  unconstrained learning; refit_conservation.py --fix physical)
#   _conserve_stab mean_var_1days_10yrs_r105_conserve_constrained_stab125  (constrained
#                  refit + stability gate, job 31494; stabdriver picked the byte-identical
#                  model.h5, so stab125 stands for both)
#
# Figures per run (same commands as the paper versions, HANDOFF_2026-09-17.md section 2):
#   amoc_62N_v2, gyre_extremes_interp, {salinity,temperature}_climatology_zonalmean_tau0.3-test,
#   ssh_10yrs_v2, surfacesnaps_yr10mean (year-10 mean)
#
# Every derived product (psi_rom_*.nc, analysis/moc, surface_fields, analysis/computed_fields,
# fields_3d) goes into that run's OWN directory; the paper run and paper figures are only read.
# The zonal-mean T/S panels need the lifted 3D ROM T,S at tau0.3 (fields_3d/rom_3d_*.nc,
# ~55 GB per run, one streaming pass over the training snapshots); FOM T,S are read from
# states_20yrs.nc. A failed run does not stop the other.
#
#   sbatch job_scripts/run_paperfigs_physscale_stab.sh                 # both runs
#   sbatch job_scripts/run_paperfigs_physscale_stab.sh conserve_stab   # one run (physscale | conserve_stab)
#   sbatch job_scripts/run_paperfigs_physscale_stab.sh conserve_norm10 # drift-gated refit (job 31505)
#   sbatch job_scripts/run_paperfigs_physscale_stab.sh <run dir> <_suffix>   # any run

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
base=${SOMA_RESULTS_ROOT:-/scratch2/shoshi/soma4/dOpInf_results}/save_roms/taus_2_4_sameIC/mean_var_1days_10yrs_r105
F=$root/figs
cd "$root" || exit 1
step() { echo "=== $* ($(date '+%F %T'))"; }
# mpiexec with retries: MPI_Init intermittently dies in the UCX UD wire-up
# (ud_ep.c assertion, job 31495) even with UCX_TLS=rc,sm,self.
mpirun_retry() {
    local i
    for i in 1 2 3; do
        mpiexec -np 62 "$@" && return 0
        echo "mpiexec attempt $i failed; retrying in 30 s"; sleep 30
    done
    return 1
}

run_figs() {  # $1 run dir, $2 figure suffix (with leading underscore)
    local D=$1 ext=$2
    # refuse to run against the paper run by mistake
    case "$D" in */mean_var_1days_10yrs_r105) echo "ERROR: D is the paper run"; return 1;; esac
    [ -f "$D/model.h5" ] || { echo "ERROR: no model.h5 in $D"; return 1; }

    # --- 1. rollouts at the ROM-only / extrapolation taus (skipped if present) -----------
    step "[$ext] rollouts"
    for T in 0.5 0.25 0.35 0.45; do
        [ -f "$D/Q_ROM_test_run_10yrspinup_tau$T.npy" ] && continue
        python3 src/predict_scenario.py --data-dir "$D" --config "$D/config.json" --tau "$T" \
            --ic training --scenario-name "run_10yrspinup_tau$T" || return 1
    done

    # --- 2. barotropic streamfunction for the seven rollouts (skipped if all present) ----
    step "[$ext] psi"
    local args=() missing=0
    for s in train:0.2 train:0.4 test:0.3 test:0.5 test:0.25 test:0.35 test:0.45; do
        local k=${s%%:*} T=${s##*:}
        args+=(--rom-file "Q_ROM_${k}_run_10yrspinup_tau$T.npy"
               --output-label "Q_ROM_${k}_run_10yrspinup_tau$T")
        ls "$D"/psi_rom_Q_ROM_${k}_run_10yrspinup_tau${T}_*.nc >/dev/null 2>&1 || missing=1
    done
    if [ $missing = 1 ]; then
        mpirun_retry python3 analysis/compute_psi_scenario.py --data-dir "$D" \
            --config "$D/config.json" "${args[@]}" || return 1
    else
        echo "all psi files present, skipping"
    fi

    # --- 3. meridional overturning, tau0.3 -> $D/analysis/moc/moc_tau0.3.nc --------------
    step "[$ext] moc"
    mpirun_retry python3 analysis/compute_moc.py --data-dir "$D" --taus 0.3 || return 1

    # --- 4. serial exports: surface fields, T/S sections, 3D ROM T/S; tau0.3 ------------
    step "[$ext] export surface"
    python3 -u analysis/export_surface_fields.py --config "$D/config.json" --data-dir "$D" \
        --scenarios tau0.3 || return 1
    step "[$ext] export vertical"
    python3 -u analysis/export_vertical_fields.py --config "$D/config.json" --data-dir "$D" \
        --vars T,S --scenarios tau0.3 || return 1
    step "[$ext] export 3d (ROM T,S)"
    python3 -u analysis/export_3d_fields.py --data-dir "$D" --scenario run_10yrspinup_tau0.3 \
        --skip-fom --rom-vars T,S || return 1

    # --- 5. figures -----------------------------------------------------------------------
    step "[$ext] figures"
    python3 analysis/plot_moc_figure.py --moc-dir "$D/analysis/moc" --tau 0.3 --lat 62 --min-depth 200 \
        --roll-which fom --lat-line - --out "$F/amoc_62N_v2$ext.pdf"
    python3 analysis/plot_gyre_extremes.py --data-dir "$D" --taus 0.2,0.3,0.4,0.5 \
        --rom-only-taus 0.25,0.35,0.45 --train-taus 0.2,0.4 --year0 3 --nyears 7 --ylim 50,200 \
        --out "$F/gyre_extremes_interp$ext.pdf"
    python3 analysis/plot_vertical_climatology.py --var S --config "$D/config.json" --data-dir "$D" \
        --section-mode zonal --out-base "$F/salinity_climatology_zonalmean_tau0.3-test$ext"
    python3 analysis/plot_vertical_climatology.py --var T --config "$D/config.json" --data-dir "$D" \
        --section-mode zonal --out-base "$F/temperature_climatology_zonalmean_tau0.3-test$ext"
    python3 analysis/plot_snapshots_timeseries.py --data-dir "$D" --var Eta --times 100,300,900,2700 \
        --out "$F/ssh_10yrs_v2$ext.pdf"
    # paper convention: mean over year 10 only (not the 10-yr mean)
    python3 analysis/plot_surface_snapshots.py --data-dir "$D" --year 10 \
        --out "$F/surfacesnaps_yr10mean$ext.pdf"
    step "[$ext] done"
    ls -la "$F"/*"$ext".pdf
}

which=${1:-all}
# generic form: <run dir> <suffix>, e.g. called at the end of run_conserve_normgate.sh
if [ "${which:0:1}" = "/" ]; then
    run_figs "${which%/}" "${2:?usage: run_paperfigs_physscale_stab.sh <run dir> <_suffix>}"; rc=$?
    step "all done"; exit $rc
fi
case "$which" in all|physscale) run_figs "${base}_physscale" _physscale;; esac
case "$which" in all|conserve_stab) run_figs "${base}_conserve_constrained_stab125" _conserve_stab;; esac
# not in "all": drift-gated constrained refit (job 31505), run only on request
case "$which" in conserve_norm10) run_figs "${base}_conserve_constrained_norm10" _conserve_norm10;; esac
step "all done"
