#!/bin/bash
#SBATCH --job-name=rom_testvar
#SBATCH --output=logs/testvar_%A_%a.out
#SBATCH --error=logs/testvar_%A_%a.err
#SBATCH --nodes=3
#SBATCH --ntasks=62
#SBATCH --time=12:00:00
#SBATCH --array=0-7%4

# Variability experiments on transient/same-IC taus_2_4 data (train 0.2 & 0.4):
#   option 1 bilinear N, option 2 variance loss, option 3 random short windows,
#   + baseline, each x {IC r90, mean r105}. Each config has its own dir_extension
#   subdir (taus_2_4_sameIC_testvar/<option>) so runs are race-free and concurrent.

nprocs=62
source activate opinf_mixed

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export UCX_TLS="${UCX_TLS:-rc,sm,self}"

root_dir=/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/
cfg_dir="$root_dir/configs_forcings/taus_2_4_sameIC_testvar"
# Bare results root; the evaluator rebuilds the preproc dir from dir_extension.
scratch_root_dir=/scratch2/shoshi/soma4/dOpInf_results/

# One config per array index
CONFIGS=(
    "$cfg_dir/config_baseline_IC.json"
    "$cfg_dir/config_baseline_mean.json"
    "$cfg_dir/config_bilinearN_IC.json"
    "$cfg_dir/config_bilinearN_mean.json"
    "$cfg_dir/config_varloss_IC.json"
    "$cfg_dir/config_varloss_mean.json"
    "$cfg_dir/config_windows_IC.json"
    "$cfg_dir/config_windows_mean.json"
)
CONFIG_FILE="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$CONFIG_FILE" ] || [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: no config for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi
echo "[timing] array $SLURM_ARRAY_TASK_ID -> $CONFIG_FILE at $(date '+%Y-%m-%d %H:%M:%S')"

JOB_ID="${SLURM_ARRAY_JOB_ID:-local}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${SLURM_JOB_ID:-$$}}"

export ROM_CONFIG_PATH="$CONFIG_FILE"
export SLURM_ARRAY_JOB_ID="$JOB_ID"
export SLURM_ARRAY_TASK_ID="$TASK_ID"
export ROM_POINTER_DIR="$root_dir"

# ---- train ----
MPI_START=$SECONDS
echo "[timing] Starting mixed_rom_forcings.py at $(date '+%Y-%m-%d %H:%M:%S')"
mpiexec -np ${nprocs} python3 $root_dir/src/mixed_rom_forcings.py
MPI_STATUS=$?
echo "[timing] Finished mixed_rom_forcings.py after $(( (SECONDS-MPI_START)/60 )) min"
if [ "$MPI_STATUS" -ne 0 ]; then
    echo "Error: mixed_rom_forcings.py failed with exit code $MPI_STATUS"
    exit "$MPI_STATUS"
fi

POINTER_FILE="$root_dir/.results_path_${JOB_ID}_${TASK_ID}"
if [ -f "$POINTER_FILE" ]; then
    RESULTS_DIR=$(cat "$POINTER_FILE")
    echo "Task $TASK_ID results: $RESULTS_DIR"
    rm "$POINTER_FILE"
else
    echo "Error: Pointer file $POINTER_FILE not found for Task $TASK_ID"
    exit 1
fi
cp "$CONFIG_FILE" "$RESULTS_DIR/config.json"

# ---- eval (produces eta_spatial_mean_timeseries, psi_month13,
#            ssh_monthly_mean_month13, temperature_vertical_sections_timemean, etc.) ----
EVAL_START=$SECONDS
echo "[timing] Starting eval_rom_forcings.py at $(date '+%Y-%m-%d %H:%M:%S')"
python3 "$root_dir/src/eval_rom_forcings.py" \
    --config "$CONFIG_FILE" \
    --root-dir "$scratch_root_dir" \
    --data-dir "$RESULTS_DIR" \
    --outdir "$RESULTS_DIR/analysis" \
    --skip-field-save
EVAL_STATUS=$?
echo "[timing] Finished eval_rom_forcings.py after $(( (SECONDS-EVAL_START)/60 )) min"
if [ "$EVAL_STATUS" -ne 0 ]; then
    echo "Error: eval_rom_forcings.py failed with exit code $EVAL_STATUS"
    exit "$EVAL_STATUS"
fi
echo "[timing] array $SLURM_ARRAY_TASK_ID done at $(date '+%Y-%m-%d %H:%M:%S')"
