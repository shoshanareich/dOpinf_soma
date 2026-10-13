#!/bin/bash
#SBATCH --job-name=rom_train
#SBATCH --output=logs/rom_train_%A.out
#SBATCH --error=logs/rom_train_%A.err
#SBATCH --nodes=3
#SBATCH --ntasks=62
#SBATCH --time=12:00:00

# Train and evaluate one ROM: parallel POD, imposed B/C, Tikhonov sweep
# (src/mixed_rom_forcings.py), then reconstruction, metrics and plots
# (src/eval_rom_forcings.py). Submit from the repository root:
#
#   sbatch run_roms.sh [config.json]      # default: the paper's base run (config_11)
#
# Output: $SOMA_RESULTS_ROOT/save_roms/<dir_extension>/<center>_<scale>_<n_days>days_<years>yrs_r<r>/

nprocs=62

source activate opinf_mixed

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
# Avoid UCX UD transport crashes during MPI_Init on this cluster.
export UCX_TLS="${UCX_TLS:-rc,sm,self}"

root_dir=${SLURM_SUBMIT_DIR:-$PWD}  # submit from the repo root
# Optional config path override: pass as first arg (sbatch run_roms.sh <config>),
# else honor $ROM_CONFIG_FILE, else fall back to the paper's base-run config.
CONFIG_FILE="${1:-${ROM_CONFIG_FILE:-$root_dir/configs_forcings/taus_2_4/config_11.json}}"
# Bare results root; the driver appends save_roms/<dir_extension>/... and the
# evaluator rebuilds the preproc dir from it.
scratch_root_dir=${SOMA_RESULTS_ROOT:-/scratch2/shoshi/soma4/dOpInf_results/}

JOB_ID="${SLURM_ARRAY_JOB_ID:-local}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${SLURM_JOB_ID:-$$}}"
JOB_START_SECONDS=$SECONDS

echo "[timing] Starting job ${TASK_ID} at $(date '+%Y-%m-%d %H:%M:%S')"

export ROM_CONFIG_PATH="$CONFIG_FILE"
export SLURM_ARRAY_JOB_ID="$JOB_ID"
export SLURM_ARRAY_TASK_ID="$TASK_ID"
export ROM_POINTER_DIR="$root_dir"

# Launch the job
MPI_START_SECONDS=$SECONDS
echo "[timing] Starting mixed_rom_forcings.py at $(date '+%Y-%m-%d %H:%M:%S')"
mpiexec -np ${nprocs} python3 $root_dir/src/mixed_rom_forcings.py
MPI_STATUS=$?
MPI_ELAPSED=$((SECONDS - MPI_START_SECONDS))
echo "[timing] Finished mixed_rom_forcings.py at $(date '+%Y-%m-%d %H:%M:%S') after $((MPI_ELAPSED / 60)) min $((MPI_ELAPSED % 60)) sec"
if [ "$MPI_STATUS" -ne 0 ]; then
    echo "Error: mixed_rom_forcings.py failed with exit code $MPI_STATUS"
    exit "$MPI_STATUS"
fi

POINTER_FILE="$root_dir/.results_path_${JOB_ID}_${TASK_ID}"
if [ -f "$POINTER_FILE" ]; then
    RESULTS_DIR=$(cat "$POINTER_FILE")
    echo "Task $TASK_ID saved results to: $RESULTS_DIR"
    rm "$POINTER_FILE"
else
    echo "Error: Pointer file $POINTER_FILE not found for Task $TASK_ID"
    exit 1
fi

cp "$CONFIG_FILE" "$RESULTS_DIR/config.json"

EVAL_START_SECONDS=$SECONDS
echo "[timing] Starting eval_rom_forcings.py at $(date '+%Y-%m-%d %H:%M:%S')"
python3 "$root_dir/src/eval_rom_forcings.py" \
    --config "$CONFIG_FILE" \
    --root-dir "$scratch_root_dir" \
    --data-dir "$RESULTS_DIR" \
    --outdir "$RESULTS_DIR/analysis"
EVAL_STATUS=$?
EVAL_ELAPSED=$((SECONDS - EVAL_START_SECONDS))
echo "[timing] Finished eval_rom_forcings.py at $(date '+%Y-%m-%d %H:%M:%S') after $((EVAL_ELAPSED / 60)) min $((EVAL_ELAPSED % 60)) sec"
if [ "$EVAL_STATUS" -ne 0 ]; then
    echo "Error: eval_rom_forcings.py failed with exit code $EVAL_STATUS"
    exit "$EVAL_STATUS"
fi

JOB_ELAPSED=$((SECONDS - JOB_START_SECONDS))
echo "[timing] Finished job ${TASK_ID} at $(date '+%Y-%m-%d %H:%M:%S') after $((JOB_ELAPSED / 60)) min $((JOB_ELAPSED % 60)) sec"
