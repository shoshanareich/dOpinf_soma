#!/bin/bash
#SBATCH --job-name=rom_N
#SBATCH --output=logs/rom_N_%A.out
#SBATCH --error=logs/rom_N_%A.err
#SBATCH --nodes=3
#SBATCH --ntasks=62
#SBATCH --time=12:00:00

# Bilinear state-input operator N(u(x)q) test: equilibrium, train 0.2&0.4, 5yr, mean, r65.
# Writes to dir_extension=taus_2_4_equilibrium_N (FRESH; does not overwrite the
# no-N taus_2_4_equilibrium results).

nprocs=62

source activate opinf_mixed

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export UCX_TLS="${UCX_TLS:-rc,sm,self}"

root_dir=/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/
CONFIG_FILE="$root_dir/configs_forcings/config_N_equil5yr_r65.json"
scratch_root_dir=/scratch2/shoshi/soma4/dOpInf_results/

JOB_ID="${SLURM_ARRAY_JOB_ID:-local}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${SLURM_JOB_ID:-$$}}"
JOB_START_SECONDS=$SECONDS

echo "[timing] Starting N job ${TASK_ID} at $(date '+%Y-%m-%d %H:%M:%S')"
export ROM_CONFIG_PATH="$CONFIG_FILE"
export SLURM_ARRAY_JOB_ID="$JOB_ID"
export SLURM_ARRAY_TASK_ID="$TASK_ID"
export ROM_POINTER_DIR="$root_dir"

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
echo "[timing] Finished N job ${TASK_ID} at $(date '+%Y-%m-%d %H:%M:%S') after $((JOB_ELAPSED / 60)) min $((JOB_ELAPSED % 60)) sec"
