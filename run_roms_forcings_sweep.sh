#!/bin/bash
#SBATCH --job-name=rom_forcings_sweep
#SBATCH --output=logs/forcings_sweep_%A_%a.out
#SBATCH --error=logs/forcings_sweep_%A_%a.err
#SBATCH --nodes=4
#SBATCH --ntasks=62
#SBATCH --time=12:00:00
#SBATCH --array=1-2%2

source activate opinf_mixed

root_dir=/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/
configs_dir="$root_dir/configs_forcings"
scratch_root_dir=/scratch/shoshi/soma4/dOpInf_results/

mkdir -p "$root_dir/logs"

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
JOB_START_SECONDS=$SECONDS

CONFIG_FILE="$configs_dir/config_forcings_${TASK_ID}.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "[timing] Starting forcing sweep task ${TASK_ID} at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Config: $CONFIG_FILE"
python3 -c "import json; d=json.load(open('$CONFIG_FILE')); print('Config summary:', {k: d.get(k) for k in ['n_year_train', 'n_year_predict', 'n_days', 'center_opt', 'scale', 'r', 'target_ret_energy', 'dir_extension']})"

export ROM_CONFIG_PATH="$CONFIG_FILE"
export SLURM_ARRAY_JOB_ID="$JOB_ID"
export SLURM_ARRAY_TASK_ID="$TASK_ID"
export ROM_POINTER_DIR="$root_dir"

POINTER_FILE="$root_dir/.results_path_${JOB_ID}_${TASK_ID}"
rm -f "$POINTER_FILE"

MPI_START_SECONDS=$SECONDS
echo "[timing] Starting mixed_rom_forcings.py at $(date '+%Y-%m-%d %H:%M:%S')"
mpiexec -np 62 python3 "$root_dir/src/mixed_rom_forcings.py"
MPI_STATUS=$?
MPI_ELAPSED=$((SECONDS - MPI_START_SECONDS))
echo "[timing] Finished mixed_rom_forcings.py at $(date '+%Y-%m-%d %H:%M:%S') after $((MPI_ELAPSED / 60)) min $((MPI_ELAPSED % 60)) sec"
if [ "$MPI_STATUS" -ne 0 ]; then
    echo "Error: mixed_rom_forcings.py failed with exit code $MPI_STATUS"
    exit "$MPI_STATUS"
fi

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
echo "[timing] Finished forcing sweep task ${TASK_ID} at $(date '+%Y-%m-%d %H:%M:%S') after $((JOB_ELAPSED / 60)) min $((JOB_ELAPSED % 60)) sec"
