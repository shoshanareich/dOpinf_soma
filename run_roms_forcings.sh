#!/bin/bash
#SBATCH --job-name=rom_forcings
#SBATCH --output=logs/forcings.out
#SBATCH --error=logs/forcings.err
#SBATCH --nodes=4
#SBATCH --ntasks=62
#SBATCH --time=12:00:00
###SBATCH --array=0-9%3       # Run 3 jobs at a time


source activate opinf_mixed

root_dir=/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/
#CONFIG_FILE="$root_dir/configs/config_${SLURM_ARRAY_TASK_ID}.json"
CONFIG_FILE="$root_dir/configs_forcings/my_experiment.json"
scratch_root_dir=/scratch/shoshi/soma4/dOpInf_results/


# Extract primary variables
n_year_train=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['n_year_train'])")
center_opt=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['center_opt'])")
scale=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['scale'])")
n_days=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['n_days'])")

# Safely extract optional variables (returns empty string if key is missing or null)
r=$(python3 -c "import json; d = json.load(open('$CONFIG_FILE')); v = d.get('r'); print(v if v is not None else '')")
energy=$(python3 -c "import json; d = json.load(open('$CONFIG_FILE')); v = d.get('target_ret_energy'); print(v if v is not None else '')")

# Determine the directory suffix
if [ -n "$r" ]; then
    SUFFIX="r${r}"
elif [ -n "$energy" ]; then
    SUFFIX="energy${energy}"
else
    SUFFIX="unknown"
fi



JOB_ID="${SLURM_ARRAY_JOB_ID:-local}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${SLURM_JOB_ID:-$$}}"
JOB_START_SECONDS=$SECONDS

echo "[timing] Starting job ${TASK_ID} at $(date '+%Y-%m-%d %H:%M:%S')"

# Link config for Python
#ln -sf ${CONFIG_FILE} $root_dir/config.json
export ROM_CONFIG_PATH="$CONFIG_FILE"
export SLURM_ARRAY_JOB_ID="$JOB_ID"
export SLURM_ARRAY_TASK_ID="$TASK_ID"
export ROM_POINTER_DIR="$root_dir"

# Launch the job
MPI_START_SECONDS=$SECONDS
echo "[timing] Starting mixed_rom_forcings.py at $(date '+%Y-%m-%d %H:%M:%S')"
mpiexec -np 62 python3 $root_dir/src/mixed_rom_forcings.py
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

    # Now you can use $RESULTS_DIR for post-processing or moving files
    # cp some_analysis.png "$RESULTS_DIR/"

    # Clean up the pointer file
    rm "$POINTER_FILE"
else
    echo "Error: Pointer file $POINTER_FILE not found for Task $TASK_ID"
    exit 1
fi

#RESULTS_DIR="${scratch_dir}/${center_opt}_${scale}_${n_days}days_${n_year_train}yrs_${SUFFIX}"
#mkdir -p "$RESULTS_DIR"

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

# Record finish time
JOB_ELAPSED=$((SECONDS - JOB_START_SECONDS))
echo "[timing] Finished job ${TASK_ID} at $(date '+%Y-%m-%d %H:%M:%S') after $((JOB_ELAPSED / 60)) min $((JOB_ELAPSED % 60)) sec"
