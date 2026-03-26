#!/bin/bash
#SBATCH --job-name=rom_sweep
#SBATCH --output=logs/sweep_monthly.out
#SBATCH --error=logs/sweep_monthly.err
#SBATCH --nodes=4
#SBATCH --ntasks=62
#SBATCH --time=12:00:00
###SBATCH --array=0-9%3       # Run 3 jobs at a time


source activate opinf_mixed

root_dir=/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/
#CONFIG_FILE="$root_dir/configs/config_${SLURM_ARRAY_TASK_ID}.json"
CONFIG_FILE="$root_dir/configs/config_monthly.json"
scratch_dir=/scratch/shoshi/soma4/dOpInf_results/save_roms/


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



echo "Starting job ${SLURM_ARRAY_TASK_ID} at $(date)"# > "$RESULTS_DIR/timing.log"

# Link config for Python
#ln -sf ${CONFIG_FILE} $root_dir/config.json
export ROM_CONFIG_PATH="$CONFIG_FILE"

# Launch the job
mpiexec -np 62 python3 $root_dir/src/mixed_rom.py

POINTER_FILE="$root_dir/.results_path_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -f "$POINTER_FILE" ]; then
    RESULTS_DIR=$(cat "$POINTER_FILE")
    echo "Task $SLURM_ARRAY_TASK_ID saved results to: $RESULTS_DIR"

    # Now you can use $RESULTS_DIR for post-processing or moving files
    # cp some_analysis.png "$RESULTS_DIR/"

    # Clean up the pointer file
    rm "$POINTER_FILE"
else
    echo "Error: Pointer file $POINTER_FILE not found for Task $SLURM_ARRAY_TASK_ID"
fi

#RESULTS_DIR="${scratch_dir}/${center_opt}_${scale}_${n_days}days_${n_year_train}yrs_${SUFFIX}"
#mkdir -p "$RESULTS_DIR"

cp "$CONFIG_FILE" "$RESULTS_DIR/config.json"
# Record finish time
echo "Finished job ${SLURM_ARRAY_TASK_ID} at $(date)"## >> "$RESULTS_DIR/timing.log"


