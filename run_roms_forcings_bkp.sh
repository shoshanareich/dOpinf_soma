#!/bin/bash
#SBATCH --job-name=rom_forcings
#SBATCH --output=logs/forcings.out
#SBATCH --error=logs/forcings.err
#SBATCH --nodes=4
#SBATCH --ntasks=62
#SBATCH --time=12:00:00
###SBATCH --array=0-4%2       # Run 2 jobs at a time (optional: uncomment for array submission)

source activate opinf_mixed

root_dir=/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/
configs_dir="$root_dir/configs_forcings/"
scratch_dir=/scratch/shoshi/soma4/dOpInf_results/

# Create logs directory if it doesn't exist
mkdir -p "$root_dir/logs"

# Array of config files to run
# If using SLURM array jobs, uncomment the line below and modify accordingly
#CONFIG_FILE="${configs_dir}config_forcings_${SLURM_ARRAY_TASK_ID}.json"
CONFIG_FILE="${configs_dir}/my_experiment.json"

# Otherwise, run all config files sequentially
#config_files=("$configs_dir"config_forcings_*.json)

if [ ${#config_files[@]} -eq 0 ] || [ ! -e "${config_files[0]}" ]; then
    echo "Error: No config files found in $configs_dir"
    exit 1
fi

echo "Found ${#config_files[@]} config file(s) to process"

for CONFIG_FILE in "${config_files[@]}"; do
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Skipping $CONFIG_FILE (not a file)"
        continue
    fi
    
    config_name=$(basename "$CONFIG_FILE" .json)
    echo ""
    echo "=========================================="
    echo "Processing: $config_name"
    echo "Config: $CONFIG_FILE"
    echo "=========================================="
    echo "Starting ROM training at $(date)"
    
    # Extract key parameters for logging
    n_year_train=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['n_year_train'])")
    dir_extension=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('dir_extension', 'unknown'))")
    
    # Set config path environment variable and run the ROM training
    export ROM_CONFIG_PATH="$CONFIG_FILE"
    
    # Launch the training job
    mpiexec -np 62 python3 "$root_dir/src/mixed_rom_forcings.py"
    
    if [ $? -eq 0 ]; then
        echo "ROM training completed successfully for $config_name at $(date)"
        
        # Extract results directory from the pointer file (if it exists)
        POINTER_FILE="$root_dir/.results_path_*_*"
        if [ -f $POINTER_FILE ]; then
            RESULTS_DIR=$(cat $POINTER_FILE)
            echo "Results saved to: $RESULTS_DIR"
            
            # Run evaluation script
            echo "Starting evaluation at $(date)"
            python3 "$root_dir/src/eval_rom_forcings.py" \
                --root-dir "$scratch_dir" \
                --data-dir "$RESULTS_DIR" \
                --center-opt "IC" \
                --scale "var" \
                --n-year-train "$n_year_train" \
                --n-year-predict 0 \
                --n-days 1 \
                --outdir "$RESULTS_DIR/analysis"
            
            if [ $? -eq 0 ]; then
                echo "Evaluation completed successfully at $(date)"
            else
                echo "Error: Evaluation failed for $config_name"
            fi
            
            # Clean up pointer file
            rm -f $POINTER_FILE
        else
            echo "Warning: Results pointer file not found"
        fi
    else
        echo "Error: ROM training failed for $config_name"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All configurations processed at $(date)"
echo "=========================================="
