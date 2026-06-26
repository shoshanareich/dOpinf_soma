#!/bin/bash
#SBATCH --job-name=rom_eval
#SBATCH --output=logs/eval_%A.out
#SBATCH --error=logs/eval_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00

# Re-run ONLY the evaluator (no retraining) on an existing results dir.
# Defaults to the single-forcing tau0.4 run.

source activate opinf_mixed

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export UCX_TLS="${UCX_TLS:-rc,sm,self}"

root_dir=/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/
CONFIG_FILE="$root_dir/configs_forcings/config_one_forcing.json"
# Bare results root; the evaluator rebuilds the preproc dir as
# <root>/save_roms/<dir_extension> (passing --preproc-dir doubled is the known bug).
scratch_root_dir=/scratch2/shoshi/soma4/dOpInf_results/

# Extract config fields used to locate the results directory
n_year_train=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['n_year_train'])")
center_opt=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['center_opt'])")
scale=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['scale'])")
n_days=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['n_days'])")
dir_extension=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('dir_extension','default_extension'))")

r=$(python3 -c "import json; d = json.load(open('$CONFIG_FILE')); v = d.get('r'); print(v if v is not None else '')")
energy=$(python3 -c "import json; d = json.load(open('$CONFIG_FILE')); v = d.get('target_ret_energy'); print(v if v is not None else '')")

if [ -n "$r" ]; then
    SUFFIX="r${r}"
elif [ -n "$energy" ]; then
    SUFFIX="energy${energy}"
else
    SUFFIX="unknown"
fi

# Results dir written by the driver: <root>/save_roms/<dir_extension>/<center>_<scale>_<n_days>days_<yrs>yrs_<SUFFIX>/
RESULTS_DIR="${scratch_root_dir}save_roms/${dir_extension}/${center_opt}_${scale}_${n_days}days_${n_year_train}yrs_${SUFFIX}"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: results dir not found: $RESULTS_DIR"
    exit 1
fi
echo "Evaluating results in: $RESULTS_DIR"

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
