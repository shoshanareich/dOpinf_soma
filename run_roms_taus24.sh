#!/bin/bash
#SBATCH --job-name=roms_taus24
#SBATCH --output=logs/taus24_%A_%a.out
#SBATCH --error=logs/taus24_%A_%a.err
#SBATCH --nodes=3
#SBATCH --ntasks=62
#SBATCH --time=12:00:00
#SBATCH --array=0-5%5      # 6 (data,years,center) groups; max 5 concurrent = 15 nodes

# Each array task = one (data, years, center) group and runs its r-values
# SEQUENTIALLY. Configs in a group share center*/transformed-surface preproc files,
# so running them sequentially avoids concurrent-write corruption. Different groups
# write different filenames (or different dir_extension), so they are safe in parallel.

nprocs=62
source activate opinf_mixed

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export UCX_TLS="${UCX_TLS:-rc,sm,self}"

root_dir=/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma
scratch_base=/scratch2/shoshi/soma4/dOpInf_results
config_dir="$root_dir/configs_forcings/taus_2_4"

GROUP=${SLURM_ARRAY_TASK_ID:-0}
# contiguous triples: group g -> config indices 3g, 3g+1, 3g+2
CONFIG_IDXS=( $((GROUP * 3)) $((GROUP * 3 + 1)) $((GROUP * 3 + 2)) )

export ROM_POINTER_DIR="$root_dir"
GROUP_START=$SECONDS
echo "[timing] group ${GROUP} start at $(date '+%Y-%m-%d %H:%M:%S') -> configs ${CONFIG_IDXS[*]}"

for IDX in "${CONFIG_IDXS[@]}"; do
    CONFIG_FILE="$config_dir/config_${IDX}.json"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "[warn] missing $CONFIG_FILE, skipping"
        continue
    fi

    # Pull the fields we need to build paths (single space-separated line)
    read -r dir_extension n_year_train center_opt scale n_days r <<EOF
$(python3 -c "import json; d=json.load(open('$CONFIG_FILE')); print(d['dir_extension'], d['n_year_train'], d['center_opt'], d['scale'], d['n_days'], d['r'])")
EOF

    scratch_root_dir="$scratch_base/save_roms/${dir_extension}/"
    RESULTS_DIR="${scratch_root_dir}${center_opt}_${scale}_${n_days}days_${n_year_train}yrs_r${r}"
    export ROM_CONFIG_PATH="$CONFIG_FILE"

    echo "=============================================================="
    echo "[timing] config ${IDX} (${dir_extension} ${n_year_train}yr ${center_opt} r${r}) start $(date '+%H:%M:%S')"
    CFG_START=$SECONDS

    mpiexec -np ${nprocs} python3 "$root_dir/src/mixed_rom_forcings.py"
    if [ $? -ne 0 ]; then
        echo "[error] training failed for config ${IDX}; skipping eval"
        continue
    fi

    mkdir -p "$RESULTS_DIR"
    cp "$CONFIG_FILE" "$RESULTS_DIR/config.json"

    python3 "$root_dir/src/eval_rom_forcings.py" \
        --config "$CONFIG_FILE" \
        --root-dir "$scratch_root_dir" \
        --preproc-dir "$scratch_root_dir" \
        --data-dir "$RESULTS_DIR" \
        --outdir "$RESULTS_DIR/analysis"
    if [ $? -ne 0 ]; then
        echo "[error] eval failed for config ${IDX}"
    fi

    echo "[timing] config ${IDX} done in $(( (SECONDS - CFG_START) / 60 )) min -> $RESULTS_DIR"
done

# tidy any pointer files this task created
rm -f "$root_dir"/.results_path_*_"${SLURM_ARRAY_TASK_ID:-}" 2>/dev/null

echo "[timing] group ${GROUP} finished in $(( (SECONDS - GROUP_START) / 60 )) min at $(date '+%Y-%m-%d %H:%M:%S')"
