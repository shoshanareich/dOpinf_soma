#!/bin/bash
#SBATCH --job-name=save_surface
#SBATCH --output=logs/save_surface_%A.out
#SBATCH --error=logs/save_surface_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00

# Write surface (k=0) fields for U,V,Eta,T,S to netCDF, for every train/test
# scenario of a ROM run + the matching FOM runs. Serial (no MPI).

source activate opinf_mixed
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

root_dir=/home/shoshi/jupyter_notebooks/OpInf/dOpinf_soma/

# Edit DATA_DIR to point at the ROM run you want surface fields for.
DATA_DIR=/scratch2/shoshi/soma4/dOpInf_results/save_roms/taus_2_4_sameIC/mean_var_1days_10yrs_r105/

echo "[timing] start $(date '+%Y-%m-%d %H:%M:%S')  data-dir=$DATA_DIR"
python3 "$root_dir/src/save_surface_fields.py" --data-dir "$DATA_DIR"
echo "[timing] done  $(date '+%Y-%m-%d %H:%M:%S')"
