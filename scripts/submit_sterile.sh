#!/bin/bash
#SBATCH -c 2
#SBATCH -p arguelles_delgado
#SBATCH --mem 2048 
#SBATCH -t 0-1:00
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --array=0-2399

# this first chunk of code is submission for the full araay of points with standard parameters also free
# CHUNK=50
# BASE=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

# for (( i=0; i<CHUNK; i++ )); do
#     POINT=$(( BASE + i ))
#     python run_analysis_sterile.py \
#         --sin2theta23 0.4 0.6 20 \
#         --dm31 2e-3 3e-3 15 \
#         --point ${POINT} \
#         --outfile 0914_sterile_full/point_${POINT}.csv
# done

# this simpler chunk of code is for fixing the standard parameters
ss
python run_analysis_sterile.py \
    --config ../config/config_ic_sterile.yaml \
    --point ${SLURM_ARRAY_TASK_ID} \
    --outfile 0915_icsterile_smbf/point_${SLURM_ARRAY_TASK_ID}.csv 