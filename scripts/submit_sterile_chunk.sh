#!/bin/bash
#SBATCH -c 1
#SBATCH -p arguelles_delgado
#SBATCH --mem 2048 
#SBATCH -t 0-1:00
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --array=0-1500

ss

CHUNK=50
BASE=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

for (( i=0; i<CHUNK; i++ )); do
    POINT=$(( BASE + i ))
    python run_analysis_sterile.py \
        --sin2theta24 1e-3 1 30 \
        --dm31 2e-3 3e-3 15 \
        --dm41 1e-5 1 50 \
        --point ${POINT} \
        --outfile 0601_s2t23_bf/point_${POINT}.csv
done

# for (( i=0; i<CHUNK; i++ )); do
#     POINT=$(( BASE + i ))
#     python run_analysis_sterile.py \
#         --sin2theta24 1e-3 1 30 \
#         --sin2theta23 0.305 0.705 15 \
#         --dm31 2e-3 3e-3 15 \
#         --dm41 1e-5 1 50 \
#         --point ${POINT} \
#         --outfile 0531_sterile_chunk/point_${POINT}.csv
# done