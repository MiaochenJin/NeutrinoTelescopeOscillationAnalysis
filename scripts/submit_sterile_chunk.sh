#!/bin/bash
#SBATCH -c 1
#SBATCH -p arguelles_delgado_gpu
#SBATCH --mem 2048 
#SBATCH -t 0-1:00
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --array=1002-9999

ss

CHUNK=10
BASE=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

for (( i=0; i<CHUNK; i++ )); do
    POINT=$(( BASE + i ))
    python run_analysis_sterile.py \
        --sin2theta24 1e-4 1e-3 10 \
        --sin2theta14 0 1e-4 10 \
        --sin2theta23 0.3 0.8 10 \
        --dm31 1e-3 4e-3 10 \
        --dm41 1e-5 1 10 \
        --point ${POINT} \
        --outfile 0529_sterile/point_${POINT}.csv
done