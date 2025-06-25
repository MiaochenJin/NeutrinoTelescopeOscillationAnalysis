#!/bin/bash
#SBATCH -c 1
#SBATCH -p arguelles_delgado_gpu
#SBATCH --mem 2048 
#SBATCH -t 0-1:00
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --array=0-198

ss
python run_analysis_loe.py \
    --sin2theta23 0.2 0.8 20 \
    --dm31 1.3e-3 3e-3 15 \
    --point ${SLURM_ARRAY_TASK_ID} \
    --outfile 0624_datafitting_var/point_${SLURM_ARRAY_TASK_ID}.csv 