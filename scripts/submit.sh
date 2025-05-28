#!/bin/bash
#SBATCH -c 1
#SBATCH -p arguelles_delgado_gpu
#SBATCH --mem 2048 
#SBATCH -t 0-1:00
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --array=0-99

ss
python run_analysis.py \
    --sin2theta23 0.4 0.7 10 \
    --dm31 2.4e-3 2.6e-3 10 \
    --point ${SLURM_ARRAY_TASK_ID} \
    --outfile 0528/point_${SLURM_ARRAY_TASK_ID}.csv 