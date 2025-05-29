#!/bin/bash
#SBATCH -c 1
#SBATCH -p arguelles_delgado_gpu
#SBATCH --mem 2048 
#SBATCH -t 0-1:00
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --array=0-399

ss
python run_analysis_sterile.py \
    --sin2theta24 1e-3 3e-2 20 \
    --dm41 1e-4 1e-1 20  \
    --point ${SLURM_ARRAY_TASK_ID} \
    --outfile 0529_sterile/point_${SLURM_ARRAY_TASK_ID}.csv 