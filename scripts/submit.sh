#!/bin/bash
#SBATCH -c 2
#SBATCH -p arguelles_delgado
#SBATCH --mem 2048 
#SBATCH -t 0-1:00
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --array=0-599

ss
python run_analysis_datafit.py \
    --sin2theta23 0.3 0.7 30 \
    --dm31 1.5e-3 3e-3 20 \
    --point ${SLURM_ARRAY_TASK_ID} \
    --outfile 0627_datafit_var/point_${SLURM_ARRAY_TASK_ID}.csv 