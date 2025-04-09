#!/bin/bash
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH --mem 2048 
#SBATCH -t 0-1:00
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --array=0-99


python run_analysis.py \
    --sin2theta23 0.5 0.65 10 \
    --t12 0.55 0.65 5 \
    --t13 0.13 0.16 5 \
    --t23 0.6 0.8 10 \
    --dm21 7e-5 7.5e-5 5 \
    --dm31 2.4e-3 2.6e-3 10 \
    --dcp 0 2*np.pi 5 \
    --point ${SLURM_ARRAY_TASK_ID} \
    --outfile results/point_${SLURM_ARRAY_TASK_ID}.csv