#!/bin/bash
#SBATCH --job-name=index
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=72:00:00

# Snellius specific
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load GCC/13.3.0
module load MPFR/4.2.1-GCCcore-13.3.0
module load MPC/1.3.1-GCCcore-13.3.0
module load ISL/0.26-GCCcore-13.3.0

time python -m indexing.indexing \
    --data_dir ... \
    --save_dir .../ \
    --num_shards 8 \
    --mem 256