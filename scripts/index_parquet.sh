#!/bin/bash
#SBATCH --job-name=index
#SBATCH --partition=...
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=72:00:00

# module loads or apptainer

python -m indexing.indexing \
    --data_dir ... \
    --save_dir .../ \
    --num_shards 8 \
    --mem 256