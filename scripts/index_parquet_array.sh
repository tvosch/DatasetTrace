#!/bin/bash
#SBATCH --job-name=index-array
#SBATCH --partition=...
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --array=0-19%8
#SBATCH --constraint=scratch-node # get local disk if possible

# Save to local disk if possible
# Aim to have 1 shard max ~50 GB corpus (rule of thumb: corpus ≈ 5× compressed file size).
#
# Before submitting, check how many shards will actually be created (instant, no decompression):
#   python -m indexing.indexing --data_dir $DATA_DIR --num_shards $NUM_SHARDS --dry_run
# Set --array=0-(N-1) accordingly to avoid wasting jobs on empty shards.

# module loads or apptainer

NUM_SHARDS=20

DATA_DIR=...
SAVE_DIR=...

echo "Array task ${SLURM_ARRAY_TASK_ID} of ${NUM_SHARDS}"

LOCAL=$TMPDIR/${SLURM_ARRAY_TASK_ID}
OUTPUT_DIR=.../index/${SLURM_ARRAY_TASK_ID}
mkdir -p $OUTPUT_DIR

python -m indexing.indexing \
    --data_dir   $DATA_DIR \
    --save_dir   $LOCAL/index \
    --temp_dir   $LOCAL/tmp \
    --num_shards $NUM_SHARDS \
    --shard_id   $SLURM_ARRAY_TASK_ID \
    --mem 300 \
    --cpus 64

# copy only the two final files — everything else can be discarded
rsync $LOCAL/index/data.fm9 $LOCAL/index/meta.fm9 $OUTPUT_DIR/




