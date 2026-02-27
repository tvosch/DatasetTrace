#!/bin/bash
#SBATCH --job-name=index-array
#SBATCH --partition=...
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --array=0-19%8
#SBATCH --constraint=scratch-node # get local disk if possible

# Save to local disk if possible
# Potentially do more shards and more arrays like NUM_SHARDS=20 and --array=0-19%8
# Aim to have 1 shard max 128 GB?


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




