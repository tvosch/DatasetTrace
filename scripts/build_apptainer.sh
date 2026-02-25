#!/bin/bash
#SBATCH --job-name=build_infini_gram_mini
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Root of the infini-gram-mini git repository on the host filesystem
PROJECT_DIR=$PWD

# Where to write the finished .sif image
OUTPUT_PATH=...
IMAGE_NAME=infini_gram_mini.sif

# Apptainer cache/tmp — use fast local RAM disk to avoid GPFS thrashing
export APPTAINER_CACHEDIR=/dev/shm/$USER/apptainer_cache
export APPTAINER_TMPDIR=/dev/shm/$USER/apptainer_tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# ---------------------------------------------------------------------------
# Generate the .def file with PROJECT_DIR substituted in %files
# (Apptainer %files does not expand variables itself, so we do it here.)
# ---------------------------------------------------------------------------
DEF_FILE=$(mktemp /tmp/infini_gram_mini_XXXXXX.def)

sed "s|PLACEHOLDER_PROJECT_DIR|${PROJECT_DIR}|g" \
    "${PROJECT_DIR}/scripts/infini_gram_mini.def" \
    > "$DEF_FILE"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "Building Apptainer image: ${OUTPUT_PATH}/${IMAGE_NAME}"
apptainer build "${OUTPUT_PATH}/${IMAGE_NAME}" "$DEF_FILE"

# Clean up
rm -f "$DEF_FILE"

echo "Done. Image written to: ${OUTPUT_PATH}/${IMAGE_NAME}"