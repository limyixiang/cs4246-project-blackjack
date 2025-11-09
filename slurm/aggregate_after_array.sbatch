#!/bin/bash
#SBATCH -J sarsa-aggregate
#SBATCH -A yixiang
#SBATCH -p normal
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -o logs/%x_%A.out
#SBATCH -e logs/%x_%A.err

set -euo pipefail
module purge

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs artifacts_merged

# Optionally pass custom roots as args when submitting
# e.g. sbatch --dependency=afterok:<ARRAY_JOB_ID> slurm/aggregate_after_array.sbatch artifacts artifacts_merged
bash slurm/aggregate_all.sh "$@"

echo "Aggregation completed: see artifacts_merged/"
