#!/bin/bash
# Parallelize across both shards and episode configurations by flattening the array.
# Example: with 3 configs and 8 shards, set --array=0-23 (3*8-1).
#SBATCH -J sarsa
#SBATCH -p long
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-20
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1121685@u.nus.edu

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs artifacts

SEED_BASE=12345

# Define episode configurations (must be same length)
PREPLAY_EPISODES=(50_000_000 100_000_000 500_000_000)
BET_EPISODES=(50_000_000 50_000_000 50_000_000)

if [ ${#PREPLAY_EPISODES[@]} -ne ${#BET_EPISODES[@]} ]; then
  echo "Config error: PREPLAY_EPISODES and BET_EPISODES length mismatch" >&2
  exit 1
fi

N_CONFIGS=${#PREPLAY_EPISODES[@]}
TOTAL_TASKS=${SLURM_ARRAY_TASK_COUNT}
if (( TOTAL_TASKS % N_CONFIGS != 0 )); then
  echo "Array size ($TOTAL_TASKS) must be a multiple of number of configs ($N_CONFIGS)" >&2
  exit 1
fi
SHARDS=$(( TOTAL_TASKS / N_CONFIGS ))

# Map flattened array id -> (config_idx, shard_idx)
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID % N_CONFIGS ))
SHARD_IDX=$(( SLURM_ARRAY_TASK_ID / N_CONFIGS ))

PRE=${PREPLAY_EPISODES[$CONFIG_IDX]}
BET=${BET_EPISODES[$CONFIG_IDX]}
SEED=$((SEED_BASE + SHARD_IDX * 100 + CONFIG_IDX))
OUTDIR="artifacts/ep_${PRE}_${BET}_task${SHARD_IDX}"
mkdir -p "$OUTDIR"
echo "[cfg ${CONFIG_IDX} / shard ${SHARD_IDX}] preplay=${PRE} bet=${BET} seed=${SEED} (configs=${N_CONFIGS} shards=${SHARDS})" | tee "$OUTDIR/run.log"

python hi_lo_variant/add_split/sarsa.py \
  --preplay-episodes ${PRE} \
  --bet-episodes ${BET} \
  --output-dir "$OUTDIR" \
  --stats-buffer 200000 \
  --save-stage1 \
  --save-prefix checkpoint \
  --collect-metrics \
  --metrics-maxlen 50000 \
  --seed ${SEED} \
  --array-task-id ${SHARD_IDX} \
  --array-task-count ${SHARDS} \

echo "Done cfg ${CONFIG_IDX} shard ${SHARD_IDX}"
