#!/bin/bash
# Parallelize across both shards and episode configurations by flattening the array.
# Example: with 2 configs and 8 shards, set --array=0-23 (3*8-1).
#SBATCH -J sarsa
#SBATCH -p normal
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-15
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1121685@u.nus.edu

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
source ~/cs4246-project-blackjack/.venv/bin/activate
mkdir -p logs artifacts

# Variant/environment selection (default: add_double)
# Usage: sbatch slurm/train_test.sbatch <variant>
# Or export VARIANT before calling sbatch.
VARIANT="${VARIANT:-${1:-add_double}}"
VARIANT_PATH="hi_lo_variant/${VARIANT}"

if [ ! -f "${VARIANT_PATH}/sarsa.py" ]; then
  echo "Variant '${VARIANT}' not found at '${VARIANT_PATH}/sarsa.py'." >&2
  echo "Valid options: directories under 'hi_lo_variant/' (e.g., add_double, add_split, add_surrender, bet_sizing, original)." >&2
  exit 1
fi

SEED_BASE=12345

# Define episode configurations (must be same length)
PREPLAY_EPISODES=(100_000 200_000)
BET_EPISODES=(100_000 100_000)

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
OUTDIR="artifacts/${VARIANT}/ep_${PRE}_${BET}_task${SHARD_IDX}"
mkdir -p "$OUTDIR"
echo "[variant=${VARIANT}] [cfg ${CONFIG_IDX} / shard ${SHARD_IDX}] preplay=${PRE} bet=${BET} seed=${SEED} (configs=${N_CONFIGS} shards=${SHARDS})" | tee "$OUTDIR/run.log"

python "${VARIANT_PATH}/sarsa.py" \
  --preplay-episodes ${PRE} \
  --bet-episodes ${BET} \
  --output-dir "$OUTDIR" \
  --stats-buffer 10000 \
  --save-stage1 \
  --save-prefix checkpoint \
  --collect-metrics \
  --metrics-maxlen 1000 \
  --seed ${SEED} \
  --array-task-id ${SHARD_IDX} \
  --array-task-count ${SHARDS} \
  --eval-episodes 50000 \
  --eval-tc-episodes 50000

echo "Done cfg ${CONFIG_IDX} shard ${SHARD_IDX}"
