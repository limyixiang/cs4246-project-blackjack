#!/bin/bash
#SBATCH -J sarsa
#SBATCH -A yixiang
#SBATCH -p long
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-7
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail
module purge
# module load python/3.11  # adjust as needed

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs artifacts

# Optional: create venv and install requirements
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt

SEED_BASE=12345

# Define multiple (preplay, bet) episode configurations.
# Adjust or extend these arrays as needed; both must have same length.
PREPLAY_EPISODES=(5000000 10000000 15000000)
BET_EPISODES=(5000000 10000000 15000000)

if [ ${#PREPLAY_EPISODES[@]} -ne ${#BET_EPISODES[@]} ]; then
  echo "Config error: PREPLAY_EPISODES and BET_EPISODES length mismatch" >&2
  exit 1
fi

for i in "${!PREPLAY_EPISODES[@]}"; do
  PRE=${PREPLAY_EPISODES[$i]}
  BET=${BET_EPISODES[$i]}
  SEED=$((SEED_BASE + SLURM_ARRAY_TASK_ID * 100 + i))
  OUTDIR="artifacts/ep_${PRE}_${BET}_task${SLURM_ARRAY_TASK_ID}"
  mkdir -p "$OUTDIR"
  echo "[task ${SLURM_ARRAY_TASK_ID}] Run ${i}: preplay=${PRE} bet=${BET} seed=${SEED}" | tee "$OUTDIR/run.log"
  python hi_lo_variant/add_split/sarsa.py \
    --preplay-episodes ${PRE} \
    --bet-episodes ${BET} \
    --output-dir "$OUTDIR" \
    --disable-tqdm \
    --stats-buffer 200000 \
    --save-stage1 \
    --save-prefix checkpoint \
    --collect-metrics \
    --metrics-maxlen 50000 \
    --seed ${SEED} \
    --array-task-id ${SLURM_ARRAY_TASK_ID} \
    --array-task-count ${SLURM_ARRAY_TASK_COUNT}
done

echo "Done all configs: task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
