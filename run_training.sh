#!/usr/bin/env bash
#
# Slurm job script to run SARSA training
# Usage (from cs4246-project-blackjack folder):
#   sbatch run_training.sh

#SBATCH --job-name=hle_eval_client
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1121685@u.nus.edu

cd ~/cs4246-project-blackjack || exit 1
source ~/cs4246-project-blackjack/.venv/bin/activate

rm -rf *.log

srun -u python hi_lo_variant/add_split/sarsa.py \
  --preplay-episodes 100_000_000 \
  --bet-episodes 100_000_000 \
