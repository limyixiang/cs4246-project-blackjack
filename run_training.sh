#!/usr/bin/env bash
#
# Slurm job script to run SARSA training
# Usage (from cs4246-project-blackjack folder):
#   sbatch run_training.sh

#SBATCH --job-name=sarsa
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=long
#SBATCH --hint=nomultithread
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1121685@u.nus.edu

cd ~/cs4246-project-blackjack || exit 1
source ~/cs4246-project-blackjack/.venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

srun -u --cpu-bind=cores --threads-per-core=1 \
  python hi_lo_variant/add_split/sarsa.py \
    --preplay-episodes 100_000_000 \
    --bet-episodes 100_000_000 \
