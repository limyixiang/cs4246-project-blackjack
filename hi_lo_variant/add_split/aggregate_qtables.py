#!/usr/bin/env python3
"""
Aggregate multiple worker NPZ checkpoints (from sarsa.py) into a merged model.

Usage:
  python aggregate_qtables.py --inputs artifacts/checkpoint_final_task*.npz --out merged_agent.npz

Aggregation strategy:
- Q_bet: weighted average by N_bet per (tc, action)
- Q_play: unweighted average (no per-state counts available); if all provided have same shape
  Note: For exact weighting, extend sarsa.py to export N_play.
"""
from __future__ import annotations
import argparse
from glob import glob
import numpy as np


def aggregate(files: list[str]) -> dict[str, np.ndarray]:
    assert files, "No input NPZ files provided"
    acc: dict[str, np.ndarray] = {}
    wbet_sum = None
    wbet_cnt = None
    qplay_sum = None
    qplay_cnt = 0
    for f in files:
        z = np.load(f, allow_pickle=False)
        Q_bet = z['Q_bet']
        N_bet = z['N_bet'] if 'N_bet' in z.files else np.ones_like(Q_bet, dtype=np.int64)
        Q_play = z['Q_play']
        if wbet_sum is None:
            wbet_sum = np.zeros_like(Q_bet, dtype=np.float64)
            wbet_cnt = np.zeros_like(N_bet, dtype=np.float64)
        wbet_sum += Q_bet * N_bet
        wbet_cnt += N_bet
        if qplay_sum is None:
            qplay_sum = np.zeros_like(Q_play, dtype=np.float64)
        qplay_sum += Q_play
        qplay_cnt += 1

    Q_bet_agg = np.divide(wbet_sum, np.maximum(wbet_cnt, 1), where=(wbet_cnt>0))
    Q_play_agg = qplay_sum / max(qplay_cnt, 1)
    return {
        'Q_bet': Q_bet_agg.astype(np.float32),
        'Q_play': Q_play_agg.astype(np.float32),
    }


def main():
    p = argparse.ArgumentParser(description="Aggregate worker NPZ checkpoints")
    p.add_argument('--inputs', nargs='+', help='NPZ files or glob patterns', required=True)
    p.add_argument('--out', type=str, default='merged_agent.npz', help='Output NPZ path')
    args = p.parse_args()

    files: list[str] = []
    for pat in args.inputs:
        files.extend(glob(pat))
    files = sorted(set(files))
    if not files:
        raise SystemExit('No NPZ files matched')
    agg = aggregate(files)
    np.savez_compressed(args.out, **agg)
    print(f"Wrote merged model to {args.out} from {len(files)} shards")


if __name__ == '__main__':
    main()
