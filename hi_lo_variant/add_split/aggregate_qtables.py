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
import json
import os
from datetime import datetime, timezone
import numpy as np


def aggregate(files: list[str]) -> dict[str, np.ndarray]:
    assert files, "No input NPZ files provided"
    acc: dict[str, np.ndarray] = {}
    wbet_sum = None
    wbet_cnt = None
    qplay_sum = None
    qplay_cnt = 0
    valid = 0
    for f in files:
        try:
            z = np.load(f, allow_pickle=False)
        except Exception as e:
            print(f"[warn] Skipping unreadable NPZ '{f}': {e}")
            continue
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
        valid += 1

    if valid == 0:
        raise ValueError("No valid NPZ files to aggregate after skipping unreadable inputs")

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
    p.add_argument('--meta-out', type=str, default=None, help='Optional JSON metadata output path')
    p.add_argument('--variant', type=str, default=None, help='Variant/environment name (for metadata)')
    p.add_argument('--config', type=str, default=None, help='Config id (e.g., ep_<pre>_<bet>) for metadata')
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

    if args.meta_out:
        meta = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'inputs': files,
            'output_npz': args.out,
            'variant': args.variant,
            'config': args.config,
            'qbet_shape': agg['Q_bet'].shape,
            'qplay_shape': agg['Q_play'].shape,
            'num_shards': len(files),
        }
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(args.meta_out)), exist_ok=True)
        with open(args.meta_out, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote metadata to {args.meta_out}")


if __name__ == '__main__':
    main()
