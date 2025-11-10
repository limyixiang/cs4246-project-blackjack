"""Standalone evaluator for merged Blackjack artifacts (add_double variant).

Loads Q_play / Q_bet tables from a merged .npz, re‑evaluates them against the
environment, and produces several diagnostic plots:
  * Average return by true count bucket (with 95% CI)
  * Greedy bet multiplier chosen per true count
  * Q_bet heatmap
  * Q_play policy & delta (Q_hit - Q_stand) heatmaps for a selected TC bucket

Usage (defaults point to the 50m_50m merged example):

    python evaluate_loaded_model.py \
        --artifact artifacts_merged/add_double/50m_50m/ep_50_000_000_50_000_000_merged.npz \
        --meta     artifacts_merged/add_double/50m_50m/ep_50_000_000_50_000_000.meta.json \
        --episodes 200000

    python.exe -X utf8 evaluate_loaded_model.py --artifact artifacts_merged/add_double/50m_50m/ep_50_000_000_50_000_000_merged.npz --meta artifacts_merged/add_double/50m_50m/ep_50_000_000_50_000_000.meta.json

Disable plots with --no-plots. Output figures saved next to the artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import importlib

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate merged Blackjack artifacts (add_double variant)")
    p.add_argument("--artifact", type=Path, default=Path("artifacts_merged/add_double/50m_50m/ep_50_000_000_50_000_000_merged.npz"), help="Path to merged .npz containing Q_play & Q_bet")
    p.add_argument("--meta", type=Path, default=Path("artifacts_merged/add_double/50m_50m/ep_50_000_000_50_000_000.meta.json"), help="Path to meta JSON (training configuration)")
    p.add_argument("--variant", type=str, default=None, help="Variant folder name under hi_lo_variant (e.g. add_double, add_split, add_surrender, original, bet_sizing). If omitted, inferred from artifact path.")
    p.add_argument("--episodes", type=int, default=1_000_000, help="Episodes for evaluation / TC return aggregation")
    p.add_argument("--bankroll-episodes", type=int, default=1_000_000, help="Episodes for bankroll ROI evaluation")
    p.add_argument("--tc-bucket", type=int, default=None, help="TC bucket index for Q_play visualisations (default=middle bucket)")
    p.add_argument("--annotate-qbet", action="store_true", help="Annotate Q_bet heatmap with numeric values")
    p.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    p.add_argument("--policy-only", action="store_true", help="Only plot greedy policy (skip delta heatmap)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def save_figure(fig, filename: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[plot] Saved {path}")
    return path


def save_json(obj, filename: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"[json] Saved {path}")
    return path


def _obs_fields(obs):
    """Extract (player_sum, dealer_val, usable_ace, tc_idx, phase) from obs.

    Supports variants with extra fields (e.g., add_split adds hand indices).
    Only the first 5 entries are used here; extras are ignored.
    """
    try:
        ps = int(obs[0]); dv = int(obs[1]); ua = int(obs[2]); tc_idx = int(obs[3]); phase = int(obs[4])
        return ps, dv, ua, tc_idx, phase
    except Exception:
        # Fallback: coerce to list then index
        o = list(obs)
        ps = int(o[0]); dv = int(o[1]); ua = int(o[2]); tc_idx = int(o[3]); phase = int(o[4])
        return ps, dv, ua, tc_idx, phase


class LoadedAgent:
    """Simple facade over loaded Q tables allowing greedy evaluation."""
    def __init__(self, env, Q_play: np.ndarray, Q_bet: np.ndarray, meta: dict | None = None):
        self.env = env
        self.Q_play = Q_play
        self.Q_bet = Q_bet
        self.meta = meta or {}
        self.rng = np.random.default_rng(12345)

    def greedy_bet(self, obs) -> int:
        tc_idx = int(obs[3])  # already bucket index 0..N-1
        q = self.Q_bet[tc_idx]
        m = q.max(); idxs = np.flatnonzero(q == m)
        return int(self.rng.choice(idxs))

    def greedy_play(self, obs) -> int:
        ps, dv, ua, tc_idx, phase = _obs_fields(obs)
        assert phase == 1, "greedy_play called outside play phase"
        q = self.Q_play[ps, dv, ua, tc_idx]
        valid = tuple(self.env.get_valid_actions_idxs())
        # Mask invalid actions
        q_masked = q.copy().astype(np.float64)
        for a in range(q_masked.shape[-1]):
            if a not in valid:
                q_masked[a] = -np.inf
        m = np.max(q_masked); idxs = np.flatnonzero(q_masked == m)
        return int(self.rng.choice(idxs))


def eval_avg_return_by_tc(agent: LoadedAgent, env, episodes=200_000, rng=None):
    """Aggregate average return per hand grouped by starting TC bucket."""
    rng = rng or np.random.default_rng()
    base = getattr(env, 'unwrapped', env)
    n_buckets = base.observation_space.spaces[3].n
    labels = np.array(getattr(base, 'tc_bucket_names', [str(i) for i in range(n_buckets)]))
    ret_sum = np.zeros(n_buckets, dtype=np.float64)
    ret_sumsq = np.zeros(n_buckets, dtype=np.float64)
    counts = np.zeros(n_buckets, dtype=np.int64)

    for _ in range(episodes):
        obs, _ = env.reset()
        tc_idx = int(obs[3])
        # betting phase
        a_bet = agent.greedy_bet(obs)
        obs, _, term, trunc, _ = env.step(a_bet)
        assert not (term or trunc), "Betting phase should not terminate"
        done = False; G = 0.0
        while not done:
            a = agent.greedy_play(obs)
            obs, r, term, trunc, _ = env.step(a)
            G += r; done = term or trunc
        ret_sum[tc_idx] += G
        ret_sumsq[tc_idx] += G * G
        counts[tc_idx] += 1

    denom = np.maximum(counts, 1)
    mean = ret_sum / denom
    var = (ret_sumsq / denom) - mean**2
    se = np.sqrt(np.maximum(var, 0.0) / denom)
    ci_lo = mean - 1.96 * se
    ci_hi = mean + 1.96 * se
    return labels, mean, (ci_lo, ci_hi), counts


def plot_avg_return_by_tc(labels, mean, ci, counts, folder: Path, prefix: str, min_visits=1000):
    ci_lo, ci_hi = ci
    mask = counts >= min_visits
    x = np.arange(len(labels))[mask]; y = mean[mask]
    yerr = np.vstack((y - ci_lo[mask], ci_hi[mask] - y))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, y, edgecolor='k')
    ax.set_xticks(x, labels[mask])
    ax.errorbar(x, y, yerr=yerr, fmt='none', capsize=2, linewidth=1, color='k')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xlabel('True Count')
    ax.set_ylabel('Average Return per Hand')
    ax.set_title('Average Return by True Count')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, f"{prefix}_avg_return_by_tc.png", folder)
    plt.close(fig)


def build_bet_df(agent: LoadedAgent, env):
    base = getattr(env, 'unwrapped', env)
    n_buckets = base.observation_space.spaces[3].n
    labels = np.array(getattr(base, 'tc_bucket_names', [str(i) for i in range(n_buckets)]))
    rows = []
    for idx in range(n_buckets):
        q_bets = np.asarray(agent.Q_bet[idx], dtype=float)
        best = int(np.argmax(q_bets))
        rows.append({
            'TC_idx': idx,
            'TC': labels[idx],
            'best_bet_action': best,
            'bet_multiplier': float(base.bet_multipliers[best]),
            'Q_bets': q_bets.copy(),
        })
    return pd.DataFrame(rows).sort_values('TC_idx').reset_index(drop=True)


def plot_bet_multiplier(df: pd.DataFrame, folder: Path, prefix: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(df['TC'], df['bet_multiplier'], color='tab:green', edgecolor='k')
    ax.set_xlabel('True Count')
    ax.set_ylabel('Greedy Bet Multiplier')
    ax.set_title('Learned Bet Multiplier by True Count (Phase 0)')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, f"{prefix}_bet_multiplier_by_tc.png", folder)
    plt.close(fig)


def plot_q_bet_heatmap(agent: LoadedAgent, env, folder: Path, prefix: str, annotate=False):
    base = getattr(env, 'unwrapped', env)
    n_tc = base.observation_space.spaces[3].n
    labels_tc = np.array(getattr(base, 'tc_bucket_names', [str(i) for i in range(n_tc)]))
    bet_multipliers = np.array(base.bet_multipliers)
    bet_labels = [f"{m:g}x" for m in bet_multipliers]
    Q = np.array(agent.Q_bet, dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(bet_labels)), max(4, 0.5 * len(labels_tc) + 1)))
    im = ax.imshow(Q, aspect='auto', cmap='viridis')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Q value')
    ax.set_xticks(np.arange(len(bet_labels)), labels=bet_labels)
    ax.set_yticks(np.arange(len(labels_tc)), labels=labels_tc)
    ax.set_xlabel('Bet action (multiplier)')
    ax.set_ylabel('True Count bucket')
    ax.set_title('Q_bet heatmap')
    if annotate:
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                ax.text(j, i, f"{Q[i, j]:.2f}", ha='center', va='center', color='w', fontsize=8)
    fig.tight_layout()
    save_figure(fig, f"{prefix}_q_bet_heatmap.png", folder)
    plt.close(fig)


def _dealer_tick_labels():
    return [str(i) for i in range(2, 11)] + ['A']


def plot_q_play(
    agent: LoadedAgent,
    env,
    folder: Path,
    prefix: str,
    action_indices: dict,
    action_labels: list[str],
    tc_idx=None,
    show='policy',
    vmin=None,
    vmax=None,
):
    base = getattr(env, 'unwrapped', env)
    n_tc = base.observation_space.spaces[3].n
    if tc_idx is None:
        tc_idx = n_tc // 2
    tc_idx = int(np.clip(tc_idx, 0, n_tc - 1))
    Q = agent.Q_play[:, :, :, tc_idx, :]  # (player_sum, dealer, usable_ace, action)
    titles = {
        'delta': 'Q_hit - Q_stand',
        'stand': 'Q_stand',
        'hit': 'Q_hit',
        'surrender': 'Q_surrender',
        'double': 'Q_double',
        'split': 'Q_split',
        'policy': 'Greedy policy'
    }
    dealer_slice = slice(1, 11)  # (Ace,2..10) raw order

    def _reorder(M):
        # Move Ace first column to end so ticks match [2..10,A]
        return np.concatenate([M[..., 1:], M[..., :1]], axis=-1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    for u, ax in enumerate(axs):
        if show == 'policy':
            A = np.argmax(Q[:, :, u, :], axis=-1)
            Mraw = A[4:22, dealer_slice]
            M = _reorder(Mraw)
            n_actions = Q.shape[-1]
            im = ax.imshow(M, aspect='auto', cmap='tab10', vmin=0, vmax=n_actions-1)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=list(range(n_actions)))
            cbar.ax.set_yticklabels(action_labels[:n_actions])
        else:
            if show == 'delta':
                hit_i = action_indices.get('HIT', 1)
                stand_i = action_indices.get('STICK', 0)
                if hit_i >= Q.shape[-1] or stand_i >= Q.shape[-1]:
                    raise ValueError("HIT/STICK indices not available for delta view")
                Mfull = Q[:, :, u, hit_i] - Q[:, :, u, stand_i]
                cbar_label = 'Q_hit - Q_stand'; cmap = 'coolwarm'
            elif show == 'stand':
                idx = action_indices.get('STICK', 0)
                Mfull = Q[:, :, u, idx]; cbar_label = 'Q value (Stand)'; cmap = 'viridis'
            elif show == 'hit':
                idx = action_indices.get('HIT', 1)
                Mfull = Q[:, :, u, idx]; cbar_label = 'Q value (Hit)'; cmap = 'viridis'
            elif show == 'surrender':
                if 'SURRENDER' not in action_indices:
                    raise ValueError("Surrender action not available for this variant")
                idx = action_indices['SURRENDER']
                Mfull = Q[:, :, u, idx]; cbar_label = 'Q value (Surrender)'; cmap = 'viridis'
            elif show == 'double':
                if 'DOUBLE' not in action_indices:
                    raise ValueError("Double action not available for this variant")
                idx = action_indices['DOUBLE']
                Mfull = Q[:, :, u, idx]; cbar_label = 'Q value (Double)'; cmap = 'viridis'
            elif show == 'split':
                if 'SPLIT' not in action_indices:
                    raise ValueError("Split action not available for this variant")
                idx = action_indices['SPLIT']
                Mfull = Q[:, :, u, idx]; cbar_label = 'Q value (Split)'; cmap = 'viridis'
            else:
                raise ValueError("Unsupported show mode")
            Mraw = Mfull[4:22, dealer_slice]
            M = _reorder(Mraw)
            im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
        ax.set_title(f'UA={u}')
        ax.set_xlabel('Dealer upcard')
        ax.set_xticks(np.arange(10), labels=_dealer_tick_labels())
        ax.set_ylabel('Player sum' if u == 0 else '')
        ax.set_yticks(np.arange(18), labels=[str(ps) for ps in range(4, 22)])
    tc_label = getattr(base, 'tc_bucket_names', [str(i) for i in range(n_tc)])[tc_idx]
    fig.suptitle(f"Q_play ({titles.get(show, show)}) at TC bucket {tc_idx} ({tc_label})")
    fig.tight_layout()
    save_figure(fig, f"{prefix}_q_play_{show}_tc{tc_idx}.png", folder)
    plt.close(fig)


def evaluate_bankroll(agent: LoadedAgent, env, episodes=50_000):
    """Evaluate EV/ROI and win/loss/push rates over a number of hands."""
    returns = np.empty(episodes, dtype=np.float64)
    total_bet = 0.0
    wins = losses = pushes = 0
    for ep in range(episodes):
        s0, _info0 = env.reset()
        a_bet = agent.greedy_bet(s0)
        bet = float(env.bet_multipliers[int(a_bet) % env.n_bets])
        total_bet += bet
        s_play, _reward0 = env.step(a_bet)[:2]  # advance to play phase
        done = False; G = 0.0
        while not done:
            a_play = agent.greedy_play(s_play)
            s_play, r_step, term, trunc, _info = env.step(a_play)
            G += r_step; done = term or trunc
        returns[ep] = G
        if G > 0: wins += 1
        elif G < 0: losses += 1
        else: pushes += 1
    ev = returns.mean()
    se = returns.std(ddof=1) / np.sqrt(episodes)
    ci = (ev - 1.96 * se, ev + 1.96 * se)
    roi = ev / (total_bet / episodes) if total_bet > 0 else 0.0
    return {
        "hands": episodes,
        "bankroll_change": float(returns.sum()),
        "ev_per_hand": float(ev),
        "ev_95%_CI": ci,
        "avg_bet": float(total_bet / episodes),
        "roi_per_hand": float(roi),
        "win_rate": wins / episodes,
        "loss_rate": losses / episodes,
        "push_rate": pushes / episodes,
    }


def _infer_variant_from_artifact(path: Path) -> str | None:
    # Expect artifacts_merged/<variant>/<group>/filename
    try:
        return path.parent.parent.name
    except Exception:
        return None


def _import_env(variant: str):
    mod = importlib.import_module(f"hi_lo_variant.{variant}.env")
    BlackjackEnv = getattr(mod, 'BlackjackEnv')
    PlayActions = getattr(mod, 'PlayActions', None)
    return mod, BlackjackEnv, PlayActions


def _build_action_info(env_module, Q_play):
    n_actions = int(Q_play.shape[-1])
    enum = getattr(env_module, 'PlayActions', None)
    name_to_idx = {}
    idx_to_name = {i: f"A{i}" for i in range(n_actions)}
    nice = {
        'STICK': 'Stand', 'HIT': 'Hit', 'SURRENDER': 'Surrender', 'DOUBLE': 'Double', 'SPLIT': 'Split'
    }
    if enum is not None:
        for name, member in enum.__members__.items():
            idx = int(member)
            if idx < n_actions:
                name_to_idx[name] = idx
                idx_to_name[idx] = nice.get(name, name.title())
    else:
        # Fallback common layout
        for name, idx in [('STICK',0), ('HIT',1), ('SURRENDER',2), ('DOUBLE',3), ('SPLIT',4)]:
            if idx < n_actions:
                name_to_idx[name] = idx
                idx_to_name[idx] = nice[name]
    labels = [idx_to_name[i] for i in range(n_actions)]
    return name_to_idx, labels


def _readable_prefix(artifact_path: Path) -> str:
    # Prefer group folder like '50m_50m' for readability
    group = artifact_path.parent.name
    return group or artifact_path.stem.replace('_merged', '')


def main():
    args = _parse_args()
    if not args.artifact.exists():
        raise FileNotFoundError(args.artifact)
    if not args.meta.exists():
        raise FileNotFoundError(args.meta)
    with np.load(args.artifact, allow_pickle=False) as data:
        Q_play = data['Q_play']
        Q_bet = data['Q_bet']
    with args.meta.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    print(f"Loaded artifact: {args.artifact}")
    print(f"Q_play shape={Q_play.shape}  Q_bet shape={Q_bet.shape}")
    print("Meta subset:", {k: meta.get(k) for k in ['tc_min','tc_max','bet_multipliers','natural','sab']})

    # Variant resolution and environment import
    variant = args.variant or _infer_variant_from_artifact(args.artifact) or 'add_double'
    try:
        env_module, BlackjackEnv, PlayActions = _import_env(variant)
    except Exception as e:
        raise RuntimeError(f"Failed to import environment for variant '{variant}': {e}")

    # Build environment using meta (fallback defaults for missing keys)
    env = BlackjackEnv(
        natural=bool(meta.get('natural', True)),
        sab=bool(meta.get('sab', False)),
        num_decks=int(meta.get('num_decks', 4)),
        tc_min=int(meta.get('tc_min', -10)),
        tc_max=int(meta.get('tc_max', 10)),
    )
    if 'bet_multipliers' in meta:
        # Override default bet multipliers if present & consistent
        bm = np.array(meta['bet_multipliers'], dtype=np.float32)
        if bm.shape[0] == env.n_bets:
            env.bet_multipliers = bm
        else:
            print("[warn] meta bet_multipliers length mismatch; keeping env defaults")

    agent = LoadedAgent(env, Q_play, Q_bet, meta)
    out_folder = args.artifact.parent  # save beside artifact
    prefix = _readable_prefix(args.artifact)
    action_indices, action_labels = _build_action_info(env_module, Q_play)

    # Average return by TC
    labels, mean, ci, counts = eval_avg_return_by_tc(agent, env, episodes=args.episodes)
    for L, m, n in zip(labels, mean, counts):
        if n:
            print(f"TC {L}: mean={m: .5f} n={n}")
    if not args.no_plots:
        plot_avg_return_by_tc(labels, mean, ci, counts, out_folder, prefix)

    # Bet multiplier summary
    bet_df = build_bet_df(agent, env)
    print(bet_df.head().to_string(index=False))
    if not args.no_plots:
        plot_bet_multiplier(bet_df, out_folder, prefix)
        plot_q_bet_heatmap(agent, env, out_folder, prefix, annotate=args.annotate_qbet)
        tc_bucket_for_play = args.tc_bucket if args.tc_bucket is not None else None
        plot_q_play(agent, env, out_folder, prefix, action_indices, action_labels, tc_idx=tc_bucket_for_play, show='policy')
        if not args.policy_only:
            # Only attempt delta if HIT/STICK available
            if 'HIT' in action_indices and 'STICK' in action_indices:
                plot_q_play(agent, env, out_folder, prefix, action_indices, action_labels, tc_idx=tc_bucket_for_play, show='delta')

    # Bankroll evaluation
    summary = evaluate_bankroll(agent, env, episodes=args.bankroll_episodes)
    print("Bankroll summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    # Persist summary alongside figures
    save_json(summary, f"{prefix}_bankroll_summary.json", out_folder)

    print("Done.")


if __name__ == "__main__":
    main()

 
