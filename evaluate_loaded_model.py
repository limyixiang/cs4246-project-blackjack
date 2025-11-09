import numpy as np, json
from pathlib import Path
from hi_lo_variant.add_double.env import BlackjackEnv

SAVE_PATH = Path("artifacts_merged/sample_merged.npz")
META_PATH = Path("artifacts_merged/sample_meta.json")

meta: dict = {}
Q_play = None
Q_bet = None
try:
    with np.load(SAVE_PATH, allow_pickle=False) as data:
        Q_play = data['Q_play']
        Q_bet = data['Q_bet']
    if META_PATH.exists():
        with META_PATH.open('r', encoding='utf-8') as f:
            meta = json.load(f)
    print('Loaded:', str(SAVE_PATH))
    if meta:
        print('Meta:', {k: meta.get(k) for k in ['tc_min','tc_max','bet_multipliers','Q_play_shape','Q_bet_shape']})
    else:
        print('Meta: <none>')
except Exception as e:
    print(f"[warn] Failed to load checkpoint {SAVE_PATH}: {e}.")
    raise SystemExit(1)

env = BlackjackEnv(num_decks=4, tc_min=-10, tc_max=10, natural=True)

# Build a lightweight agent facade reusing the same BlackjackAgent API for evaluation
class LoadedAgent:
    def __init__(self, env, Q_play, Q_bet, meta: dict | None = None):
        self.env = env
        self.Q_play = Q_play
        self.Q_bet = Q_bet
        self.meta = meta or {}
        # Use training-time tc_min/tc_max if available to align indices
        self.tc_min = int(self.meta.get('tc_min', getattr(env.unwrapped, 'tc_min', -10)))
        self.rng = np.random.default_rng(12345)
    def greedy_bet(self, obs):
        tc = int(obs[3]); q = self.Q_bet[tc]
        m = q.max(); idxs = np.flatnonzero(q == m)
        return int(self.rng.choice(idxs))

    def greedy_play(self, obs):
        ps, dv, ua, tc, _ = int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3]), int(obs[4])
        q = self.Q_play[ps, dv, ua, tc]
        valid = tuple(env.unwrapped.get_valid_actions_idxs())
        q_masked = q.copy().astype(np.float64)
        for a in range(q_masked.shape[-1]):
            if a not in valid:
                q_masked[a] = -np.inf
        m = np.max(q_masked); idxs = np.flatnonzero(q_masked == m)
        return int(self.rng.choice(idxs))
    
def evaluate_bankroll(agent, env, episodes=200_000, rng=None):
    """Evaluate the trained agent.

    The win/lose/push classification is based on the SIGN of the accumulated
    rewards (return) per episode (hand), consistent with split-hand aggregation.
      >0  => win
      =0  => push
      <0  => loss
    """
    rng = rng or np.random.default_rng()
    returns = np.empty(episodes, dtype=np.float64)
    total_bet = 0.0
    wins = losses = pushes = 0

    for ep in range(episodes):
        s0, _ = env.reset()                    # phase 0 (bet)
        a_bet = agent.greedy_bet(s0)
        bet = float(env.unwrapped.bet_multipliers[int(a_bet) % env.unwrapped.n_bets])
        total_bet += bet

        # play hand (may include splits)
        s, _ = env.step(a_bet)[:2]
        done = False; G = 0.0
        while not done:
            a = agent.greedy_play(s)
            s, r, term, trunc, _ = env.step(a)
            G += r; done = term or trunc

        returns[ep] = G
        if G > 0: wins += 1
        elif G < 0: losses += 1
        else: pushes += 1

    ev = returns.mean()                        # units/hand
    se = returns.std(ddof=1) / np.sqrt(episodes)
    ci = (ev - 1.96*se, ev + 1.96*se)
    roi = ev / (total_bet / episodes)          # profit per unit bet

    summary = {
        "hands": episodes,
        "bankroll_change": returns.sum(),
        "ev_per_hand": ev,
        "ev_95%_CI": ci,
        "avg_bet": total_bet / episodes,
        "roi_per_hand": roi,
        "win_rate": wins / episodes,
        "loss_rate": losses / episodes,
        "push_rate": pushes / episodes,
    }
    return summary

loaded_agent = LoadedAgent(env, Q_play, Q_bet, meta)
summary = evaluate_bankroll(loaded_agent, env, episodes=1_000)
print(summary)