# Hi-Lo Blackjack Environment (Gymnasium-compatible)

This folder contains a variant of the Blackjack environment that augments Gymnasium’s vanilla version with a finite multi-deck shoe, Hi‑Lo card counting, a true‑count bucket in the observation, and an explicit betting phase with bet multipliers.

## How it differs from Gymnasium’s vanilla Blackjack

- Finite shoe and reshuffle
  - Vanilla: samples cards with replacement (effectively infinite deck).
  - Hi‑Lo variant: uses a real shoe of `num_decks` decks via `Deck`. The shoe is reshuffled when fewer than ~20% of the cards remain.

- Hi‑Lo running count and True Count (TC) bucket
  - Vanilla: no notion of card counting.
  - Hi‑Lo variant: maintains a running Hi‑Lo count (2–6 = +1, 10/J/Q/K/A = −1, 7–9 = 0). True count is computed as running count divided by decks left; it is rounded to the nearest integer and clamped to a configurable range. The TC bucket index is included in the observation, and pretty string labels are exposed via `env.unwrapped.tc_bucket_names`.

- Two-phase episode with betting
  - Vanilla: a single phase with actions {stick, hit}.
  - Hi‑Lo variant: Phase 0 (bet) followed by Phase 1 (play).
    - Phase 0: choose a bet action (index into `bet_multipliers`, e.g. 1×, 2×, 4×). No cards are dealt yet; reward is 0; episode continues.
    - Phase 1: standard play with actions {0=stick, 1=hit}. Payouts are multiplied by the chosen bet.

- Observation and action spaces
  - Vanilla observation: `(player_sum, dealer_upcard, usable_ace)`.
  - Hi‑Lo observation: `(player_sum, dealer_upcard, usable_ace, tc_bucket, phase)` where `phase ∈ {0,1}`. In Phase 0 before cards are dealt, the tuple is `(0,0,0, tc_bucket, 0)`; in Phase 1 it is the regular sums plus the TC bucket and `phase=1`.
  - Action space is `Discrete(max(2, n_bets))`. In Phase 0, valid actions are bet indices; in Phase 1, valid actions are {0,1}. You can query `env.get_valid_actions_idxs()` for the current phase.

- Rewards and natural payout
  - Vanilla: unit‑sized bets; optional natural payout control.
  - Hi‑Lo variant: rewards are scaled by the selected bet multiplier.
    - On stick, reward is `cmp(player, dealer) * current_bet`, with optional natural bonus.
    - On bust, reward is `-1 * current_bet` immediately.
  - Natural handling (mirrors Gymnasium semantics):
    - `sab=True`: a natural is an automatic win but pays 1× (Sutton & Barto definition).
    - `sab=False` and `natural=True`: natural win pays 1.5×; otherwise 1×.

## Environment API

Constructor (key arguments):
- `BlackjackEnv(natural=False, sab=False, num_decks=1, tc_min=-10, tc_max=10)`
  - `num_decks`: size of the shoe used by `Deck`.
  - `tc_min`, `tc_max`: inclusive integer bounds for TC bucketing. The number of TC buckets is `tc_max - tc_min + 1`.
  - `tc_bucket_names`: pretty string labels for the buckets, e.g., `('-3', '-2', '-1', '+0', '+1', '+2', '+3')` for `[-3..+3]`.
  - `bet_multipliers`: numpy array of bet sizing options (default `[1.0, 2.0, 4.0]`).

Key methods/attributes:
- `reset()` → `(obs, info)` starts in Phase 0 (betting). No cards are dealt in Phase 0.
- `step(action)` → `(obs, reward, terminated, truncated, info)`
  - Phase 0 (bet): `action` selects an index into `bet_multipliers`. Returns Phase 1 observation with reward `0.0`.
  - Phase 1 (play): `action ∈ {0=stick, 1=hit}`; reward is scaled by the chosen bet.
- `get_valid_actions_idxs()` returns the valid action indices for the current phase.
- `unwrapped.tc_bucket_names` and `unwrapped.bet_multipliers` expose labels and bet sizes for visualization.

True‑count details:
- Running Hi‑Lo is updated for all visible draws (both player cards, dealer’s upcard; dealer’s hole card is counted when revealed on stick; and subsequent dealer hits).
- `decks_left = max(shoe_size / 52, 1.0)` to avoid spikes when the shoe is nearly empty.
- TC is rounded to the nearest integer and clamped to `[tc_min, tc_max]` before mapping to the bucket index.

## Minimal example

```python
import gymnasium as gym
from hi_lo_variant.env import BlackjackEnv

env = BlackjackEnv(num_decks=3, tc_min=-3, tc_max=3, natural=True)
obs, info = env.reset()              # Phase 0: (0, 0, 0, tc_bucket, 0)

# Phase 0: choose a bet (index into bet_multipliers)
a_bet = 1  # e.g., 1.0x if bet_multipliers=[0.5, 1.0, 2.0, 4.0]
obs, r, terminated, truncated, info = env.step(a_bet)

# Phase 1: play the hand with actions {0=stick, 1=hit}
done = False
while not done:
    action = 0  # stick (example)
    obs, r, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

## Notes and tips

- The observation space’s TC bucket is dynamic and adjusts automatically to your chosen `[tc_min..tc_max]` range. Your agent should use `env.observation_space.spaces[3].n` and `env.unwrapped.tc_bucket_names` at runtime.
- Rewards are always scaled by the selected bet multiplier for the episode.
- If you require a different cut card (reshuffle threshold), adjust the `cut_frac` in `reset()`.
- The included notebooks (`sarsa.ipynb`, `q-learning.ipynb`) demonstrate training with a split betting/playing policy, visualizations for both Q tables, and analysis by TC bucket.

### Split / Double / Surrender variant (hi_lo_variant/add_split)

The `add_split` environment extends play actions with `SPLIT`, `DOUBLE`, and `SURRENDER`, and supports multiple player hands after splits. Key rules:

- Split eligibility: a hand with exactly two cards can be split if the ranks are identical or both are 10‑value ranks (10/J/Q/K).
- Maximum hands: up to 4 hands (i.e., up to 3 splits across the round).
- Ace split restrictions: an Ace pair may be split only once per round. After splitting Aces, each resulting hand is "no‑hit":
  - HIT is disallowed on those hands.
  - DOUBLE and SURRENDER are also disallowed (first action is considered done on those hands).
  - STICK remains available to proceed to the next hand.
- Double down: only allowed as the first decision on a two‑card hand; the hand bet is doubled and exactly one card is drawn.
- Late surrender: only allowed as the first decision on the initial two‑card single hand; loses half the bet unless the dealer has a natural, in which case the full bet is lost.
- Rewards are incremental per step: bust penalties are emitted immediately; final dealer resolution step adds only outcomes for unresolved hands. Use `env.cumulative_episode_reward()` if you need the episode’s net reward for your training loop.

## Compatibility

- Built against Gymnasium’s Blackjack semantics and standard Python scientific stack (NumPy, Matplotlib). The environment is a plain `gymnasium.Env` and can be wrapped (e.g., `RecordEpisodeStatistics`).
