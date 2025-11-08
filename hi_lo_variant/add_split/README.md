# Hi‑Lo Blackjack with Split / Double / Surrender (add_split)

This folder provides a Blackjack environment that extends the Hi‑Lo variant with play options for SPLIT, DOUBLE, and (late) SURRENDER, plus multiple concurrent hands after splits. It is compatible with Gymnasium’s API (`gymnasium.Env`).

## Highlights

- Finite multi‑deck shoe with reshuffle (cut card) and Hi‑Lo running count
- True‑count bucket in the observation for count‑aware agents
- Two‑phase episodes: Phase 0 (bet sizing) then Phase 1 (play)
- Play actions: STICK, HIT, SURRENDER (late), DOUBLE, SPLIT
- Splitting rules including Ace‑split restrictions (no hit/double/surrender on split Aces)
- Incremental rewards per step; a helper to read back the cumulative episode reward

## Import

```python
from hi_lo_variant.add_split.env import BlackjackEnv, PlayActions
```

## Observation and action spaces

- Observation (7‑tuple):
  `(player_sum, dealer_upcard_value, usable_ace, true_count_bucket, phase, active_hand_index, total_hands)`
  - `phase`: 0 (betting) or 1 (playing)
  - In Phase 0 (before the initial deal): `(0, 0, 0, tc_bucket, 0, 0, 0)`
- Action space: a single `Discrete(n)` whose valid meaning depends on the phase. Query `env.get_valid_actions_idxs()` for allowed actions at any time.
  - Phase 0 (betting): bet indices `0..len(bet_multipliers)-1`
  - Phase 1 (playing):
    - `0`: STICK
    - `1`: HIT
    - `2`: SURRENDER (late; only on initial unsplit two‑card hand and only as the first decision)
    - `3`: DOUBLE (only as the first decision on a two‑card hand; exactly one card drawn)
    - `4`: SPLIT (only before any action on the current hand, if eligible; see splitting rules below)

## Key rules and behaviors

- Shoe and Hi‑Lo
  - Real shoe of `num_decks` decks; reshuffles when cards remaining drop below an internal threshold
  - Hi‑Lo running count: 2–6 = +1, 7–9 = 0, 10/J/Q/K/A = −1
  - True count bucket: running count divided by decks left (truncated to int), clamped to `[tc_min, tc_max]`
- Dealer policy
  - Dealer hits on soft 17 (i.e., hits when total is 17 with a usable Ace)
- Betting (Phase 0)
  - Choose a bet index into `bet_multipliers` (defaults to `[1.0, 2.0, 4.0]`)
  - Reward is `0.0`; observation transitions to Phase 1 after dealing
- Late surrender
  - Only allowed as the first decision on the initial unsplit two‑card hand
  - If dealer has a natural: lose full bet (−1×)
  - Otherwise: lose half the bet (−0.5×)
- Double down
  - Only allowed as the first decision on a two‑card hand
  - Hand bet is doubled; exactly one card is drawn for that hand
- Splitting
  - Eligible if the current hand has exactly two cards of (a) the same rank or (b) both 10‑value ranks (10/J/Q/K)
  - Max 4 hands total (i.e., up to 3 splits across the round)
  - Splitting Aces:
    - Only one Ace split per round is allowed
    - After splitting Aces, each resulting hand is marked as no‑hit:
      - HIT is disallowed
      - DOUBLE and SURRENDER are also disallowed (first action is considered done for those hands)
      - STICK remains available to move on to the next hand

## Rewards

Rewards are incremental per step. The sum of all step rewards over the episode equals the net outcome.

- Bust: −1 × hand bet (emitted immediately on the HIT/DOUBLE that causes bust)
- Win/Lose after dealer resolution: ±1 × hand bet
- Draw (push): 0
- Natural win (only when there is a single initial hand and dealer is not natural):
  - `sab=False` and `natural=True`: +1.5 × bet
  - Otherwise: +1.0 × bet

Training convenience: call `env.cumulative_episode_reward()` to retrieve the episode’s net reward (sum of all incremental rewards emitted so far in the episode).

## API surface

```python
env = BlackjackEnv(natural=False, sab=False, num_decks=1, tc_min=-10, tc_max=10)

obs, info = env.reset()                  # Start in Phase 0 (betting)
obs, r, terminated, truncated, info = env.step(bet_index)

valid_actions = list(env.get_valid_actions_idxs())
obs, r, terminated, truncated, info = env.step(PlayActions.HIT)

total_ep_reward = env.cumulative_episode_reward()
```

- `bet_multipliers`: numpy array of bet sizing options (default `[1.0, 2.0, 4.0]`)
- `tc_bucket_names`: pretty string labels for TC buckets (e.g., `('-3', '-2', '-1', '+0', '+1', '+2', '+3')`)

## Deterministic scenarios (testing)

You can inject a fixed draw order for deterministic tests by replacing the internal deque on the `Deck`:

```python
from collections import deque
from hi_lo_variant.add_split.env import BlackjackEnv, PlayActions

env = BlackjackEnv()
obs, _ = env.reset()
# Left to right is draw order: player1, player2, dealer_up, dealer_hole, ...
forced_cards = [
    '8H','8D',  # player two cards (pair, can split)
    '5C','9S',  # dealer up and hole
    '2C','3C',  # split draws: first new hand gets 2C, second gets 3C
    '7D','6H'   # later draws (e.g., double or dealer hits) as needed
]
env.deck.deck = deque(forced_cards)

# Phase 0 bet
obs, r, term, trunc, info = env.step(0)
# Split
obs, r, term, trunc, info = env.step(PlayActions.SPLIT)
```

## Tips

- Check allowed actions with `env.get_valid_actions_idxs()`; do not assume an action is always valid.
- For split hands, use `obs[5]` (active hand index) and `obs[6]` (total hand count) to sequence your policy.
- The true count bucket is computed dynamically from running count and estimated decks remaining; use `env.unwrapped.tc_bucket_names` for display/debug.

## Notes

- This environment hits soft 17.
- The action integer encoding overlaps across phases by design; validity is phase‑dependent.
- If you need auto‑stand on split Aces rather than explicit STICK, this can be added with a small change.
