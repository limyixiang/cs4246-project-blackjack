# Hi-Lo Variant: Add Double Down

This folder contains a Blackjack environment that extends the base hi–lo variant by adding a Double Down action. It keeps the betting phase and true-count observation features while allowing the player to double their initial bet, receive exactly one additional card, and then stand.

-   Module: `hi_lo_variant.add_double.env`
-   Class: `BlackjackEnv`
-   Enum: `PlayActions` with actions `STICK=0`, `HIT=1`, `SURRENDER=2`, `DOUBLE=3`

## What’s new

-   Double Down (action `3`) is allowed only as the first decision when the player has exactly two cards.
-   On Double Down, the player draws exactly one card and then stands automatically.
-   The final outcome is multiplied by `2 × bet`.

## Key rules and behavior

-   Two phases per episode:
    -   Phase 0 (betting): choose a bet index from preset bet multipliers.
    -   Phase 1 (playing): choose play actions depending on the current state.
-   Dealer reveals hole card and draws to 17 or more only when needed (stick resolution or on non-bust doubles):
    -   After STICK, reveal hole and finish dealer play.
    -   After SURRENDER (late), reveal hole to check if dealer has a natural.
    -   After DOUBLE, if player busts on the single draw, the dealer’s hole is not revealed; otherwise reveal and resolve normally.
-   Naturals: if player has a natural blackjack, the hand auto-resolves before any play actions.
-   Dealer stands on 17 (draws while sum < 17).
-   Reshuffle: deck reshuffles when fewer than 25% of cards remain.

## Observation space

Observation is a 5‑tuple `(player_sum, dealer_upcard_value, usable_ace, true_count_bucket, phase)`:

-   `player_sum` ∈ {0..31} (0 only during betting phase before cards are dealt)
-   `dealer_upcard_value` ∈ {0..10} (0 during betting phase; Ace is 1, face/10s are 10)
-   `usable_ace` ∈ {0,1}
-   `true_count_bucket` ∈ {0..(tc_max - tc_min)} (see hi‑lo/true count below)
-   `phase` ∈ {0 (betting), 1 (playing)}

## Action space

Actions depend on phase:

-   Phase 0 (betting): choose a bet index `i` in `bet_multipliers`. Default multipliers are `[1.0, 2.0, 4.0]`.
-   Phase 1 (playing):
    -   `0` = STICK (stand)
    -   `1` = HIT (draw a card)
    -   `2` = SURRENDER (late; only allowed as first decision with exactly two cards)
    -   `3` = DOUBLE (only allowed as first decision with exactly two cards; draw one card then stand)

Use `env.get_valid_actions_idxs()` to query the currently valid actions subset.

## Rewards (scaled by selected bet multiplier)

-   Win: `+1 × bet`
-   Lose: `−1 × bet`
-   Draw: `0`
-   Natural win (player has blackjack, dealer does not):
    -   `+1.5 × bet` if `natural=True` and `sab=False`
    -   `+1 × bet` otherwise (Sutton & Barto alignment)
-   Late surrender: `−0.5 × bet` (unless dealer has a natural, then `−1 × bet`)
-   Double down: result is multiplied by `2 × bet`

## Hi–Lo count and true count

-   Hi–Lo updates on every revealed/drawn card:
    -   +1 for ranks 2–6
    -   0 for ranks 7–9
    -   −1 for 10/J/Q/K/A
-   The observation includes a true count bucket computed from `hi_lo_count / decks_left`, truncated to an integer and clamped to `[tc_min, tc_max]`, then shifted to start at 0.
-   Hole card counting behavior:
    -   Revealed during STICK resolution and SURRENDER.
    -   For DOUBLE: revealed only if the player does not bust on the one-card draw.

## Constructor and config

```python
BlackjackEnv(
    natural: bool = False,
    sab: bool = False,
    num_decks: int = 1,
    tc_min: int = -10,
    tc_max: int = 10,
)
```

-   `natural`: pay 3:2 on player natural blackjack (unless `sab=True`).
-   `sab`: match Sutton & Barto rules (overrides `natural` payout behavior).
-   `num_decks`: number of standard 52-card decks concatenated.
-   `tc_min`/`tc_max`: clamp range for true count integer buckets.

Bet multipliers are currently fixed inside the class to `[1.0, 2.0, 4.0]`.

## Basic usage

```python
from hi_lo_variant.add_double.env import BlackjackEnv, PlayActions

env = BlackjackEnv(natural=False, sab=False, num_decks=1)
obs, info = env.reset()

# Phase 0: place a bet (index 0 -> 1×, 1 -> 2×, 2 -> 4×)
obs, reward, terminated, truncated, info = env.step(0)

# Phase 1: valid actions now include DOUBLE if first decision with exactly two cards
if int(PlayActions.DOUBLE) in env.get_valid_actions_idxs():
    obs, reward, terminated, truncated, info = env.step(int(PlayActions.DOUBLE))
else:
    obs, reward, terminated, truncated, info = env.step(int(PlayActions.HIT))
```

## Notebook

See `sarsa.ipynb` in this folder for learning experiments that incorporate the Double Down action.

## Tests

There are unit tests for the Double Down behavior:

```powershell
# From the repo root
C:/repos/cs4246-project/.venv/Scripts/python.exe -m unittest discover -v -s tests
```

Covered cases include:

-   Double and player bust: −2× bet; dealer hole not revealed to the count.
-   Double win with dealer already on 17: +2× bet; dealer hole revealed; no extra dealer draw.
-   Double not allowed after a prior HIT.

## Requirements

Dependencies are listed in `requirements.txt`. Key packages:

-   gymnasium
-   numpy
-   pandas, matplotlib, tqdm, pygame, ipywidgets (for notebooks/visualizations)

Install into your virtual environment, then run notebooks or tests as shown above.
