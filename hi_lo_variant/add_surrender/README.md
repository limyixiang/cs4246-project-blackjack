# Hi-Lo Blackjack (Betting + Late Surrender)

This variant extends the Hi-Lo, multi-deck Blackjack environment with:

-   A two-phase episode (Phase 0 = betting, Phase 1 = playing)
-   Late surrender as an additional action during the playing phase

## Key differences

-   Action space size: `Discrete(max(3, n_bets))`
    -   Phase 0 (betting): valid actions are bet indices `0..n_bets-1`.
    -   Phase 1 (playing):
        -   `0 = stick`, `1 = hit`, `2 = surrender` (available only as the first decision with exactly 2 cards).
-   Observation space: `(player_sum, dealer_upcard_value, usable_ace, true_count_bucket, phase)` where `phase ∈ {0,1}`.
-   Rewards are scaled by the chosen bet multiplier.
-   Late surrender (casino-style):
    -   Reveal dealer’s hole card.
    -   If the dealer has a natural blackjack, the player loses the full bet (`−1 × bet`).
    -   Otherwise the player loses half the bet (`−0.5 × bet`).
-   Naturals are auto-resolved before any action is applied.

## Minimal example

```python
from hi_lo_variant.add_surrender import BlackjackEnv

env = BlackjackEnv(num_decks=3, tc_min=-3, tc_max=3, natural=True)

# Phase 0: choose a bet (index into env.unwrapped.bet_multipliers)
obs, info = env.reset()            # (0, 0, 0, tc_bucket, 0)
obs, r, terminated, truncated, _ = env.step(1)  # bet index 1

# Phase 1: choose among {0=stick, 1=hit, 2=surrender}
obs, r, terminated, truncated, _ = env.step(2)  # late surrender
```

## Notes

-   Query `env.get_valid_actions_idxs()` to know the allowed actions at any point.
-   Surrender is only valid as the first decision when the player has exactly two cards.
-   If the player has a natural blackjack, the hand is immediately resolved and the chosen action is ignored.
-   The true-count bucket labels are available via `env.unwrapped.tc_bucket_names`.
