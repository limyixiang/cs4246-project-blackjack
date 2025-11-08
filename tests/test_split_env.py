from collections import deque

from hi_lo_variant.add_split.env import BlackjackEnv, PlayActions


def test_split_and_double_reward_sequence():
    """Integration test for split then double on second hand.

    Scenario construction (deck order left->right draw order):
    Player initial: 8H, 8D (pair to split)
    Dealer initial: 5C (upcard), 9S (hole)  => dealer starts at 14, will need one hit
    Split draws: 2C (for first split hand -> 8+2=10), 3C (for second split hand -> 8+3=11)
    Double second hand draw: 7D (second hand total 18 with doubled bet)
    Dealer hit: 6H (dealer final total 20)

    Outcomes:
      Hand1: 10 < 20  => -1 bet
      Hand2 (doubled): 18 < 20  => -2 bet (bet doubled)
      Total final step reward: -3.0
    """
    env = BlackjackEnv()
    
    # Reset first, then override deck so the environment doesn't reshuffle away our sequence
    obs, _ = env.reset()
    # Override deck with controlled sequence (only first 8 cards matter for this test).
    forced_cards = ['8H','8D','5C','9S','2C','3C','7D','6H']
    env.deck.deck = deque(forced_cards)

    # Phase 0: place minimum bet (index 0)
    obs, r, terminated, truncated, info = env.step(0)
    assert terminated is False and r == 0.0

    # Split action
    obs, r, terminated, truncated, info = env.step(PlayActions.SPLIT)
    assert r == 0.0 and terminated is False
    # After split we should have 2 hands; active hand index 0
    assert obs[6] == 2  # total number of hands
    assert obs[5] == 0  # active hand index

    # Stick first hand (total 10) -> move to next hand
    obs, r, terminated, truncated, info = env.step(PlayActions.STICK)
    assert r == 0.0 and terminated is False
    assert obs[5] == 1  # active hand index now second hand

    # Double second hand -> draws 7D, end of round triggers dealer play & final reward
    obs, r, terminated, truncated, info = env.step(PlayActions.DOUBLE)
    assert terminated is True
    assert r == -3.0, f"Expected total reward -3.0, got {r}"

    # Validate internal bet bookkeeping: second hand bet doubled
    assert env.player.hand_bets == [1.0, 2.0]
