from collections import deque

from hi_lo_variant.add_split.env import BlackjackEnv, PlayActions


def test_ace_split_no_hit_allowed():
    # Player: AH, AD (pair to split)
    # Dealer up: 5C, hole: 9S
    # Split draws: 2C, 3C (for each new hand)
    env = BlackjackEnv()
    obs, _ = env.reset()
    forced_cards = ['AH', 'AD', '5C', '9S', '2C', '3C', '6H']
    env.deck.deck = deque(forced_cards)

    # Place bet
    obs, r, terminated, truncated, info = env.step(0)
    assert terminated is False

    # Split
    obs, r, terminated, truncated, info = env.step(PlayActions.SPLIT)
    assert r == 0.0 and terminated is False

    # After splitting Aces, the active hand should be marked as no-hit and should
    # not allow HIT/DOUBLE/SURRENDER as valid actions.
    valid = list(env.get_valid_actions_idxs())
    assert int(PlayActions.HIT) not in valid, f"HIT should be disallowed after splitting aces, got {valid}"
    assert int(PlayActions.DOUBLE) not in valid, f"DOUBLE should be disallowed after splitting aces, got {valid}"
    assert int(PlayActions.SURRENDER) not in valid, f"SURRENDER should be disallowed after splitting aces, got {valid}"


def test_cumulative_reward_helper_matches_sum():
    # Simple bust scenario where the player hits and busts; accumulated reward should be -1.0
    env = BlackjackEnv()
    obs, _ = env.reset()
    forced_cards = ['KH', '5H', '5C', '9C', '10H']  # player K,5 | dealer 5,9 | hit 10 -> bust
    env.deck.deck = deque(forced_cards)

    obs, r, term, trunc, info = env.step(0)
    assert term is False

    obs, r, term, trunc, info = env.step(PlayActions.HIT)
    assert term is True
    assert r == -1.0

    # cumulative helper should match the player's accumulated_reward
    assert env.cumulative_episode_reward() == env.player.accumulated_reward
    assert env.cumulative_episode_reward() == -1.0
