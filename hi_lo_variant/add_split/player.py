from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

# Support both running as a script from this folder and importing as a package
try:
    from deck import Deck
    from hand import Hand
except ImportError:  # pragma: no cover - fallback for package import
    from .deck import Deck
    from .hand import Hand

class Player(ABC):
    """Abstract player that manages one or more blackjack hands.

    Responsibilities
    - Owns the list of Hand instances for the current round
    - Tracks which hand is currently active
    - Tracks per-hand bet multipliers (to support double, split)
    - Tracks whether the current hand has taken its first action

    Subclasses can override policies or bookkeeping, but the base API
    should remain stable for the environment to use.
    """

    def __init__(self) -> None:
        self.hands: List[Hand] = []
        self.active_index: int = 0
        # Per-hand bet multipliers; start with [1.0] then duplicate/modify on split/double
        self.hand_bets: List[float] = []
        # Tracks whether the first action has been taken for each hand
        self.first_action_done: List[bool] = []
        self.accumulated_reward: float = 0.0
        self.ace_split_done: bool = False  # Track if an Ace split has been done

    @abstractmethod
    def start_new_round(self, initial_hand: Hand, base_bet: float) -> None:
        """Initialize the player for a new round (no split yet)."""
        raise NotImplementedError

    # ----- convenience helpers used by env -----
    @property
    def active_hand(self) -> Hand:
        return self.hands[self.active_index]
    
    @property
    def active_bet(self) -> float:
        return self.hand_bets[self.active_index]

    def set_first_action_done(self) -> None:
        self.first_action_done[self.active_index] = True

    def is_first_action(self) -> bool:
        return not self.first_action_done[self.active_index]

    def can_split(self) -> bool:
        """Whether the active hand can be split.
        Default rule: exactly two cards of the same rank OR both 10-value cards,
        and only before first action.
        Ace split: only one split allowed (up to 2 hands with one Ace each).
        """
        if len(self.hands) >= 4: # limit to max 4 hands (or 3 splits)
            return False
        if not self.is_first_action():
            return False
        hand = self.active_hand
        if hand.has_two_aces():
            # Only allow one split for Aces
            return self.ace_split_done == False
        return hand.can_split()

    def do_split(self, deck: Deck) -> Tuple[str, str]:
        """Perform a split on the active hand.
        Returns the two newly drawn card strings (one for each new hand)
        so the caller can update any external counters.
        """
        assert self.can_split(), "Split not allowed in current state"
        i = self.active_index
        old = self.hands[i]
        is_ace_pair = old.has_two_aces()
        if is_ace_pair:
            self.ace_split_done = True

        # Create two new hands, each starting with one of the originals
        h1 = Hand([old[0]])
        h2 = Hand([old[1]])

        # Replace the old hand in-place and insert the second new hand right after it.
        # This keeps ordering consistent and preserves active_index pointing at the
        # first split hand.
        self.hands[i] = h1
        self.hands.insert(i + 1, h2)

        # Duplicate bet for the split hand at the same indices
        base = self.hand_bets[i]
        self.hand_bets[i] = base
        self.hand_bets.insert(i + 1, base)

        # Maintain first_action_done bookkeeping for both new hands
        # If this was an Ace split, many casinos forbid hitting/doubling on the
        # newly created hands; mark their first-action as done to prevent double/surrender
        # and set the hand-level `no_hit` flag on each new Hand so env can disallow HIT.
        if is_ace_pair:
            self.first_action_done[i] = True
            self.first_action_done.insert(i + 1, True)
            h1.no_hit = True
            h2.no_hit = True
        else:
            self.first_action_done[i] = False
            self.first_action_done.insert(i + 1, False)

        # Deal one card to each split hand to make them two cards each
        c1 = deck.draw_card()
        h1.append(c1)
        c2 = deck.draw_card()
        h2.append(c2)
        return c1, c2

    def advance_to_next_hand(self) -> bool:
        """Advance to next hand if available. Returns True if advanced, False otherwise."""
        if self.active_index + 1 < len(self.hands):
            self.active_index += 1
            return True
        return False

    def apply_double_down(self) -> None:
        """Double the bet for the active hand (player will receive exactly one card outside)."""
        assert self.is_first_action(), "Double down only allowed as first action"
        self.hand_bets[self.active_index] *= 2.0
        self.set_first_action_done()

    def record_reward(self, reward: float) -> None:
        """Accumulate reward for the player over multiple hands."""
        self.accumulated_reward += reward


class SimplePlayer(Player):
    """Concrete Player with default bookkeeping.

    Keeps minimal policy/logic; the environment orchestrates the card flows.
    """

    def start_new_round(self, initial_hand: Hand, base_bet: float) -> None:
        self.hands = [initial_hand]
        self.active_index = 0
        self.hand_bets = [base_bet]
        self.first_action_done = [False]
        self.accumulated_reward = 0.0
        self.ace_split_done = False
