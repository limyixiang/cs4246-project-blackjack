from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

try:
    from deck import Deck
    from env import Hand, _rank, TEN_VALUE_RANKS
except Exception:  # pragma: no cover - package import fallback
    from .deck import Deck
    from .env import Hand, _rank, TEN_VALUE_RANKS


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

    @abstractmethod
    def start_new_round(self, initial_hand: Hand, base_bet: float) -> None:
        """Initialize the player for a new round (no split yet)."""
        raise NotImplementedError

    # ----- convenience helpers used by env -----
    @property
    def active_hand(self) -> Hand:
        return self.hands[self.active_index]

    def set_first_action_done(self) -> None:
        self.first_action_done[self.active_index] = True

    def is_first_action(self) -> bool:
        return not self.first_action_done[self.active_index]

    def can_split(self) -> bool:
        """Whether the active hand can be split.
        Default rule: exactly two cards of the same rank OR both 10-value cards,
        and only when there is a single hand (pre-split) and before first action.
        """
        if len(self.hands) != 1:
            return False
        hand = self.active_hand
        if len(hand) != 2:
            return False
        if not self.is_first_action():
            return False
        r1, r2 = _rank(hand[0]), _rank(hand[1])
        if r1 == r2:
            return True
        if r1 in TEN_VALUE_RANKS and r2 in TEN_VALUE_RANKS:
            return True
        return False

    def do_split(self, deck: Deck) -> Tuple[str, str]:
        """Perform a split on the active hand.
        Returns the two newly drawn card strings (one for each new hand)
        so the caller can update any external counters.
        """
        assert self.can_split(), "Split not allowed in current state"
        old = self.active_hand
        # Create two new hands, each starting with one of the originals
        h1 = Hand([old[0]])
        h2 = Hand([old[1]])
        # Replace hands list with the two new hands
        self.hands = [h1, h2]
        self.active_index = 0
        # Duplicate bet for the split hand
        base = self.hand_bets[0]
        self.hand_bets = [base, base]
        self.first_action_done = [False, False]
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
        self.hand_bets[self.active_index] *= 2.0
        self.set_first_action_done()


class SimplePlayer(Player):
    """Concrete Player with default bookkeeping.

    Keeps minimal policy/logic; the environment orchestrates the card flows.
    """

    def start_new_round(self, initial_hand: Hand, base_bet: float) -> None:
        self.hands = [initial_hand]
        self.active_index = 0
        self.hand_bets = [base_bet]
        self.first_action_done = [False]
