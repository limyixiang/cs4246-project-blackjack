ACE_RANKS = {"A"} 
TEN_VALUE_RANKS = {"10", "J", "Q", "K"}  # ranks worth 10

def _rank(card: str) -> str:
    """Return the rank portion of a card like '10H' -> '10', 'AS' -> 'A'."""
    return card[:-1]  # last char is suit

def _hard_value(card: str) -> int:
    """Blackjack hard value (Ace as 1)."""
    r = _rank(card)
    if r in ACE_RANKS:
        return 1
    if r in TEN_VALUE_RANKS:
        return 10
    return int(r)  # '2'..'9'

class Hand:
    """A blackjack hand abstraction wrapping a list of card strings.

    Contract
    - Cards are strings like 'AS', '10H'.
    - Iterable, indexable, and has len() like a list.
    - Provides helpers for totals, usable ace, bust, score, and natural.
    """

    def __init__(self, cards: list[str] | None = None):
        self._cards: list[str] = list(cards) if cards is not None else []
        # When a hand is created as the result of splitting two Aces,
        # casino rules typically prohibit hitting further on the resulting hands.
        # `no_hit` signals the environment/player to disallow HIT on this hand.
        self.no_hit: bool = False

    # --- list-like protocol ---
    def __iter__(self):
        return iter(self._cards)

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, idx):
        return self._cards[idx]

    def add(self, card: str):
        self._cards.append(card)

    # Keep compatibility with existing code paths that call .append
    append = add

    # --- blackjack helpers ---
    def hard_total(self) -> int:
        """Sum with all Aces counted as 1."""
        return sum(_hard_value(c) for c in self._cards)

    def usable_ace(self) -> int:
        """1 if an Ace can be counted as 11 without busting, else 0."""
        s = self.hard_total()
        has_ace = any(_rank(c) in ACE_RANKS for c in self._cards)
        return int(has_ace and s + 10 <= 21)

    def total(self) -> int:
        """Best total for the hand (treat one Ace as 11 if usable)."""
        s = self.hard_total()
        return s + 10 if self.usable_ace() else s

    def total_and_usable_ace(self) -> tuple[int, int]:
        s = self.hard_total()
        if any(_rank(c) in ACE_RANKS for c in self._cards) and s + 10 <= 21:
            return s + 10, 1
        return s, 0

    def is_bust(self) -> bool:
        return self.total() > 21

    def score(self) -> int:
        return 0 if self.is_bust() else self.total()

    def is_natural(self) -> bool:
        if len(self) != 2:
            return False
        ranks = {_rank(c) for c in self._cards}
        return bool((ranks & ACE_RANKS) and (ranks & TEN_VALUE_RANKS))
    
    def can_split(self) -> bool:
        """Whether this hand can be split.
        Criteria: exactly two cards of the same rank OR both 10-value cards.
        """
        if len(self) != 2:
            return False
        r1, r2 = _rank(self[0]), _rank(self[1])
        if r1 == r2:
            return True
        if r1 in TEN_VALUE_RANKS and r2 in TEN_VALUE_RANKS:
            return True
        return False
    
    def has_two_aces(self) -> bool:
        """Whether this hand consists of exactly two Aces."""
        if len(self) != 2:
            return False
        r1, r2 = _rank(self[0]), _rank(self[1])
        return r1 in ACE_RANKS and r2 in ACE_RANKS

    # --- printing helpers ---
    def __repr__(self) -> str:
        """Unambiguous representation useful for debugging.

        Shows card list, current best total, whether a usable ace exists,
        and whether the hand is a natural blackjack.
        """
        return (
            f"Hand(cards={self._cards!r}, total={self.total()}, "
            f"usable_ace={self.usable_ace()}, natural={self.is_natural()}, no_hit={self.no_hit})"
        )

    def __str__(self) -> str:
        """Human-friendly one-line description for printing.

        Examples:
          'AS 10H -> 21 (usable ace, natural)'
          '5H 9D -> 14'
        """
        cards_str = " ".join(self._cards)
        parts = [f"{cards_str} -> {self.total()}"]
        if self.usable_ace():
            parts.append("usable ace")
        if self.is_natural():
            parts.append("natural")
        if self.no_hit:
            parts.append("no-hit-after-split-aces")
        return " - ".join(parts)
    