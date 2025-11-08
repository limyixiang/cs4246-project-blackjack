import random
from collections import deque

# Support both running as a script from this folder and importing as a package
try:
    from hand import Hand
except ImportError:  # pragma: no cover - fallback for package import
    from .hand import Hand

class Deck:
    def __init__(self, num_decks):
        cards = []
        suits = ["C", "D", "H", "S"]
        ranks = [str(i) for i in range(2, 11)] + ['J', 'Q', 'K', 'A']
        for _ in range(num_decks):
            for suit in suits:
                for rank in ranks:
                    cards.append(rank + suit)   # e.g., '2H', 'AD', 'QS'
        self.deck = deque(cards)
        assert len(self.deck) == 52 * num_decks
        assert all(c in cards for c in ['2H', 'AD', 'QS'])

    def shuffle(self):
        cards = list(self.deck)
        random.shuffle(cards)
        self.deck = deque(cards)

    def draw_card(self) -> str:
        if not self.deck:
            raise IndexError("Deck is empty: no card to draw")
        return self.deck.popleft()
    
    def draw_hand(self) -> Hand:
        return Hand([self.draw_card(), self.draw_card()])
    
    def size(self) -> int:
        return len(self.deck)
    