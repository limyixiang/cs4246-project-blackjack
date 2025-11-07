import random
from collections import deque

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

    def draw_card(self):
        if not self.deck:
            raise IndexError("Deck is empty: no card to draw")
        return self.deck.popleft()
    
    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]
    
    def size(self):
        return len(self.deck)
    