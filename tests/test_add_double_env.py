import unittest
from collections import deque

from hi_lo_variant.add_double.env import BlackjackEnv, PlayActions, _rank


def set_deck_order(env: BlackjackEnv, order):
    """Replace the underlying deck order with our custom sequence.
    Order should be a list of card strings in the exact draw order.
    """
    env.deck.deck = deque(order)


class TestDoubleDown(unittest.TestCase):
    def setUp(self):
        self.env = BlackjackEnv(natural=False, sab=False, num_decks=1)
        # Ensure phase starts at betting and avoid automatic reshuffle by setting deck after reset
        self.env.reset()

    def test_double_bust_no_hole_reveal(self):
        # Player: K,10 (20) | Dealer up: 5, hole: 9 | Double draw: 2 -> 22 (bust)
        order = [
            'KH', '10H',  # player two cards
            '5H', '9H',   # dealer up, dealer hole
            '2H',         # player's double draw (busts)
            '3C', '4C', '6D'  # filler
        ]
        # Reset first, then inject our deck order so reset does not overwrite it
        obs, _ = self.env.reset()
        set_deck_order(self.env, order)

        # Place bet index 0 (bet multiplier = 1.0)
        obs, r, term, trunc, _ = self.env.step(0)
        self.assertFalse(term)

        # Double should be allowed as first decision with 2 cards
        self.assertIn(int(PlayActions.DOUBLE), list(self.env.get_valid_actions_idxs()))

        # Double and bust; dealer hole should NOT be revealed to the count
        obs, r, term, trunc, _ = self.env.step(int(PlayActions.DOUBLE))
        self.assertTrue(term)
        self.assertAlmostEqual(r, -2.0, places=6)

        # Hi-lo count progression: K(-1) + 10(-1) + 5(+1) + 2(+1) = 0
        self.assertEqual(self.env.hi_lo_count, 0)

        # Player received exactly one extra card
        self.assertEqual(len(self.env.player), 3)

    def test_double_win_reveals_hole_and_no_dealer_draw(self):
        # Player: 5,6 (11) | Dealer up: 7, hole: 10 (17) | Double draw: 10 -> 21 (win)
        order = [
            '5H', '6H',   # player two cards
            '7H', '10H',  # dealer up, dealer hole (17)
            '10C',        # player's double draw -> 21
            '2C', '3D', '4S'  # filler
        ]
        obs, _ = self.env.reset()
        set_deck_order(self.env, order)
        obs, r, term, trunc, _ = self.env.step(0)
        self.assertFalse(term)
        self.assertIn(int(PlayActions.DOUBLE), list(self.env.get_valid_actions_idxs()))

        obs, r, term, trunc, _ = self.env.step(int(PlayActions.DOUBLE))
        self.assertTrue(term)
        # Win: +2 * bet (bet=1.0)
        self.assertAlmostEqual(r, 2.0, places=6)

        # Count: 5(+1)+6(+1)+7(0)+10(-1)+10(-1 revealed) = 0
        self.assertEqual(self.env.hi_lo_count, 0)

        # Dealer should not draw further (already at 17)
        self.assertEqual(len(self.env.dealer), 2)

    def test_double_not_allowed_after_hit(self):
        # Player: 2,3 | Dealer: 9,8 (17) | Hit draws 4 -> 9 (not terminal). Double should no longer be valid.
        order = [
            '2H', '3H',
            '9H', '8H',
            '4H',       # player's hit
            '2C', '3C', '4C'
        ]
        obs, _ = self.env.reset()
        set_deck_order(self.env, order)
        obs, r, term, trunc, _ = self.env.step(0)
        self.assertFalse(term)
        # First decision: double is allowed
        self.assertIn(int(PlayActions.DOUBLE), list(self.env.get_valid_actions_idxs()))

        # Take a HIT
        obs, r, term, trunc, _ = self.env.step(int(PlayActions.HIT))
        self.assertFalse(term)

        # Now only STICK/HIT allowed
        valid = list(self.env.get_valid_actions_idxs())
        self.assertNotIn(int(PlayActions.DOUBLE), valid)
        self.assertIn(int(PlayActions.STICK), valid)
        self.assertIn(int(PlayActions.HIT), valid)


if __name__ == '__main__':
    unittest.main()
