from player import SimplePlayer
from hand import Hand

# Split alignment test (was broken before)
p = SimplePlayer()
p.start_new_round(Hand(["8H","8D"]), 1.0)

# simulate prior extra hand
p.hands.append(Hand(["9H","9D"]))
p.hand_bets.append(2.0)
p.first_action_done.append(False)

# split; ensure alignment preserved
p.active_index = 1
from deck import Deck
d = Deck(1); d.shuffle()
p.do_split(d)
print(p.hands)

assert len(p.hands) == 3
assert len(p.hand_bets) == 3
assert len(p.first_action_done) == 3
assert p.hand_bets[p.active_index] == 2.0
assert not p.first_action_done[p.active_index]
# assert "9H" in p.hands[0]     # p.active_index = 0
# assert "9D" in p.hands[0]     # p.active_index = 0
# assert "8H" in p.hands[1]     # p.active_index = 0
# assert "8D" in p.hands[2]     # p.active_index = 0
assert "8H" in p.hands[0]       # p.active_index = 1
assert "8D" in p.hands[0]       # p.active_index = 1
assert "9H" in p.hands[1]       # p.active_index = 1
assert "9D" in p.hands[2]       # p.active_index = 1

# Ace-split resets per round
p.ace_split_done = True
p.start_new_round(Hand(["AH","AD"]), 1.0)
assert p.can_split()  # should be allowed again in a new round
