import os, math
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import numpy as np

from deck import Deck

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

def cmp(a, b):
    """Returns +1 if a > b, 0 if a = b, -1 if a < b"""
    return float(a > b) - float(a < b)

def usable_ace(hand):
    """Returns 1 if hand has usable ace: A in hand and sum of hand + 10 <= 21"""
    s = sum(_hard_value(c) for c in hand)
    has_ace = any(_rank(c) in ACE_RANKS for c in hand)
    return int(has_ace and s + 10 <= 21)

def sum_hand(hand):
    """Returns highest total of current hand, forces ace to be used as 11"""
    s = sum(_hard_value(c) for c in hand)
    return s + 10 if usable_ace(hand) else s

def is_bust(hand):
    """Returns True if sum of hand > 21"""
    return sum_hand(hand) > 21

def score(hand):
    """Returns the score of the hand (0 if bust)"""
    return 0 if is_bust(hand) else sum_hand(hand)

def is_natural(hand):
    if len(hand) != 2:
        return False
    ranks = {_rank(c) for c in hand}
    return bool((ranks & ACE_RANKS) and (ranks & TEN_VALUE_RANKS))

def _hand_sum_and_usable_ace(hand):
    """
    Helper to compute both sum with and without ace counted as 11,
    and whether the hand has a usable ace.
    Returns (effective_sum, usable_ace: int)
    """
    s = sum(_hard_value(c) for c in hand)
    if any(_rank(c) in ACE_RANKS for c in hand) and s + 10 <= 21:
        return s + 10, 1
    return s, 0

class BlackjackEnv(gym.Env):
    """
    Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.

    ## Description
    The game starts with the dealer having one face up and one face down card,
    while the player has two face up cards. 
    All cards are drawn from a finite deck, consisting of a pre-determined number of decks. 
    The deck will be reset and shuffled when the number of cards remaining drops below a certain threshold.

    The card values are:
    - Face cards (Jack, Queen, King) have a point value of 10.
    - Aces can either count as 11 (called a 'usable ace') or 1.
    - Numerical cards (2-10) have a value equal to their number.

    The player has the sum of cards held. The player can request
    additional cards (hit) until they decide to stop (stick) or exceed 21 (bust,
    immediate loss).

    After the player sticks, the dealer reveals their facedown card, and draws cards
    until their sum is 17 or greater. If the dealer goes bust, the player wins.

    If neither the player nor the dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto [<a href="#blackjack_ref">1</a>].

    ## Action Space
    The action shape is `(1,)` in the range `{0, 1}` indicating
    whether to stick or hit.

    - 0: Stick
    - 1: Hit

    ## Observation Space
    The observation consists of a 5-tuple containing: the player's current sum,
    the value of the dealer's one showing card (1-10 where 1 is ace),
    whether the player holds a usable ace (0 or 1),
    the true count indicating the hi-lo count / number of decks remaining,
    and the state phase (betting or playing).

    The observation is returned as `(int(), int(), int(), int())`.

    ## Starting State
    The starting state is initialised with the following values.

    | Observation               | Values                        |
    |---------------------------|-------------------------------|
    | Player current sum        |  4, 5, ..., 21                |
    | Dealer showing card value |  1, 2, ..., 10                |
    | Usable Ace                |  0, 1                         |
    | True Count                |  0, ..., tc_max - tc_min      |

    ## Rewards
    - win game: +1
    - lose game: -1
    - draw game: 0
    - win game with natural blackjack:
    +1.5 (if <a href="#nat">natural</a> is True)
    +1 (if <a href="#nat">natural</a> is False)

    ## Episode End
    The episode ends if the following happens:

    - Termination:
    1. The player hits and the sum of hand exceeds 21.
    2. The player sticks.
    """
    def __init__(self, natural = False, sab = False, num_decks: int = 1, tc_min: int = -10, tc_max: int = 10):
        self.num_decks = num_decks
        self.tc_min, self.tc_max = tc_min, tc_max
        self.tc_bucket_names = ("≤-3", "-2", "-1", "0", "+1", "+2", "≥+3")
        self.bet_multipliers = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        self.n_bets = len(self.bet_multipliers)
        # 2 actions in playing phase and n_bet actions in betting phase
        self.action_space = spaces.Discrete(max(2, self.n_bets))
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),                    # player_sum
            spaces.Discrete(11),                    # dealer_upcard
            spaces.Discrete(2),                     # usable_ace
            spaces.Discrete(len(self.tc_bucket_names)),   # true-count bucket
            spaces.Discrete(2)                      # phase: 0=betting, 1=playing
        ))

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural

        # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
        self.sab = sab

        self.deck = Deck(self.num_decks)
        self.deck.shuffle()
        self.hi_lo_count = 0


    def _tc_to_bucket7(self, tc_int: int) -> int:
        """
        Map integer TC to 7 buckets:
        0: ≤-3, 1: -2, 2: -1, 3: 0, 4: +1, 5: +2, 6: ≥+3
        """
        if tc_int <= -3: return 0
        if tc_int == -2: return 1
        if tc_int == -1: return 2
        if tc_int == 0:  return 3
        if tc_int == 1:  return 4
        if tc_int == 2:  return 5
        return 6  # tc_int >= +3

    def _true_count_bucket(self):
        decks_left = max(self.deck.size() / 52.0, 1e-6)
        tc = self.hi_lo_count / decks_left
        tc_trunc = int(np.trunc(tc))
        return self._tc_to_bucket7(tc_trunc)

    def update_hi_lo_count(self, card_drawn):
        rank = _rank(card_drawn)
        if rank in ['2', '3', '4', '5', '6']:
            self.hi_lo_count += 1
        elif rank in ['10', 'J', 'Q', 'K', 'A']:
            self.hi_lo_count -= 1
        else:
            pass # neutral

    def get_valid_actions_idxs(self):
        return range(self.n_bets) if self.phase == 0 else (0, 1) 

    def step(self, action):
        assert self.action_space.contains(action)
        # ----- Phase 0: betting phase -----
        if self.phase == 0:
            idx = int(action) % self.n_bets
            self.current_bet = float(self.bet_multipliers[idx])

            # Deal initial hand and update counts for visible cards
            self.player = self.deck.draw_hand()
            self.dealer = self.deck.draw_hand()
            self.update_hi_lo_count(self.player[0])
            self.update_hi_lo_count(self.player[1])
            self.update_hi_lo_count(self.dealer[0])

            self.phase = 1
            assert self.observation_space.contains(self._get_obs())

            return self._get_obs(), 0.0, False, False, {}
        
        # ----- Phase 1: playing phase -----
        a = int(action) % 2
        if a == 1: # add a card to player's hand and return
            drawn_card = self.deck.draw_card()
            self.update_hi_lo_count(drawn_card)
            self.player.append(drawn_card)
            if is_bust(self.player):
                terminated = True
                reward = -1.0 * self.current_bet
            else:
                terminated = False
                reward = 0.0
        else: # stick: play out the dealer's hand, and score
            terminated = True
            self.update_hi_lo_count(self.dealer[1]) # update hi-lo count for dealer's 2nd (unseen) card
            while sum_hand(self.dealer) < 17:
                drawn_card = self.deck.draw_card()
                self.update_hi_lo_count(drawn_card)
                self.dealer.append(drawn_card)
            reward = cmp(score(self.player), score(self.dealer))
            if self.sab and is_natural(self.player) and not is_natural(self.dealer):
                # Player automatically wins. Rules consistent with S&B
                reward = 1.0
            elif (
                not self.sab
                and self.natural
                and is_natural(self.player)
                and reward == 1.0
            ):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5
            reward *= self.current_bet
            
        assert self.observation_space.contains(self._get_obs())

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), reward, terminated, False, {}
    
    def _dealer_up_value(self):
        r = _rank(self.dealer[0])
        if r == 'A': return 1
        if r in TEN_VALUE_RANKS: return 10
        return int(r)
    
    def _get_obs(self):
        if self.phase == 0:
            return (0, 0, 0, self._true_count_bucket(), 0)
        player_sum, player_usable_ace = _hand_sum_and_usable_ace(self.player)
        return (player_sum, self._dealer_up_value(), player_usable_ace, self._true_count_bucket(), 1)
    
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        cut_frac = 0.2  # reshuffle when 20% of deck remains
        # Reshuffle when low on cards
        if self.deck.size() < int(52 * self.num_decks * cut_frac):
            self.deck = Deck(self.num_decks)
            self.deck.shuffle()
            self.hi_lo_count = 0  # reset count

        # Betting phase first: no cards dealt here
        self.phase = 0
        self.current_bet = 1.0
        assert self.observation_space.contains(self._get_obs())
        return self._get_obs(), {}
    

# Pixel art from Mariia Khmelnytska (https://www.123rf.com/photo_104453049_stock-vector-pixel-art-playing-cards-standart-deck-vector-set.html)
# Environment modified from gymnasium: (https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/blackjack.py)
    