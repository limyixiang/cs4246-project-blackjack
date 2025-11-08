import os, math
import gymnasium as gym
from enum import IntEnum
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import numpy as np

# Support both running as a script from this folder and importing as a package
try:
    from deck import Deck
except ImportError:  # pragma: no cover - fallback for package import
    from .deck import Deck

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

def cmp(a, b):
    """Returns +1 if a > b, 0 if a = b, -1 if a < b"""
    return float(a > b) - float(a < b)

class PlayActions(IntEnum):
    STICK = 0
    HIT = 1
    SURRENDER = 2
    DOUBLE = 3
    SPLIT = 4

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
    Two phases with different valid action sets:
    - Phase 0 (betting): choose a bet index in `bet_multipliers`.
    - Phase 1 (playing):
        - 0: Stick (stand)
        - 1: Hit (draw a card)
        - 2: Surrender (late surrender; only allowed as the first decision with exactly 2 cards)
        - 3: Double down (only allowed as the first decision with exactly 2 cards; draw exactly one card then stand)

    Notes:
    - Late surrender implementation: when surrendering the dealer's hole card is revealed. If dealer has a natural blackjack the player loses the full bet; otherwise loses half the bet.
    - `get_valid_actions_idxs()` returns the context‑appropriate subset.

    ## Observation Space
    The observation is a 5‑tuple:
    `(player_sum, dealer_upcard_value, usable_ace, true_count_bucket, phase)`
    where:
    - `phase` ∈ {0 (betting), 1 (playing)}.
    - In Phase 0 before cards are dealt: `(0, 0, 0, tc_bucket, 0)`.

    ## Starting State
    The starting state is initialised with the following values.

    | Observation               | Values                        |
    |---------------------------|-------------------------------|
    | Player current sum        |  4, 5, ..., 21                |
    | Dealer showing card value |  1, 2, ..., 10                |
    | Usable Ace                |  0, 1                         |
    | True Count                |  0, ..., tc_max - tc_min      |

    ## Rewards (scaled by selected bet multiplier)
    - Win game: +1 × bet
    - Lose game: −1 × bet
    - Draw game: 0
    - Natural win (player has blackjack, dealer does not):
        - +1.5 × bet (if `natural=True` and `sab=False`)
        - +1 × bet (otherwise / `sab=True`)
    - Late surrender (first decision with exactly two cards, dealer not natural): −0.5 × bet
    - Late surrender when dealer has a natural blackjack: −1 × bet
    - Double down: outcome is resolved immediately after drawing one card; final result is multiplied by 2 × bet

    ## Episode End
    Termination occurs when any of the following:
    1. Player busts after a hit.
    2. Player sticks and dealer finishes play.
    3. Player (late) surrenders.
    4. Player natural (auto resolution).
    """
    def __init__(self, natural = False, sab = False, num_decks: int = 1, tc_min: int = -10, tc_max: int = 10):
        self.num_decks = num_decks
        # Dynamic true-count bucket configuration: integer buckets from tc_min..tc_max (inclusive)
        self.tc_min, self.tc_max = int(tc_min), int(tc_max)
        assert self.tc_max >= self.tc_min, "tc_max must be >= tc_min"
        # Pretty labels
        self.tc_bucket_names = tuple(f"{v:+d}" for v in range(self.tc_min, self.tc_max + 1))
        self.bet_multipliers = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        self.n_bets = len(self.bet_multipliers)
        # To cover all actions and all bet multipliers
        self.action_space = spaces.Discrete(max(len(PlayActions), self.n_bets))
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

    def _true_count_bucket(self):
        decks_left = max(self.deck.size() / 52.0, 1e-6)
        tc = self.hi_lo_count / decks_left
        tc_truncated = int(np.trunc(tc))
        tc_clamped = min(max(tc_truncated, self.tc_min), self.tc_max)
        return tc_clamped - self.tc_min

    def update_hi_lo_count(self, card_drawn): # common hi-low counting strategy (simplified by Harvey Dubner)
        rank = _rank(card_drawn)
        if rank in ['2', '3', '4', '5', '6']:
            self.hi_lo_count += 1
        elif rank in ['10', 'J', 'Q', 'K', 'A']:
            self.hi_lo_count -= 1
        else:
            pass # neutral

    # def update_hi_lo_count(self, card_drawn): # Thorp's strategy
    #     rank = _rank(card_drawn)
    #     if rank in ['A', '2', '3', '4', '5', '6', '7', '8', '9']:
    #         self.hi_lo_count += 4
    #     elif rank in ['10', 'J', 'Q', 'K']:
    #         self.hi_lo_count -= 9
    #     else:
    #         pass

    def get_valid_actions_idxs(self):
        # Betting phase: bet index options
        if self.phase == 0:
            return range(self.n_bets)
        # Playing phase:
        if len(self.player) == 2:
            return (PlayActions.STICK, PlayActions.HIT, PlayActions.SURRENDER, PlayActions.DOUBLE)  # allowed only as first decision
        return (PlayActions.STICK, PlayActions.HIT)  # stick, hit

    def step(self, action):
        assert self.action_space.contains(action)
        # ----- Phase 0: betting phase -----
        if self.phase == 0:
            # Only allow explicit bet indices in [0, n_bets)
            assert action in self.get_valid_actions_idxs()
            idx = int(action)
            self.current_bet = float(self.bet_multipliers[idx])

            # Deal initial hand and update counts for visible cards
            self.player = Hand(self.deck.draw_hand())
            self.dealer = Hand(self.deck.draw_hand())
            self.update_hi_lo_count(self.player[0])
            self.update_hi_lo_count(self.player[1])
            self.update_hi_lo_count(self.dealer[0])

            self.phase = 1
            assert self.observation_space.contains(self._get_obs())

            return self._get_obs(), 0.0, False, False, {}
        
        # ----- Phase 1: playing phase -----
        # Auto win if natural hand
        if len(self.player) == 2 and self.player.is_natural():
            self.update_hi_lo_count(self.dealer[1]) # update hi-lo count for dealer's 2nd (unseen) card
            terminated = True
            if self.dealer.is_natural():
                reward = 0.0
            else:
                reward = 1.5 if self.natural else 1.0
            reward *= self.current_bet
            assert self.observation_space.contains(self._get_obs())
            return self._get_obs(), reward, True, False, {}

        assert action in self.get_valid_actions_idxs()
        if action == PlayActions.HIT:  # Hit: draw card
            drawn_card = self.deck.draw_card()
            self.update_hi_lo_count(drawn_card)
            self.player.append(drawn_card)
            if self.player.is_bust():
                terminated = True
                reward = -1.0 * self.current_bet
            else:
                terminated = False
                reward = 0.0
        elif action == PlayActions.STICK:  # Stick: resolve dealer hand
            terminated = True
            self.update_hi_lo_count(self.dealer[1]) # update hi-lo count for dealer's 2nd (unseen) card
            while self.dealer.total() < 17:
                drawn_card = self.deck.draw_card()
                self.update_hi_lo_count(drawn_card)
                self.dealer.append(drawn_card)
            reward = cmp(self.player.score(), self.dealer.score())
            if self.sab and self.player.is_natural() and not self.dealer.is_natural():
                # Player automatically wins. Rules consistent with S&B
                reward = 1.0
            elif (
                not self.sab
                and self.natural
                and self.player.is_natural()
                and reward == 1.0
            ):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5
            reward *= self.current_bet
        elif action == PlayActions.SURRENDER:  # Late surrender
            terminated = True
            self.update_hi_lo_count(self.dealer[1]) # update hi-lo count for dealer's 2nd (unseen) card
            # if dealer has blackjack: lose full bet
            if self.dealer.is_natural():
                reward = -1.0 * self.current_bet
            else:
                reward = -0.5 * self.current_bet
        else: # Double down (assume only 1 player and 1 dealer)
            terminated = True
            player_drawn_card = self.deck.draw_card()
            self.update_hi_lo_count(player_drawn_card)
            # Player must receive exactly one card and then stand
            self.player.append(player_drawn_card)
            if self.player.is_bust():
                # On player bust, dealer hole card is not revealed in typical rules
                reward = -1.0 * 2 * self.current_bet
            else:
                # Reveal dealer hole and complete dealer play to 17+
                self.update_hi_lo_count(self.dealer[1])  # reveal dealer's 2nd (previously unseen) card
                while self.dealer.total() < 17:
                    dealer_drawn_card = self.deck.draw_card()
                    self.update_hi_lo_count(dealer_drawn_card)
                    self.dealer.append(dealer_drawn_card)
                reward = cmp(self.player.score(), self.dealer.score()) * 2 * self.current_bet

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
        player_sum, player_usable_ace = self.player.total_and_usable_ace()
        return (player_sum, self._dealer_up_value(), player_usable_ace, self._true_count_bucket(), 1)
    
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        cut_frac = 0.25  # reshuffle when 25% of deck remains
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
    