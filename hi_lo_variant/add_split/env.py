import os, math
import gymnasium as gym
from enum import IntEnum
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import numpy as np

# Support both running as a script from this folder and importing as a package
try:
    from deck import Deck
    from player import SimplePlayer
    from hand import Hand, _rank, _hard_value, ACE_RANKS, TEN_VALUE_RANKS
except ImportError:  # pragma: no cover - fallback for package import
    from .deck import Deck
    from .player import SimplePlayer
    from .hand import Hand, _rank, _hard_value, ACE_RANKS, TEN_VALUE_RANKS

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
    """Blackjack environment with Hi-Lo count, betting phase, and support for
    split & double-down actions (up to 4 concurrent hands after successive splits).

    This is an extension of the classic Sutton & Barto blackjack to include:
    - Dynamic true count buckets (Hi-Lo system)
    - Explicit betting phase with selectable bet multipliers
    - Late surrender
    - Double down
        - Splitting (including 10-value splits; at most one Ace split, max 4 hands total). After splitting Aces,
            HIT is disallowed on the resulting hands and DOUBLE/SURRENDER are also disabled for those hands.

    ## Phases
    Two distinct phases are modelled to simplify learning different decisions:
    1. Betting phase (phase = 0): agent selects a bet index.
    2. Playing phase (phase = 1): agent plays one or more hands sequentially.

    ## Action Space Encoding
    A single `Discrete(n)` where the integer meaning depends on phase:
    - Phase 0 (betting): indices 0 .. len(bet_multipliers)-1 choose a bet.
    - Phase 1 (playing):
        - 0: STICK (stand current hand; if more hands remain, advance)
        - 1: HIT (draw one card for current hand)
        - 2: SURRENDER (late surrender; only first decision with exactly 2 cards on a single initial hand)
        - 3: DOUBLE (double bet for current hand; only first decision with exactly 2 cards)
        - 4: SPLIT (only if current hand can split: two cards of same rank or both 10-value; only before first action;
                   limited to max 4 hands; only one Ace split permitted)

    Use `get_valid_actions_idxs()` to obtain the currently allowed subset.

    ## Observation Space (7-tuple)
    `(player_sum, dealer_upcard_value, usable_ace, true_count_bucket, phase, active_hand_index, total_hands)`
    - `player_sum`: best total (with usable Ace counted as 11) for the CURRENT active hand.
    - `dealer_upcard_value`: value of dealer's visible card (Ace reported as 1, 10-value cards as 10).
    - `usable_ace`: 1 if current hand has an Ace usable as 11, else 0.
    - `true_count_bucket`: index into configured Hi-Lo buckets from `tc_min..tc_max`.
    - `phase`: 0 (betting) or 1 (playing).
    - `active_hand_index`: which hand (0-based) is currently being acted upon.
    - `total_hands`: total number of player hands this round (after any splits).
    During phase 0 (before initial deal) the observation is `(0,0,0,tc_bucket,0,0,0)`.

    ## Rewards
    Per-step rewards are incremental, not cumulative: bust penalties are given immediately; final resolution
    adds only the remaining unresolved hand outcomes. Thus the sum of step rewards over an episode equals the
    net outcome across all hands.
    - Win (non-natural): +1 × hand bet
    - Loss (non-bust resolved at dealer play): −1 × hand bet
    - Bust: −1 × hand bet (reward given instantly on the HIT / DOUBLE step that causes bust)
    - Draw (push): 0
    - Natural win (only when exactly one initial hand and player natural vs dealer non-natural):
        - +1.5 × bet if `natural=True` and not `sab`
        - +1.0 × bet otherwise
    - Late surrender (first decision only, two-card single hand):
        - Dealer natural: −1 × bet
        - Dealer non-natural: −0.5 × bet
    - Double down: hand bet is doubled then one card drawn; outcome (win/lose/draw/bust) scored using doubled bet.
    - Split: no immediate reward; creates a new hand with duplicated bet.

    ## Episode Termination Conditions
    Episode ends when:
    1. Betting phase completes and all hands (including split hands) are resolved (stood or bust) and dealer play (if needed) finished.
    2. Player late surrenders (immediate termination).
    3. Player obtains a qualifying natural blackjack on the initial (unsplit) hand (immediate resolution).
    4. Final hand busts (and all hands are bust); dealer hole card is not revealed for counting in that case.

    ## Splitting Rules Implemented
    - Allowed only before any action is taken on the current hand.
    - Hand must consist of exactly two cards: either same rank or both 10-value ranks.
    - Ace pair may be split only once per round (producing two hands), tracked by `ace_split_done`.
    - After splitting Aces, each resulting hand is flagged as "no‑hit":
        - HIT is not a valid action on those hands.
        - Because first action is marked done, DOUBLE and SURRENDER are also not valid on those hands.
        - STICK remains available to advance to the next hand.
    - Each new hand receives one draw card immediately after the split.
    - Maximum total hands: 4.

    ## Notes
    - True count bucket is computed from the Hi-Lo running count divided by decks remaining (truncated then clamped).
    - Rewards returned at the final step do NOT repeat previously emitted bust penalties.
    - The overlapping integer encoding of actions between phases is intentional; the environment determines legal actions based on `phase`.
    - Training convenience: use `cumulative_episode_reward()` to read back the sum of incremental rewards for the episode.
    """
    def __init__(self, natural = False, sab = False, num_decks: int = 1, tc_min: int = -10, tc_max: int = 10, cut_frac: float = 0.25, full_info: bool = False):
        self.num_decks = num_decks
        # Dynamic true-count bucket configuration: integer buckets from tc_min..tc_max (inclusive)
        self.tc_min, self.tc_max = int(tc_min), int(tc_max)
        assert self.tc_max >= self.tc_min, "tc_max must be >= tc_min"
        # Pretty labels
        self.tc_bucket_names = tuple(f"{v:+d}" for v in range(self.tc_min, self.tc_max + 1))
        self.cut_frac = cut_frac # reshuffle when {self.cut_frac} deck remains
        self.full_info = full_info
        self.bet_multipliers = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        self.n_bets = len(self.bet_multipliers)
        # To cover all actions and all bet multipliers
        self.action_space = spaces.Discrete(max(len(PlayActions), self.n_bets))
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),                    # player_sum
            spaces.Discrete(11),                    # dealer_upcard
            spaces.Discrete(2),                     # usable_ace
            spaces.Discrete(len(self.tc_bucket_names)),   # true-count bucket
            spaces.Discrete(2),                     # phase: 0=betting, 1=playing
            spaces.Discrete(4),                     # current hand idx
            spaces.Discrete(5),                     # total number of hands [0..4]
        ))

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural

        # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
        self.sab = sab

        self.deck = Deck(self.num_decks)
        self.deck.shuffle()
        self.hi_lo_count = 0
        self.player = SimplePlayer()

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
        actions = [PlayActions.STICK, PlayActions.HIT]
        # If the active hand is a split-Ace-created hand, disallow HIT
        if getattr(self.player.active_hand, "no_hit", False):
            # remove HIT if present
            if PlayActions.HIT in actions:
                actions.remove(PlayActions.HIT)
        if len(self.player.hands) == 1 and len(self.player.active_hand) == 2:
            # First decision with exactly 2 cards: allow late surrender
            actions += [PlayActions.SURRENDER]
        if self.player.is_first_action() and len(self.player.active_hand) == 2:
            # First decision with exactly 2 cards: allow double down
            actions += [PlayActions.DOUBLE]
        if self.player.can_split():
            actions += [PlayActions.SPLIT]
        return actions
    
    def _dealer_should_hit(self):
        total, usable = self.dealer.total_and_usable_ace()
        return total < 17 or (total == 17 and usable == 1)

    def step(self, action):
        assert self.action_space.contains(action)
        # ----- Phase 0: betting phase -----
        if self.phase == 0:
            # Only allow explicit bet indices in [0, n_bets)
            assert action in self.get_valid_actions_idxs()
            idx = int(action)
            self.current_bet = float(self.bet_multipliers[idx])

            # Deal initial hand and update counts for visible cards
            player_initial_hand = self.deck.draw_hand()
            dealer_initial_hand = self.deck.draw_hand()
            self.player.start_new_round(player_initial_hand, self.current_bet)
            self.dealer = dealer_initial_hand
            self.update_hi_lo_count(self.player.active_hand[0])
            self.update_hi_lo_count(self.player.active_hand[1])
            self.update_hi_lo_count(self.dealer[0])

            self.phase = 1
            assert self.observation_space.contains(self._get_obs())

            return self._get_obs(), 0.0, False, False, {}
        
        # ----- Phase 1: playing phase -----
        # Auto win on first hand if natural hand
        if len(self.player.hands) == 1 and self.player.active_hand.is_natural():
            self.update_hi_lo_count(self.dealer[1]) # update hi-lo count for dealer's 2nd (unseen) card
            terminated = True
            if self.dealer.is_natural(): # push since both have naturals
                reward = 0.0
            else:
                reward = 1.5 if self.natural else 1.0
            reward *= self.player.active_bet
            assert self.observation_space.contains(self._get_obs())
            return self._get_obs(), reward, True, False, {}

        assert action in self.get_valid_actions_idxs()
        if action == PlayActions.HIT:  # Hit: draw card
            drawn_card = self.deck.draw_card()
            self.update_hi_lo_count(drawn_card)
            self.player.active_hand.append(drawn_card)
            if self.player.active_hand.is_bust():
                if self.player.advance_to_next_hand():
                    # More hands to play
                    terminated = False
                else:
                    # All hands played
                    terminated = True
                reward = -1.0 * self.player.active_bet
                self.player.record_reward(reward) # Accumulate reward to calculate reward for splitting
            else:
                terminated = False
                reward = 0.0
        elif action == PlayActions.STICK:  # Stick: move to next hand or dealer play
            reward = 0.0
            if self.player.advance_to_next_hand():
                # More hands to play
                terminated = False
            else:
                # All hands played
                terminated = True
        elif action == PlayActions.SURRENDER:  # Late surrender
            terminated = True
            # if dealer has blackjack: lose full bet
            if self.dealer.is_natural():
                self.update_hi_lo_count(self.dealer[1]) # update hi-lo count for dealer's 2nd (unseen) card
                reward = -1.0 * self.player.active_bet
            else: # dealer does not reveal his unseen card
                if self.full_info:
                    self.update_hi_lo_count(self.dealer[1]) # update hi-lo count for dealer's 2nd (unseen) card
                reward = -0.5 * self.player.active_bet
        elif action == PlayActions.DOUBLE: # Double down (assume only 1 player and 1 dealer)
            self.player.apply_double_down() # doubles the bet for the active hand
            player_drawn_card = self.deck.draw_card()
            self.update_hi_lo_count(player_drawn_card)
            self.player.active_hand.append(player_drawn_card)
            if self.player.active_hand.is_bust():
                reward = -1.0 * self.player.active_bet
                self.player.record_reward(reward) # Accumulate reward to calculate reward for splitting
            else:
                reward = 0.0 # will be resolved after dealer play
            if self.player.advance_to_next_hand():
                # More hands to play
                terminated = False
            else:
                # All hands played; resolve dealer hand
                terminated = True
            # terminated = True
            # player_drawn_card = self.deck.draw_card()
            # self.update_hi_lo_count(player_drawn_card)
            # # Player must receive exactly one card and then stand
            # self.player.append(player_drawn_card)
            # if self.player.is_bust():
            #     # On player bust, dealer hole card is not revealed in typical rules
            #     reward = -1.0 * 2 * self.current_bet
            # else:
            #     # Reveal dealer hole and complete dealer play to 17+
            #     self.update_hi_lo_count(self.dealer[1])  # reveal dealer's 2nd (previously unseen) card
            #     while self.dealer.total() < 17:
            #         dealer_drawn_card = self.deck.draw_card()
            #         self.update_hi_lo_count(dealer_drawn_card)
            #         self.dealer.append(dealer_drawn_card)
            #     reward = cmp(self.player.score(), self.dealer.score()) * 2 * self.current_bet
        else: # Split
            c1, c2 = self.player.do_split(self.deck)
            self.update_hi_lo_count(c1)
            self.update_hi_lo_count(c2)
            terminated = False
            reward = 0.0
        
        # Resolve dealer's hand if terminal state and not surrender
        if terminated and action != PlayActions.SURRENDER:
            # if player busted all his hands, dealer's hole card is not revealed in typical rules
            if all(hand.is_bust() for hand in self.player.hands):
                if self.full_info:
                    self.update_hi_lo_count(self.dealer[1]) # update hi-lo count for dealer's 2nd (unseen) card
                else:
                    pass
            else:
                other_hand_rewards = 0.0
                self.update_hi_lo_count(self.dealer[1]) # update hi-lo count for dealer's 2nd (unseen) card
                while self._dealer_should_hit():
                    drawn_card = self.deck.draw_card()
                    self.update_hi_lo_count(drawn_card)
                    self.dealer.append(drawn_card)
                for i, hand in enumerate(self.player.hands):
                    if hand.is_bust():
                        continue  # already recorded loss on bust
                    hand_reward = cmp(hand.score(), self.dealer.score())
                    other_hand_rewards += hand_reward * self.player.hand_bets[i]
                self.player.record_reward(other_hand_rewards) # Accumulate reward to calculate reward for splitting
                reward += other_hand_rewards

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
            return (0, 0, 0, self._true_count_bucket(), 0, 0, 0)
        player_sum, player_usable_ace = self.player.active_hand.total_and_usable_ace()
        return (player_sum, self._dealer_up_value(), player_usable_ace, self._true_count_bucket(), 1, self.player.active_index, len(self.player.hands))
    
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        # Reshuffle when low on cards
        if self.deck.size() < int(52 * self.num_decks * self.cut_frac):
            self.deck = Deck(self.num_decks)
            self.deck.shuffle()
            self.hi_lo_count = 0  # reset count

        # Betting phase first: no cards dealt here
        self.phase = 0
        self.current_bet = 1.0
        assert self.observation_space.contains(self._get_obs())
        return self._get_obs(), {}

    def cumulative_episode_reward(self) -> float:
        """Return the cumulative reward accumulated by the player over the current episode.

        This is a convenience helper for training loops that want the net episode
        reward (sum of incremental rewards emitted throughout the episode).
        """
        return float(self.player.accumulated_reward)
    

# Pixel art from Mariia Khmelnytska (https://www.123rf.com/photo_104453049_stock-vector-pixel-art-playing-cards-standart-deck-vector-set.html)
# Environment modified from gymnasium: (https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/blackjack.py)
    