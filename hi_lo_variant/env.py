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
    return (ranks & ACE_RANKS) and (ranks & TEN_VALUE_RANKS)

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
    while the player has two face up cards. All cards are drawn from a finite deck, consisting of a pre-determined number of decks. The deck will be reset and shuffled when the number of cards remaining drops below a certain threshold
    (i.e. with replacement).

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
    The observation consists of a 4-tuple containing: the player's current sum,
    the value of the dealer's one showing card (1-10 where 1 is ace),
    whether the player holds a usable ace (0 or 1),
    and the true count indicating the hi-lo count / number of decks remaining.

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
    def __init__(self, render_mode: str | None = None, natural = False, sab = False, num_decks: int = 1, tc_min: int = -10, tc_max: int = 10):
        self.num_decks = num_decks
        self.tc_min, self.tc_max = tc_min, tc_max
        self.action_space = spaces.Discrete(2)
        # player_sum, dealer_shown_card, usable_ace
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),                    # player_sum
            spaces.Discrete(11),                    # dealer_upcard
            spaces.Discrete(2),                     # usable_ace
            spaces.Discrete(tc_max - tc_min + 1)    # true-count bucket
        ))

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural

        # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
        self.sab = sab

        self.render_mode = render_mode

        self.deck = Deck(self.num_decks)
        self.deck.shuffle()
        self.hi_lo_count = 0

    def _true_count_bucket(self):
        # Treat less than one deck left as 1.0 deck to avoid extreme spikes in tc.
        decks_left = self.deck.size() / 52.0
        if decks_left < 1.0:
            decks_left = 1.0
        tc = self.hi_lo_count / decks_left
        tc_rounded = int(np.round(tc))
        tc_clamped = int(np.clip(tc_rounded, self.tc_min, self.tc_max))
        return tc_clamped - self.tc_min # map to [0..range]

    def update_hi_lo_count(self, card_drawn):
        rank = _rank(card_drawn)
        if rank in ['2', '3', '4', '5', '6']:
            self.hi_lo_count -= 1
        elif rank in ['10', 'J', 'Q', 'K', 'A']:
            self.hi_lo_count += 1
        else:
            pass # neutral

    def step(self, action):
        assert self.action_space.contains(action)
        if action: # add a card to player's hand and return
            drawn_card = self.deck.draw_card()
            self.update_hi_lo_count(drawn_card)
            self.player.append(drawn_card)
            if is_bust(self.player):
                terminated = True
                reward = -1.0
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
            
        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), reward, terminated, False, {}
    
    def _dealer_up_value(self):
        r = _rank(self.dealer[0])
        if r == 'A': return 1
        if r in TEN_VALUE_RANKS: return 10
        return int(r)
    
    def _get_obs(self):
        # Optimization: Compute sum and usable ace in one pass per step, not two.
        player_sum, player_usable_ace = _hand_sum_and_usable_ace(self.player)
        return (player_sum, self._dealer_up_value(), player_usable_ace, self._true_count_bucket())
    
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

        self.dealer = self.deck.draw_hand()
        self.update_hi_lo_count(self.dealer[0])
        self.player = self.deck.draw_hand()
        self.update_hi_lo_count(self.player[0])
        self.update_hi_lo_count(self.player[1])

        dealer_top_card = self.dealer[0]

        # suits = ["C", "D", "H", "S"]
        self.dealer_top_card_suit = dealer_top_card[-1]
        self.dealer_top_card_value_str = _rank(dealer_top_card)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}
    
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e

        player_sum, dealer_card_value, usable_ace, _ = self._get_obs()
        screen_width, screen_height = 600, 500
        card_img_height = screen_height // 3
        card_img_width = int(card_img_height * 142 / 197)
        spacing = screen_height // 20

        bg_color = (7, 99, 36)
        white = (255, 255, 255)

        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        self.screen.fill(bg_color)

        def get_image(path):
            cwd = os.path.dirname(__file__)
            image = pygame.image.load(os.path.join(cwd, path))
            return image

        def get_font(path, size):
            cwd = os.path.dirname(__file__)
            font = pygame.font.Font(os.path.join(cwd, path), size)
            return font

        small_font = get_font(
            os.path.join("font", "Minecraft.ttf"), screen_height // 15
        )
        dealer_text = small_font.render(
            "Dealer: " + str(dealer_card_value), True, white
        )
        dealer_text_rect = self.screen.blit(dealer_text, (spacing, spacing))

        def scale_card_img(card_img):
            return pygame.transform.scale(card_img, (card_img_width, card_img_height))

        dealer_card_img = scale_card_img(
            get_image(
                os.path.join(
                    "img",
                    f"{self.dealer_top_card_suit}{self.dealer_top_card_value_str}.png",
                )
            )
        )
        dealer_card_rect = self.screen.blit(
            dealer_card_img,
            (
                screen_width // 2 - card_img_width - spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        hidden_card_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
        self.screen.blit(
            hidden_card_img,
            (
                screen_width // 2 + spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        player_text = small_font.render("Player", True, white)
        player_text_rect = self.screen.blit(
            player_text, (spacing, dealer_card_rect.bottom + 1.5 * spacing)
        )

        large_font = get_font(os.path.join("font", "Minecraft.ttf"), screen_height // 6)
        player_sum_text = large_font.render(str(player_sum), True, white)
        player_sum_text_rect = self.screen.blit(
            player_sum_text,
            (
                screen_width // 2 - player_sum_text.get_width() // 2,
                player_text_rect.bottom + spacing,
            ),
        )

        if usable_ace:
            usable_ace_text = small_font.render("usable ace", True, white)
            self.screen.blit(
                usable_ace_text,
                (
                    screen_width // 2 - usable_ace_text.get_width() // 2,
                    player_sum_text_rect.bottom + spacing // 2,
                ),
            )
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if hasattr(self, "screen"):
            import pygame

            pygame.display.quit()
            pygame.quit()


# Pixel art from Mariia Khmelnytska (https://www.123rf.com/photo_104453049_stock-vector-pixel-art-playing-cards-standart-deck-vector-set.html)
# Environment modified from gymnasium: (https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/blackjack.py)
    