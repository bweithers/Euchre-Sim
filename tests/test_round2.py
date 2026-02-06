import random
import unittest

from euchre.cards import Card, Deck, SUITS
from euchre.game import Player, EuchreGame
from euchre.strategy import (
    NeuralStrategy, AlwaysPassStrategy, RandomCallStrategy, CallingStrategy,
    EvolutionEngine, encode_hand_relative,
)


class TestRound2Strategy(unittest.TestCase):
    """Test choose_trump_round2 on all strategy classes."""

    def test_neural_returns_suit_or_none(self):
        random.seed(0)
        strategy = NeuralStrategy()
        deck = Deck()
        deck.shuffle()
        hand = deck.deal(5)
        result = strategy.choose_trump_round2(hand, 'Hearts', 1)
        self.assertTrue(result is None or result in SUITS)

    def test_neural_never_returns_turned_suit(self):
        random.seed(1)
        strategy = NeuralStrategy()
        for i in range(100):
            random.seed(i)
            deck = Deck()
            deck.shuffle()
            hand = deck.deal(5)
            for turned_suit in SUITS:
                result = strategy.choose_trump_round2(hand, turned_suit, 2)
                if result is not None:
                    self.assertNotEqual(result, turned_suit)

    def test_always_pass_returns_none(self):
        strategy = AlwaysPassStrategy()
        result = strategy.choose_trump_round2([], 'Spades', 1)
        self.assertIsNone(result)

    def test_random_call_returns_valid_suit(self):
        random.seed(42)
        strategy = RandomCallStrategy(call_rate=1.0)
        deck = Deck()
        deck.shuffle()
        hand = deck.deal(5)
        result = strategy.choose_trump_round2(hand, 'Diamonds', 3)
        self.assertIn(result, [s for s in SUITS if s != 'Diamonds'])

    def test_random_call_zero_rate_returns_none(self):
        strategy = RandomCallStrategy(call_rate=0.0)
        deck = Deck()
        deck.shuffle()
        hand = deck.deal(5)
        result = strategy.choose_trump_round2(hand, 'Clubs', 1)
        self.assertIsNone(result)

    def test_calling_strategy_returns_none(self):
        strategy = CallingStrategy()
        result = strategy.choose_trump_round2([], 'Hearts', 1)
        self.assertIsNone(result)


class TestRound2Encoding(unittest.TestCase):
    """Test that round 2 encoding zeroes out turned card inputs."""

    def test_turned_card_none_zeroes_inputs_17_18(self):
        hand = [Card('Hearts', 'Jack'), Card('Hearts', 'Ace'),
                Card('Clubs', '9'), Card('Spades', 'King'), Card('Diamonds', '10')]
        inputs = encode_hand_relative(hand, 'Hearts', 1, turned_card=None)
        self.assertEqual(inputs[17], 0.0)
        self.assertEqual(inputs[18], 0.0)

    def test_turned_card_present_sets_inputs(self):
        hand = [Card('Hearts', '9'), Card('Clubs', '10'),
                Card('Spades', 'Queen'), Card('Diamonds', 'King'), Card('Diamonds', 'Ace')]
        turned = Card('Hearts', 'Jack')
        inputs = encode_hand_relative(hand, 'Hearts', 1, turned_card=turned)
        self.assertEqual(inputs[17], 1.0)  # turned is bower


class TestBiddingRound2(unittest.TestCase):
    """Test the game engine's round 2 bidding flow."""

    def _make_game(self, strategy):
        players = [
            Player('A1', 'Team A'),
            Player('B1', 'Team B'),
            Player('A2', 'Team A'),
            Player('B2', 'Team B'),
        ]
        for p in players:
            p.calling_strategy = strategy
        return EuchreGame(players=players, silent=True)

    def test_all_pass_both_rounds_scores_zero(self):
        game = self._make_game(AlwaysPassStrategy())
        game.dealer_index = 0
        scores = game.play_hand()
        self.assertEqual(scores, {"Team A": 0, "Team B": 0})

    def test_round2_sets_non_turned_trump(self):
        game = self._make_game(AlwaysPassStrategy())
        game.dealer_index = 0
        game.deal_cards()
        turned_suit = game.turned_card.suit

        # Force round 1 to fail
        self.assertFalse(game.bidding_round_1())

        # Manually give a player a strategy that calls round 2
        game.players[1].calling_strategy = RandomCallStrategy(call_rate=1.0)
        self.assertTrue(game.bidding_round_2())
        self.assertIsNotNone(game.trump_suit)
        self.assertNotEqual(game.trump_suit, turned_suit)

    def test_play_hand_with_round2(self):
        """Round 2 fires when round 1 passes and a R2 strategy is present."""
        random.seed(99)
        # Use a strategy that always passes R1 but always calls R2
        game = self._make_game(AlwaysPassStrategy())
        for p in game.players:
            p.calling_strategy = RandomCallStrategy(call_rate=0.0)

        # Override one player to always call in round 2
        class R2OnlyStrategy:
            def should_call(self, hand, turned_card, game, seat_offset, is_dealer):
                return False
            def choose_trump_round2(self, hand, turned_card_suit, seat_offset):
                candidates = [s for s in SUITS if s != turned_card_suit]
                return candidates[0]

        game.players[1].calling_strategy = R2OnlyStrategy()
        game.dealer_index = 0
        scores = game.play_hand()
        # Should have played a hand (non-zero scores) since R2 always calls
        self.assertTrue(scores["Team A"] > 0 or scores["Team B"] > 0)

    def test_hasattr_guard_no_round2_method(self):
        """Player with a strategy lacking choose_trump_round2 returns None."""
        class LegacyStrategy:
            def should_call(self, hand, turned_card, game, seat_offset, is_dealer):
                return False

        player = Player('Test', 'Team A')
        player.calling_strategy = LegacyStrategy()
        result = player.decide_call_round2('Hearts', 1)
        self.assertIsNone(result)


class TestEvolutionRound2(unittest.TestCase):
    """Test evolution engine with round 2 call rate."""

    def test_run_generation_includes_r2_rate(self):
        random.seed(7)
        engine = EvolutionEngine(population_size=4, hands_per_matchup=10)
        result = engine.run_generation()
        self.assertIn('call_rate', result)
        self.assertIn('call_rate_r2', result)
        self.assertIsInstance(result['call_rate'], float)
        self.assertIsInstance(result['call_rate_r2'], float)
        self.assertGreaterEqual(result['call_rate'], 0.0)
        self.assertGreaterEqual(result['call_rate_r2'], 0.0)
        self.assertLessEqual(result['call_rate'], 1.0)
        self.assertLessEqual(result['call_rate_r2'], 1.0)


if __name__ == '__main__':
    unittest.main()
