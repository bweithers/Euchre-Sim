import random
import json
import math
from .cards import SAME_COLOR, SUITS, RANKS
from .game import Player, EuchreGame


def _sigmoid(x):
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _tanh(x):
    if x < -500:
        return -1.0
    if x > 500:
        return 1.0
    return math.tanh(x)


# Relative encoding - exploits trump symmetry
# Inputs:
#   0-6:   Trump cards (right bower, left bower, ace, king, queen, 10, 9)
#   7-9:   Off-suit aces (3 suits)
#   10-12: Voids in off-suits (3 suits)
#   13-16: Position (one-hot, seats 1-4)
#   17:    Turned card is right bower (dealer would get it)
#   18:    Turned card is ace or king (strong pickup for dealer)
INPUT_SIZE = 19
HIDDEN_SIZE = 12
OUTPUT_SIZE = 1

# For labeling/analysis
INPUT_NAMES = [
    "has_right_bower", "has_left_bower", "has_trump_ace", "has_trump_king",
    "has_trump_queen", "has_trump_10", "has_trump_9",
    "off_ace_1", "off_ace_2", "off_ace_3",
    "void_1", "void_2", "void_3",
    "seat_1", "seat_2", "seat_3", "seat_4",
    "turned_is_bower", "turned_is_high",
]


def encode_hand_relative(hand, trump_suit, seat_offset, turned_card=None):
    """Encode hand relative to trump suit - exploits symmetry."""
    inputs = [0.0] * INPUT_SIZE

    left_bower_suit = SAME_COLOR[trump_suit]
    off_suits = [s for s in SUITS if s != trump_suit]

    # Track what we have
    trump_ranks = []
    off_suit_cards = {s: [] for s in off_suits}

    for card in hand:
        if card.suit == trump_suit:
            trump_ranks.append(card.rank)
        elif card.rank == 'Jack' and card.suit == left_bower_suit:
            # Left bower counts as trump
            inputs[1] = 1.0  # has_left_bower
        else:
            off_suit_cards[card.suit].append(card.rank)

    # Trump cards (indices 0-6)
    if 'Jack' in trump_ranks:
        inputs[0] = 1.0  # right bower
    if 'Ace' in trump_ranks:
        inputs[2] = 1.0
    if 'King' in trump_ranks:
        inputs[3] = 1.0
    if 'Queen' in trump_ranks:
        inputs[4] = 1.0
    if '10' in trump_ranks:
        inputs[5] = 1.0
    if '9' in trump_ranks:
        inputs[6] = 1.0

    # Off-suit aces and voids (indices 7-12)
    for i, suit in enumerate(off_suits):
        if 'Ace' in off_suit_cards[suit]:
            inputs[7 + i] = 1.0  # off-suit ace
        if len(off_suit_cards[suit]) == 0:
            # Check if we're truly void (not counting left bower)
            has_left_bower_in_suit = (suit == left_bower_suit and inputs[1] == 1.0)
            if not has_left_bower_in_suit:
                inputs[10 + i] = 1.0  # void

    # Position (indices 13-16)
    inputs[13 + (seat_offset - 1)] = 1.0

    # Turned card info (indices 17-18)
    if turned_card:
        if turned_card.rank == 'Jack':
            inputs[17] = 1.0  # turned card is right bower
        elif turned_card.rank in ('Ace', 'King'):
            inputs[18] = 1.0  # turned card is high

    return inputs


class NeuralStrategy:
    """Neural network strategy for trump calling decisions.

    Architecture: 19 inputs -> 12 hidden (tanh) -> 1 output (sigmoid)

    Uses relative encoding that exploits trump symmetry:
    - Trump cards encoded as "has right bower" not "has Jack of Spades"
    - Position and turned card info included

    Weights are stored as flat list:
    - w1: INPUT_SIZE * HIDDEN_SIZE weights for input->hidden
    - b1: HIDDEN_SIZE biases for hidden layer
    - w2: HIDDEN_SIZE * OUTPUT_SIZE weights for hidden->output
    - b2: OUTPUT_SIZE biases for output layer
    """

    NUM_WEIGHTS = (INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE +
                   HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE)

    def __init__(self, weights=None):
        if weights is not None:
            self.weights = list(weights)
        else:
            # Xavier initialization
            scale1 = math.sqrt(2.0 / (INPUT_SIZE + HIDDEN_SIZE))
            scale2 = math.sqrt(2.0 / (HIDDEN_SIZE + OUTPUT_SIZE))

            self.weights = []
            # Input -> Hidden weights
            for _ in range(INPUT_SIZE * HIDDEN_SIZE):
                self.weights.append(random.gauss(0, scale1))
            # Hidden biases
            for _ in range(HIDDEN_SIZE):
                self.weights.append(0.0)
            # Hidden -> Output weights
            for _ in range(HIDDEN_SIZE * OUTPUT_SIZE):
                self.weights.append(random.gauss(0, scale2))
            # Output biases
            for _ in range(OUTPUT_SIZE):
                self.weights.append(0.0)

    def _forward(self, inputs):
        """Forward pass through the network."""
        # Unpack weights
        w1_end = INPUT_SIZE * HIDDEN_SIZE
        b1_end = w1_end + HIDDEN_SIZE
        w2_end = b1_end + HIDDEN_SIZE * OUTPUT_SIZE

        w1 = self.weights[:w1_end]
        b1 = self.weights[w1_end:b1_end]
        w2 = self.weights[b1_end:w2_end]
        b2 = self.weights[w2_end:]

        # Hidden layer
        hidden = []
        for h in range(HIDDEN_SIZE):
            total = b1[h]
            for i in range(INPUT_SIZE):
                total += inputs[i] * w1[i * HIDDEN_SIZE + h]
            hidden.append(_tanh(total))

        # Output layer
        output = b2[0]
        for h in range(HIDDEN_SIZE):
            output += hidden[h] * w2[h]

        return _sigmoid(output)

    def should_call(self, hand, turned_card, game, seat_offset, is_dealer):
        trump_suit = turned_card.suit
        inputs = encode_hand_relative(hand, trump_suit, seat_offset, turned_card)
        prob = self._forward(inputs)
        return prob > 0.5

    def get_call_probability(self, hand, turned_card, seat_offset):
        """Get the raw probability of calling (useful for analysis)."""
        trump_suit = turned_card.suit
        inputs = encode_hand_relative(hand, trump_suit, seat_offset, turned_card)
        return self._forward(inputs)

    def choose_trump_round2(self, hand, turned_card_suit, seat_offset):
        """Evaluate each non-turned suit, call the best one if prob > 0.5."""
        best_suit = None
        best_prob = 0.5
        for suit in SUITS:
            if suit == turned_card_suit:
                continue
            inputs = encode_hand_relative(hand, suit, seat_offset, turned_card=None)
            prob = self._forward(inputs)
            if prob > best_prob:
                best_prob = prob
                best_suit = suit
        return best_suit

    def copy(self):
        return NeuralStrategy(weights=list(self.weights))

    def to_dict(self):
        return {
            "type": "neural",
            "architecture": [INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE],
            "weights": [round(w, 6) for w in self.weights]
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d):
        return cls(weights=d["weights"])

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))


# Keep old linear strategy for backwards compatibility
FEATURE_NAMES = [
    "trump_count", "has_right_bower", "has_left_bower", "has_trump_ace",
    "has_trump_king", "off_suit_aces", "void_count", "seat_offset",
    "is_dealer", "dealer_extra_trump", "bias",
]

NUM_FEATURES = len(FEATURE_NAMES)


def extract_features(hand, turned_card, trump_suit, seat_offset, is_dealer):
    """Extract features for linear model (legacy)."""
    left_bower_suit = SAME_COLOR[trump_suit]
    trump_count = 0
    has_right_bower = 0
    has_left_bower = 0
    has_trump_ace = 0
    has_trump_king = 0
    off_suit_aces = 0
    suits_in_hand = set()

    for card in hand:
        if card.suit == trump_suit:
            trump_count += 1
            if card.rank == 'Jack':
                has_right_bower = 1
            elif card.rank == 'Ace':
                has_trump_ace = 1
            elif card.rank == 'King':
                has_trump_king = 1
            suits_in_hand.add(trump_suit)
        elif card.rank == 'Jack' and card.suit == left_bower_suit:
            trump_count += 1
            has_left_bower = 1
            suits_in_hand.add(trump_suit)
        else:
            if card.rank == 'Ace':
                off_suit_aces += 1
            suits_in_hand.add(card.suit)

    dealer_extra_trump_val = 0
    if is_dealer:
        if turned_card.suit == trump_suit or (turned_card.rank == 'Jack' and turned_card.suit == left_bower_suit):
            dealer_extra_trump_val = 1

    off_suits = [s for s in SUITS if s != trump_suit]
    void_count = sum(1 for s in off_suits if s not in suits_in_hand)

    return [
        trump_count / 5.0, has_right_bower, has_left_bower, has_trump_ace,
        has_trump_king, off_suit_aces / 3.0, void_count / 3.0, seat_offset / 4.0,
        float(is_dealer), float(dealer_extra_trump_val), 1.0,
    ]


class CallingStrategy:
    """Linear strategy (legacy, kept for compatibility)."""

    def __init__(self, weights=None):
        if weights is not None:
            self.weights = list(weights)
        else:
            self.weights = [random.gauss(0, 1) for _ in range(NUM_FEATURES)]

    def should_call(self, hand, turned_card, game, seat_offset, is_dealer):
        trump_suit = turned_card.suit
        features = extract_features(hand, turned_card, trump_suit, seat_offset, is_dealer)
        score = sum(w * f for w, f in zip(self.weights, features))
        return score > 0

    def choose_trump_round2(self, hand, turned_card_suit, seat_offset):
        return None

    def copy(self):
        return CallingStrategy(weights=list(self.weights))

    def to_dict(self):
        return {name: round(w, 6) for name, w in zip(FEATURE_NAMES, self.weights)}

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d):
        weights = [d[name] for name in FEATURE_NAMES]
        return cls(weights=weights)

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))


class AlwaysPassStrategy:
    def should_call(self, hand, turned_card, game, seat_offset, is_dealer):
        return False

    def choose_trump_round2(self, hand, turned_card_suit, seat_offset):
        return None


class RandomCallStrategy:
    def __init__(self, call_rate=0.5):
        self.call_rate = call_rate

    def should_call(self, hand, turned_card, game, seat_offset, is_dealer):
        return random.random() < self.call_rate

    def choose_trump_round2(self, hand, turned_card_suit, seat_offset):
        if random.random() < self.call_rate:
            candidates = [s for s in SUITS if s != turned_card_suit]
            return random.choice(candidates)
        return None


def _make_players(strategy_a, strategy_b):
    players = [
        Player("A1", "Team A"),
        Player("B1", "Team B"),
        Player("A2", "Team A"),
        Player("B2", "Team B"),
    ]
    players[0].calling_strategy = strategy_a
    players[1].calling_strategy = strategy_b
    players[2].calling_strategy = strategy_a
    players[3].calling_strategy = strategy_b
    return players


def evaluate_fitness(strategy_a, strategy_b, num_hands=200):
    players = _make_players(strategy_a, strategy_b)
    game = EuchreGame(players=players, silent=True)

    total_a = 0
    total_b = 0
    for hand_num in range(num_hands):
        game.dealer_index = hand_num % 4
        scores = game.play_hand()
        total_a += scores["Team A"]
        total_b += scores["Team B"]

    return total_a, total_b


def _tournament_select(population, fitnesses, k=3):
    candidates = random.sample(list(zip(population, fitnesses)), k)
    return max(candidates, key=lambda x: x[1])[0]


def _crossover_neural(parent_a, parent_b):
    """Uniform crossover for neural network weights."""
    child_weights = []
    for wa, wb in zip(parent_a.weights, parent_b.weights):
        child_weights.append(wa if random.random() < 0.5 else wb)
    return NeuralStrategy(weights=child_weights)


def _mutate_neural(strategy, mutation_rate=0.1, mutation_strength=0.3):
    """Mutate neural network weights."""
    for i in range(len(strategy.weights)):
        if random.random() < mutation_rate:
            strategy.weights[i] += random.gauss(0, mutation_strength)


class EvolutionEngine:
    """Evolution engine for neural network strategies."""

    def __init__(self, population_size=50, hands_per_matchup=200,
                 mutation_rate=0.1, mutation_strength=0.3, elite_count=5,
                 strategy_class=NeuralStrategy):
        self.population_size = population_size
        self.hands_per_matchup = hands_per_matchup
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_count = elite_count
        self.strategy_class = strategy_class
        self.population = [strategy_class() for _ in range(population_size)]
        self.generation = 0

    def _evaluate_population(self):
        n = len(self.population)
        fitnesses = [0.0] * n

        for i in range(n):
            for j in range(i + 1, n):
                score_i, score_j = evaluate_fitness(
                    self.population[i], self.population[j],
                    num_hands=self.hands_per_matchup
                )
                fitnesses[i] += score_i - score_j
                fitnesses[j] += score_j - score_i

        return fitnesses

    def _compute_call_rate(self, strategy, num_samples=500):
        """Estimate how often a strategy calls trump in round 1 and round 2."""
        from .cards import Deck
        r1_calls = 0
        r2_calls = 0
        for _ in range(num_samples):
            deck = Deck()
            deck.shuffle()
            hand = deck.deal(5)
            turned = deck.deal(1)[0]
            seat_offset = random.choice([1, 2, 3, 4])
            is_dealer = (seat_offset == 4)
            if strategy.should_call(hand, turned, None, seat_offset, is_dealer):
                r1_calls += 1
            elif hasattr(strategy, 'choose_trump_round2'):
                result = strategy.choose_trump_round2(hand, turned.suit, seat_offset)
                if result is not None:
                    r2_calls += 1
        return r1_calls / num_samples, r2_calls / num_samples

    def run_generation(self):
        self.generation += 1
        fitnesses = self._evaluate_population()

        paired = list(zip(self.population, fitnesses))
        paired.sort(key=lambda x: x[1], reverse=True)

        best_fitness = paired[0][1]
        median_fitness = paired[len(paired) // 2][1]
        best_strategy = paired[0][0]
        call_rate, call_rate_r2 = self._compute_call_rate(best_strategy)

        # Elitism
        new_population = [s.copy() for s, _ in paired[:self.elite_count]]

        # Fill with offspring
        while len(new_population) < self.population_size:
            parent_a = _tournament_select(self.population, fitnesses, k=3)
            parent_b = _tournament_select(self.population, fitnesses, k=3)

            if self.strategy_class == NeuralStrategy:
                child = _crossover_neural(parent_a, parent_b)
                _mutate_neural(child, self.mutation_rate, self.mutation_strength)
            else:
                # Legacy linear crossover
                child_weights = [wa if random.random() < 0.5 else wb
                                for wa, wb in zip(parent_a.weights, parent_b.weights)]
                child = CallingStrategy(weights=child_weights)
                for i in range(len(child.weights)):
                    if random.random() < self.mutation_rate:
                        child.weights[i] += random.gauss(0, self.mutation_strength)

            new_population.append(child)

        self.population = new_population

        return {
            "generation": self.generation,
            "best_fitness": best_fitness,
            "median_fitness": median_fitness,
            "call_rate": call_rate,
            "call_rate_r2": call_rate_r2,
            "best_strategy": best_strategy,
        }

    def best(self):
        fitnesses = self._evaluate_population()
        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        return self.population[best_idx]
