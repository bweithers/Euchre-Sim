import argparse
import json
import random
import time

from euchre.strategy import (
    EvolutionEngine, NeuralStrategy, CallingStrategy,
    AlwaysPassStrategy, RandomCallStrategy, evaluate_fitness,
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, INPUT_NAMES,
    encode_hand_relative,
)


def print_neural_info(strategy):
    """Print summary info about neural network strategy."""
    print(f"\n  Architecture: {INPUT_SIZE} -> {HIDDEN_SIZE} -> {OUTPUT_SIZE}")
    print(f"  Total weights: {len(strategy.weights)}")

    # Compute some basic stats
    weights = strategy.weights
    w_min = min(weights)
    w_max = max(weights)
    w_mean = sum(weights) / len(weights)
    w_std = (sum((w - w_mean)**2 for w in weights) / len(weights)) ** 0.5

    print(f"  Weight stats: min={w_min:.3f}, max={w_max:.3f}, mean={w_mean:.3f}, std={w_std:.3f}")


def analyze_neural_strategy(strategy, num_samples=2000):
    """Analyze what the neural network learned by testing specific scenarios."""
    from euchre.cards import Card, Deck, SUITS, RANKS, SAME_COLOR

    print("\n  Decision analysis (call probability by scenario):")
    print("  " + "-" * 60)

    # Test different trump counts
    for trump_count in [1, 2, 3, 4]:
        probs = []
        for _ in range(num_samples // 4):
            deck = Deck()
            deck.shuffle()
            trump_suit = random.choice(SUITS)
            hand = []

            trump_cards = [c for c in deck.cards if c.suit == trump_suit]
            random.shuffle(trump_cards)
            hand.extend(trump_cards[:trump_count])

            non_trump = [c for c in deck.cards if c.suit != trump_suit and c not in hand]
            random.shuffle(non_trump)
            hand.extend(non_trump[:5 - trump_count])

            turned = Card(trump_suit, random.choice(RANKS))
            seat = random.choice([1, 2, 3, 4])
            prob = strategy.get_call_probability(hand, turned, seat)
            probs.append(prob)

        avg_prob = sum(probs) / len(probs)
        bar_len = int(avg_prob * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        print(f"    {trump_count} trump:  {avg_prob:.1%}  [{bar}]")

    # Test high cards
    print()
    for card_name, make_hand in [
        ("Right bower + 1", lambda ts: [Card(ts, 'Jack'), Card(ts, '9')]),
        ("Left bower + 1", lambda ts: [Card(SAME_COLOR[ts], 'Jack'), Card(ts, '9')]),
        ("Trump Ace + 1", lambda ts: [Card(ts, 'Ace'), Card(ts, '9')]),
        ("Trump 9 + 10", lambda ts: [Card(ts, '9'), Card(ts, '10')]),
    ]:
        probs = []
        for _ in range(num_samples // 4):
            trump_suit = random.choice(SUITS)
            key_cards = make_hand(trump_suit)

            deck = Deck()
            deck.cards = [c for c in deck.cards if c not in key_cards]
            deck.shuffle()

            hand = list(key_cards)
            non_trump = [c for c in deck.cards if c.suit != trump_suit and c.suit != SAME_COLOR[trump_suit]]
            random.shuffle(non_trump)
            while len(hand) < 5:
                hand.append(non_trump.pop())

            turned = Card(trump_suit, random.choice(['9', '10', 'Queen']))
            seat = random.choice([1, 2, 3, 4])
            prob = strategy.get_call_probability(hand, turned, seat)
            probs.append(prob)

        avg_prob = sum(probs) / len(probs)
        bar_len = int(avg_prob * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        print(f"    {card_name:15s}:  {avg_prob:.1%}  [{bar}]")

    # Test position effect
    print()
    for seat in [1, 2, 3, 4]:
        seat_name = {1: "Left of dealer", 2: "Partner", 3: "Right of dealer", 4: "Dealer"}[seat]
        probs = []
        for _ in range(num_samples // 4):
            deck = Deck()
            deck.shuffle()
            hand = deck.deal(5)
            turned = deck.deal(1)[0]
            prob = strategy.get_call_probability(hand, turned, seat)
            probs.append(prob)

        avg_prob = sum(probs) / len(probs)
        bar_len = int(avg_prob * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        print(f"    {seat_name:15s}:  {avg_prob:.1%}  [{bar}]")

    # Round 2 analysis
    print()
    print("  Round 2 (choosing trump from non-turned suits):")
    r2_calls = 0
    r2_total = 0
    for _ in range(num_samples):
        deck = Deck()
        deck.shuffle()
        hand = deck.deal(5)
        turned = deck.deal(1)[0]
        seat = random.choice([1, 2, 3, 4])
        r2_total += 1
        result = strategy.choose_trump_round2(hand, turned.suit, seat)
        if result is not None:
            r2_calls += 1
    r2_rate = r2_calls / r2_total
    print(f"    Overall R2 call rate: {r2_rate:.1%}")

    # R2 by trump count in best candidate suit
    for trump_count in [1, 2, 3, 4]:
        probs = []
        for _ in range(num_samples // 4):
            deck = Deck()
            deck.shuffle()
            trump_suit = random.choice(SUITS)
            # Pick a turned card of a different suit
            other_suits = [s for s in SUITS if s != trump_suit]
            turned_suit = random.choice(other_suits)
            hand = []

            trump_cards = [c for c in deck.cards if c.suit == trump_suit]
            random.shuffle(trump_cards)
            hand.extend(trump_cards[:trump_count])

            non_trump = [c for c in deck.cards if c.suit != trump_suit and c not in hand]
            random.shuffle(non_trump)
            hand.extend(non_trump[:5 - trump_count])

            seat = random.choice([1, 2, 3, 4])
            inputs = encode_hand_relative(hand, trump_suit, seat, turned_card=None)
            prob = strategy._forward(inputs)
            probs.append(prob)

        avg_prob = sum(probs) / len(probs)
        bar_len = int(avg_prob * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        print(f"    {trump_count} trump (R2): {avg_prob:.1%}  [{bar}]")

    print("  " + "-" * 60)


def benchmark(strategy, num_hands=10000):
    """Benchmark strategy against baselines."""
    score_a, score_b = evaluate_fitness(strategy, AlwaysPassStrategy(), num_hands=num_hands)
    print(f"  vs Always-Pass ({num_hands} hands): {score_a} - {score_b}")

    score_a, score_b = evaluate_fitness(strategy, RandomCallStrategy(0.5), num_hands=num_hands)
    print(f"  vs Random-50%  ({num_hands} hands): {score_a} - {score_b}")


def main():
    parser = argparse.ArgumentParser(description="Evolve Euchre trump-calling neural networks")
    parser.add_argument("--population", type=int, default=50, help="Population size (default: 50)")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations (default: 100)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--hands", type=int, default=200, help="Hands per matchup (default: 200)")
    parser.add_argument("--linear", action="store_true", help="Use linear model instead of neural net")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    strategy_type = "linear" if args.linear else "neural"
    strategy_class = CallingStrategy if args.linear else NeuralStrategy

    print(f"Evolving {strategy_type} strategy with population={args.population}, "
          f"generations={args.generations}, hands/matchup={args.hands}")
    if not args.linear:
        print(f"Neural architecture: {INPUT_SIZE} inputs -> {HIDDEN_SIZE} hidden -> {OUTPUT_SIZE} output")
        print(f"Total parameters: {NeuralStrategy.NUM_WEIGHTS}")
    print()

    engine = EvolutionEngine(
        population_size=args.population,
        hands_per_matchup=args.hands,
        strategy_class=strategy_class,
    )

    best_strategy = None
    start_time = time.time()

    for gen in range(args.generations):
        gen_start = time.time()
        result = engine.run_generation()
        gen_elapsed = time.time() - gen_start

        best_strategy = result["best_strategy"]

        print(f"Gen {result['generation']:3d}  |  "
              f"best: {result['best_fitness']:+8.1f}  "
              f"median: {result['median_fitness']:+8.1f}  "
              f"R1: {result['call_rate']:.1%}  "
              f"R2: {result['call_rate_r2']:.1%}  "
              f"({gen_elapsed:.1f}s)")

    total_time = time.time() - start_time
    print(f"\nEvolution complete in {total_time:.1f}s")

    print(f"\n=== Best Evolved {strategy_type.title()} Strategy ===")
    if args.linear:
        from euchre.strategy import FEATURE_NAMES
        print("\n  Feature weights:")
        for name, weight in zip(FEATURE_NAMES, best_strategy.weights):
            bar_len = int(min(abs(weight), 3) / 3 * 20)
            if weight >= 0:
                bar = " " * 20 + "|" + "#" * bar_len
            else:
                bar = " " * (20 - bar_len) + "#" * bar_len + "|"
            print(f"    {name:20s} {weight:+7.3f}  {bar}")
    else:
        print_neural_info(best_strategy)
        analyze_neural_strategy(best_strategy)

    print("\n=== Benchmark ===")
    benchmark(best_strategy)

    # Save to file
    output_path = "best_strategy.json"
    with open(output_path, "w") as f:
        json.dump(best_strategy.to_dict(), f, indent=2)
    print(f"\nBest strategy saved to {output_path}")


if __name__ == "__main__":
    main()
