from .cards import Card, Deck, SAME_COLOR


class Player:
    def __init__(self, name, team):
        self.name = name
        self.hand = []
        self.team = team
        self.calling_strategy = None

    def receive_cards(self, cards):
        self.hand.extend(cards)

    def play_card(self, card):
        self.hand.remove(card)
        return card

    def choose_card(self, trick, game):
        led_suit = game.effective_suit(trick[0]) if trick else None
        same_suit_cards = [card for card in self.hand if game.effective_suit(card) == led_suit]

        if same_suit_cards:
            return max(same_suit_cards, key=lambda x: game.card_value(x))

        partner_winning = False
        if len(trick) >= 2:
            partner_card = trick[2] if len(trick) == 3 else trick[0]
            current_winner = game.compare_cards(trick, led_suit)
            partner_winning = partner_card == current_winner

        if partner_winning:
            return min(self.hand, key=lambda x: game.card_value(x))

        trump_cards = [card for card in self.hand if game.is_trump(card)]
        if trump_cards:
            return max(trump_cards, key=lambda x: game.card_value(x))

        return min(self.hand, key=lambda x: game.card_value(x))

    def decide_call(self, turned_card, game, seat_offset, is_dealer):
        if self.calling_strategy is None:
            return False
        return self.calling_strategy.should_call(self.hand, turned_card, game, seat_offset, is_dealer)

    def choose_discard(self, game):
        non_trump = [c for c in self.hand if not game.is_trump(c)]
        if non_trump:
            worst = min(non_trump, key=lambda x: game.card_value(x))
        else:
            worst = min(self.hand, key=lambda x: game.card_value(x))
        self.hand.remove(worst)
        return worst


class EuchreGame:
    def __init__(self, players=None, silent=False):
        self.deck = Deck()
        if players is not None:
            self.players = players
        else:
            self.players = [
                Player("Player 1", "Team A"),
                Player("Player 2", "Team B"),
                Player("Player 3", "Team A"),
                Player("Player 4", "Team B"),
            ]
        self.trump_suit = None
        self.turned_card = None
        self.calling_team = None
        self.dealer_index = 0
        self.team_tricks = {"Team A": 0, "Team B": 0}
        self.silent = silent

    def _print(self, msg):
        if not self.silent:
            print(msg)

    def deal_cards(self):
        self.deck.reset()
        self.deck.shuffle()
        for player in self.players:
            player.hand = []
            player.receive_cards(self.deck.deal(5))
        self.turned_card = self.deck.deal(1)[0]
        self._print(f"\nTurned up card: {self.turned_card}")

    def bidding_round_1(self):
        for i in range(4):
            seat = (self.dealer_index + 1 + i) % 4
            player = self.players[seat]
            seat_offset = i + 1  # 1 = left of dealer, ..., 4 = dealer
            is_dealer = (seat == self.dealer_index)
            if player.decide_call(self.turned_card, self, seat_offset, is_dealer):
                self.trump_suit = self.turned_card.suit
                self.calling_team = player.team
                self._print(f"{player.name} calls {self.trump_suit}!")
                self._dealer_pickup()
                return True
        self._print("All players pass.")
        return False

    def _dealer_pickup(self):
        dealer = self.players[self.dealer_index]
        dealer.receive_cards([self.turned_card])
        discarded = dealer.choose_discard(self)
        self._print(f"{dealer.name} picks up {self.turned_card} and discards {discarded}")

    def card_value(self, card):
        rank_order = {'9': 1, '10': 2, 'Jack': 3, 'Queen': 4, 'King': 5, 'Ace': 6}
        if card.suit == self.trump_suit:
            if card.rank == 'Jack':
                return 8  # Right bower
        elif card.rank == 'Jack' and card.suit == SAME_COLOR.get(self.trump_suit):
            return 7  # Left bower
        return rank_order[card.rank]

    def is_trump(self, card):
        if card.suit == self.trump_suit:
            return True
        if card.rank == 'Jack' and card.suit == SAME_COLOR.get(self.trump_suit):
            return True
        return False

    def effective_suit(self, card):
        if self.is_trump(card):
            return self.trump_suit
        return card.suit

    def compare_cards(self, cards_played, led_suit):
        trump_cards = [card for card in cards_played if self.is_trump(card)]
        if trump_cards:
            return max(trump_cards, key=lambda x: self.card_value(x))

        on_suit_cards = [card for card in cards_played if card.suit == led_suit and not self.is_trump(card)]
        if on_suit_cards:
            return max(on_suit_cards, key=lambda x: self.card_value(x))

        return cards_played[0]

    def play_trick(self, leader_index):
        trick = []
        play_order = []
        for i in range(4):
            seat = (leader_index + i) % 4
            player = self.players[seat]
            self._print(f"  {player.name}'s hand: {', '.join(str(c) for c in player.hand)}")
            card_to_play = player.choose_card(trick, self)
            card_played = player.play_card(card_to_play)
            trick.append(card_played)
            play_order.append(seat)
            self._print(f"  {player.name} plays {card_played}")

        winning_card = self.compare_cards(trick, self.effective_suit(trick[0]))
        winner_pos = trick.index(winning_card)
        winner_seat = play_order[winner_pos]
        winner = self.players[winner_seat]
        self._print(f"  {winner.name} wins the trick with {winning_card}")
        self.team_tricks[winner.team] += 1
        return winner_seat

    def _score_hand(self):
        if self.calling_team is None:
            return {"Team A": 0, "Team B": 0}

        caller_tricks = self.team_tricks[self.calling_team]
        defending_team = "Team B" if self.calling_team == "Team A" else "Team A"

        scores = {"Team A": 0, "Team B": 0}
        if caller_tricks >= 5:
            scores[self.calling_team] = 2  # march
        elif caller_tricks >= 3:
            scores[self.calling_team] = 1
        else:
            scores[defending_team] = 2  # euchred

        return scores

    def play_hand(self):
        self.team_tricks = {"Team A": 0, "Team B": 0}
        self.trump_suit = None
        self.calling_team = None
        self.deal_cards()

        called = self.bidding_round_1()
        if not called:
            return {"Team A": 0, "Team B": 0}

        self._print(f"\nTrump suit: {self.trump_suit}  (called by {self.calling_team})")

        leader = (self.dealer_index + 1) % 4
        for trick_num in range(5):
            self._print(f"\n--- Trick {trick_num + 1} ---")
            leader = self.play_trick(leader)

        scores = self._score_hand()
        self._print(f"\nTeam A tricks: {self.team_tricks['Team A']}")
        self._print(f"Team B tricks: {self.team_tricks['Team B']}")
        for team, pts in scores.items():
            if pts > 0:
                self._print(f"{team} scores {pts} point(s)")
        return scores

    def play_game(self):
        """Play a single hand (legacy interface for backwards compatibility)."""
        self.deal_cards()
        # Legacy mode: pick trump randomly if no bidding strategy
        if self.trump_suit is None:
            import random
            self.trump_suit = random.choice(['Hearts', 'Diamonds', 'Clubs', 'Spades'])
            self.calling_team = "Team A"

        self._print(f"\nTrump suit: {self.trump_suit}")
        leader = (self.dealer_index + 1) % 4
        for trick_num in range(5):
            self._print(f"\n--- Trick {trick_num + 1} ---")
            leader = self.play_trick(leader)

        self._print("\nGame over")
        self._print(f"Team A tricks: {self.team_tricks['Team A']}")
        self._print(f"Team B tricks: {self.team_tricks['Team B']}")
        winning_team = max(self.team_tricks, key=self.team_tricks.get)
        self._print(f"{winning_team} wins the hand!")
