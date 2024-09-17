import random

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.rank} of {self.suit}"

class Deck:
    def __init__(self):
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['9', '10', 'Jack', 'Queen', 'King', 'Ace']
        self.cards = [Card(suit, rank) for suit in suits for rank in ranks]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards):
        return [self.cards.pop() for _ in range(num_cards)]

class Player:
    def __init__(self, name, team):
        self.name = name
        self.hand = []
        self.team = team

    def receive_cards(self, cards):
        self.hand.extend(cards)

    def play_card(self, card):
        self.hand.remove(card)
        return card

class EuchreGame:
    def __init__(self):
        self.deck = Deck()
        self.players = [
            Player("Player 1", "Team A"),
            Player("Player 2", "Team B"),
            Player("Player 3", "Team A"),
            Player("Player 4", "Team B")
        ]
        self.trump_suit = None
        self.trick_winner = None
        self.team_tricks = {"Team A": 0, "Team B": 0}

    def deal_cards(self):
        self.deck.shuffle()
        for player in self.players:
            player.receive_cards(self.deck.deal(5))
        self.trump_suit = random.choice(['Hearts', 'Diamonds', 'Clubs', 'Spades'])

    def compare_cards(self, cards_played, led_suit):
        trump_cards = [card for card in cards_played if card.suit == self.trump_suit]
        if trump_cards:
            return max(trump_cards, key=lambda x: self.card_value(x))
        
        on_suit_cards = [card for card in cards_played if card.suit == led_suit]
        if on_suit_cards:
            return max(on_suit_cards, key=lambda x: self.card_value(x))
        
        return cards_played[0]  # If no trump or on-suit cards, first card wins

    def card_value(self, card):
        rank_order = {'9': 1, '10': 2, 'Jack': 3, 'Queen': 4, 'King': 5, 'Ace': 6}
        if card.suit == self.trump_suit:
            if card.rank == 'Jack':
                return 8  # Right bower
            rank_order['Jack'] = 7  # Left bower
        return rank_order[card.rank]

    def play_round(self):
        print(f"\nTrump suit: {self.trump_suit}")
        cards_played = []
        for player in self.players:
            print(f"{player.name}'s hand: {', '.join(str(card) for card in player.hand)}")
            if player.hand:
                card_played = player.play_card(player.hand[0])  # Simplified: just play the first card
                cards_played.append(card_played)
                print(f"{player.name} plays {card_played}")

        winning_card = self.compare_cards(cards_played, cards_played[0].suit)
        self.trick_winner = self.players[cards_played.index(winning_card)]
        print(f"{self.trick_winner.name} wins the trick with {winning_card}")
        self.team_tricks[self.trick_winner.team] += 1

    def play_game(self):
        self.deal_cards()
        for _ in range(5):  # 5 tricks in a hand
            self.play_round()
        
        print("\nGame over")
        print(f"Team A tricks: {self.team_tricks['Team A']}")
        print(f"Team B tricks: {self.team_tricks['Team B']}")
        winning_team = max(self.team_tricks, key=self.team_tricks.get)
        print(f"{winning_team} wins the hand!")

if __name__ == "__main__":
    game = EuchreGame()
    game.play_game()