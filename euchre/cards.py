import random


class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.rank} of {self.suit}"

    def __repr__(self):
        return f"Card('{self.suit}', '{self.rank}')"

    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return hash((self.suit, self.rank))


SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
RANKS = ['9', '10', 'Jack', 'Queen', 'King', 'Ace']

SAME_COLOR = {
    'Hearts': 'Diamonds', 'Diamonds': 'Hearts',
    'Clubs': 'Spades', 'Spades': 'Clubs'
}


class Deck:
    def __init__(self):
        self.cards = []
        self.reset()

    def reset(self):
        self.cards = [Card(suit, rank) for suit in SUITS for rank in RANKS]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards):
        return [self.cards.pop() for _ in range(num_cards)]
