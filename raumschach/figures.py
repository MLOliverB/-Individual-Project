from abc import ABCMeta, abstractmethod
import sys

class Figure(metaclass=ABCMeta):
    def __init__(self):
        # ...
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def value(self):
        pass

class King(Figure):
    name = ("King", 'K', 'k')
    value = sys.maxsize

class Queen(Figure):
    name = ("Queen", 'Q', 'q')
    value = 15 # Subject to change

class Knight(Figure):
    name = ("Knight", 'N', 'n')
    value = 9 # Subject to change

class Bishop(Figure):
    name = ("Bishop", 'B', 'b')
    value =  5 # Subject to change

class Rook(Figure):
    name = ("Rook", 'R', 'r')
    value = 5 # Subject to change

class Unicorn(Figure):
    name = ("Unicorn", 'U', 'u')
    value = 3 # Subject to change

class Pawn(Figure):
    name = ("Pawn", 'P', 'p')
    value = 1 # Set as the unit value