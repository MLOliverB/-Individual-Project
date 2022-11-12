from abc import ABCMeta, abstractmethod
import numpy as np


class Colour(object):
    WHITE = 1
    BLACK = -1

    @staticmethod
    def string(colour):
        return "White" if colour == Colour.WHITE else "Black"


class Figure(object, metaclass=ABCMeta):
    def __init__(self):
        # ...
        pass

    @property
    @abstractmethod
    def id(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def value(self):
        pass

    @property
    @abstractmethod
    def can_jump(self):
        pass

    @property
    @abstractmethod
    def passive_or_capture(self):
        pass

    @property
    @abstractmethod
    def passives(self):
        pass

    @property
    @abstractmethod
    def captures(self):
        pass


class Pawn(Figure):
    id = 1
    name = ('p', "Pawn", 'P')
    value = 1 # Set as the unit value
    can_jump = False
    passive_or_capture = []
    passives = [
        lambda x, c: (1*c, 0, 0, None),
        lambda x, c: (0, 1*c, 0, None),
    ]
    captures = [
        lambda x, c: (0, 1*c, 1*c, None),
        lambda x, c: (0, 1*c, -1*c, None),
        lambda x, c: (1*c, 0, 1*c, None),
        lambda x, c: (1*c, 0, -1*c, None),
        # lambda x, c: (1*c, 1*c, 0), # Variant that is debatable - Supported in A Guide to Fairy Chess by Anthony Dickens
    ]

class Unicorn(Figure):
    id = 2
    name = ('u', "Unicorn", 'U')
    value = 3 # Subject to change
    can_jump = False
    passive_or_capture = [
        lambda x, c: (x, x, x, x+1),
        lambda x, c: (-x, -x, -x, x+1),
        lambda x, c: (x, -x, x, x+1),
        lambda x, c: (-x, x, -x, x+1),
        lambda x, c: (-x, -x, x, x+1),
        lambda x, c: (x, x, -x, x+1),
        lambda x, c: (-x, x, x, x+1),
        lambda x, c: (x, -x, -x, x+1),
    ]
    passives = []
    captures = []

class Rook(Figure):
    id = 3
    name = ('r', "Rook", 'R')
    value = 5 # Subject to change
    can_jump = False
    passive_or_capture = [
        lambda x, c: (x, 0, 0, x+1),
        lambda x, c: (-x, 0, 0, x+1),
        lambda x, c: (0, x, 0, x+1),
        lambda x, c: (0, -x, 0, x+1),
        lambda x, c: (0, 0, x, x+1),
        lambda x, c: (0, 0, -x, x+1),
    ]
    passives = []
    captures = []

class Bishop(Figure):
    id = 4
    name = ('b', "Bishop", 'B')
    value =  5 # Subject to change
    can_jump = False
    passive_or_capture = [
        lambda x, c: (x, x, 0, x+1),
        lambda x, c: (-x, -x, 0, x+1),
        lambda x, c: (0, x, -x, x+1),
        lambda x, c: (0, -x, x, x+1),
        lambda x, c: (-x, x, 0, x+1),
        lambda x, c: (x, -x, 0, x+1),
        lambda x, c: (0, x, x, x+1),
        lambda x, c: (0, -x, -x, x+1),
    ]
    passives = []
    captures = []

class Knight(Figure):
    id = 5
    name = ('n', "Knight", 'N')
    value = 9 # Subject to change
    can_jump = True
    passive_or_capture = [
        lambda x, c: (0, 1, 2, None),
        lambda x, c: (0, 1, -2, None),
        lambda x, c: (0, -1, 2, None),
        lambda x, c: (0, -1, -2, None),
        lambda x, c: (0, 2, 1, None),
        lambda x, c: (0, 2, -1, None),
        lambda x, c: (0, -2, 1, None),
        lambda x, c: (0, -2, -1, None),

        lambda x, c: (1, 0, 2, None),
        lambda x, c: (1, 0, -2, None),
        lambda x, c: (1, 2, 0, None),
        lambda x, c: (1, -2, 0, None),
        lambda x, c: (-1, 0, 2, None),
        lambda x, c: (-1, 0, -2, None),
        lambda x, c: (-1, 2, 0, None),
        lambda x, c: (-1, -2, 0, None),
        
        lambda x, c: (2, 0, 1, None),
        lambda x, c: (2, 0, -1, None),
        lambda x, c: (2, 1, 0, None),
        lambda x, c: (2, -1, 0, None),
        lambda x, c: (-2, 0, 1, None),
        lambda x, c: (-2, 0, -1, None),
        lambda x, c: (-2, 1, 0, None),
        lambda x, c: (-2, -1, 0, None),
    ]
    passives = []
    captures = []

class Queen(Figure):
    id = 6
    name = ('q', "Queen", 'Q')
    value = 15 # Subject to change
    can_jump = False
    passive_or_capture = [
        # Unicorn Moves
        lambda x, c: (x, x, x, x+1),
        lambda x, c: (-x, -x, -x, x+1),
        lambda x, c: (x, -x, x, x+1),
        lambda x, c: (-x, x, -x, x+1),
        lambda x, c: (-x, -x, x, x+1),
        lambda x, c: (x, x, -x, x+1),
        lambda x, c: (-x, x, x, x+1),
        lambda x, c: (x, -x, -x, x+1),
        # Rook Moves
        lambda x, c: (x, 0, 0, x+1),
        lambda x, c: (-x, 0, 0, x+1),
        lambda x, c: (0, x, 0, x+1),
        lambda x, c: (0, -x, 0, x+1),
        lambda x, c: (0, 0, x, x+1),
        lambda x, c: (0, 0, -x, x+1),
        # Bishop Moves
        lambda x, c: (x, x, 0, x+1),
        lambda x, c: (-x, -x, 0, x+1),
        lambda x, c: (0, x, -x, x+1),
        lambda x, c: (0, -x, x, x+1),
        lambda x, c: (-x, x, 0, x+1),
        lambda x, c: (x, -x, 0, x+1),
        lambda x, c: (0, x, x, x+1),
        lambda x, c: (0, -x, -x, x+1),
    ]
    passives = []
    captures = []

class King(Figure):
    id = 7
    name = ('k', "King", 'K')
    value = np.inf
    can_jump = False
    passive_or_capture = [
        lambda x, c: (-1, -1, -1, None),
        lambda x, c: (-1, -1, 0, None),
        lambda x, c: (-1, -1, 1, None),
        lambda x, c: (-1, 0, -1, None),
        lambda x, c: (-1, 0, 0, None),
        lambda x, c: (-1, 0, 1, None),
        lambda x, c: (-1, 1, -1, None),
        lambda x, c: (-1, 1, 0, None),
        lambda x, c: (-1, 1, 1, None),

        lambda x, c: (0, -1, -1, None),
        lambda x, c: (0, -1, 0, None),
        lambda x, c: (0, -1, 1, None),
        lambda x, c: (0, 0, -1, None),
        # lambda x, c: (0, 0, 0, None), # This is just the current position
        lambda x, c: (0, 0, 1, None),
        lambda x, c: (0, 1, -1, None),
        lambda x, c: (0, 1, 0, None),
        lambda x, c: (0, 1, 1, None),

        lambda x, c: (1, -1, -1, None),
        lambda x, c: (1, -1, 0, None),
        lambda x, c: (1, -1, 1, None),
        lambda x, c: (1, 0, -1, None),
        lambda x, c: (1, 0, 0, None),
        lambda x, c: (1, 0, 1, None),
        lambda x, c: (1, 1, -1, None),
        lambda x, c: (1, 1, 0, None),
        lambda x, c: (1, 1, 1, None),
    ]
    passives = []
    captures = []


PROMOTABLE_FIGURES = [Unicorn, Rook, Bishop, Knight, Queen]


FIGURE_ID_MAP = {
    Colour.WHITE * Pawn.id    : (Pawn, Colour.WHITE),
    Colour.BLACK * Pawn.id    : (Pawn, Colour.BLACK),

    Colour.WHITE * Unicorn.id : (Unicorn, Colour.WHITE),
    Colour.BLACK * Unicorn.id : (Unicorn, Colour.BLACK),

    Colour.WHITE * Rook.id    : (Rook, Colour.WHITE),
    Colour.BLACK * Rook.id    : (Rook, Colour.BLACK),

    Colour.WHITE * Bishop.id  : (Bishop, Colour.WHITE),
    Colour.BLACK * Bishop.id  : (Bishop, Colour.BLACK),

    Colour.WHITE * Knight.id  : (Knight, Colour.WHITE),
    Colour.BLACK * Knight.id  : (Knight, Colour.BLACK),

    Colour.WHITE * Queen.id   : (Queen, Colour.WHITE),
    Colour.BLACK * Queen.id   : (Queen, Colour.BLACK),

    Colour.WHITE * King.id    : (King, Colour.WHITE),
    Colour.BLACK * King.id    : (King, Colour.BLACK),
}

FIGURE_NAME_MAP = {
    Pawn.name[1+(1*Colour.WHITE)] : (Pawn, Colour.WHITE),
    Pawn.name[1+(1*Colour.BLACK)] : (Pawn, Colour.BLACK),

    Unicorn.name[1+(1*Colour.WHITE)] : (Unicorn, Colour.WHITE),
    Unicorn.name[1+(1*Colour.BLACK)] : (Unicorn, Colour.BLACK),

    Rook.name[1+(1*Colour.WHITE)] : (Rook, Colour.WHITE),
    Rook.name[1+(1*Colour.BLACK)] : (Rook, Colour.BLACK),

    Bishop.name[1+(1*Colour.WHITE)] : (Bishop, Colour.WHITE),
    Bishop.name[1+(1*Colour.BLACK)] : (Bishop, Colour.BLACK),

    Knight.name[1+(1*Colour.WHITE)] : (Knight, Colour.WHITE),
    Knight.name[1+(1*Colour.BLACK)] : (Knight, Colour.BLACK),

    Queen.name[1+(1*Colour.WHITE)] : (Queen, Colour.WHITE),
    Queen.name[1+(1*Colour.BLACK)] : (Queen, Colour.BLACK),

    King.name[1+(1*Colour.WHITE)] : (King, Colour.WHITE),
    King.name[1+(1*Colour.BLACK)] : (King, Colour.BLACK),
}
