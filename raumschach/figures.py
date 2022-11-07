from abc import ABCMeta, abstractmethod
import sys


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
    def single_passive_or_capture(self):
        pass

    @property
    @abstractmethod
    def single_passives(self):
        pass

    @property
    @abstractmethod
    def single_captures(self):
        pass

    @property
    @abstractmethod
    def successive_passive_or_capture(self):
        pass

    @property
    @abstractmethod
    def successive_passives(self):
        pass

    @property
    @abstractmethod
    def successive_captures(self):
        pass


class Pawn(Figure):
    id = 1
    name = ('p', "Pawn", 'P')
    value = 1 # Set as the unit value
    single_passives_or_captures = []
    single_passives = [
        [ 1, 0, 0],
        [ 0, 1, 0],
    ]
    single_captures = [
        [ 0, 1, 1],
        [ 0, 1,-1],
        [ 1, 0, 1],
        [ 1, 0,-1],
        #[1, 1, 0] # Variant that is debatable - Supported in A Guide to Fairy Chess by Anthony Dickens
    ]
    successive_passives_or_captures = []
    successive_passives = []
    successive_captures = []

class Unicorn(Figure):
    id = 2
    name = ('u', "Unicorn", 'U')
    value = 3 # Subject to change
    single_passives_or_captures = []
    single_passives = []
    single_captures = []
    successive_passives_or_captures = [
        [ 1,  1,  1],
        [-1, -1, -1],
        [ 1, -1,  1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
    ]
    successive_passives = []
    successive_captures = []

class Rook(Figure):
    id = 3
    name = ('r', "Rook", 'R')
    value = 5 # Subject to change
    single_passives_or_captures = []
    single_passives = []
    single_captures = []
    successive_passives_or_captures = [
        [ 1,  0,  0],
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
        [ 0,  0, -1],
    ]
    successive_passives = []
    successive_captures = []

class Bishop(Figure):
    id = 4
    name = ('b', "Bishop", 'B')
    value =  5 # Subject to change
    single_passives_or_captures = []
    single_passives = []
    single_captures = []
    successive_passives_or_captures = [
        [ 0,  1, -1],
        [ 0, -1,  1],
        [ 0,  1,  1],
        [ 0, -1, -1],
        [ 1,  1,  0],
        [-1, -1,  0],
        [-1,  1,  0],
        [ 1, -1,  0],
        
    ]
    successive_passives = []
    successive_captures = []

class Knight(Figure):
    id = 5
    name = ('n', "Knight", 'N')
    value = 9 # Subject to change
    single_passives_or_captures = [
        [ 0,  1,  2],
        [ 0, -1, -2],
        [ 0,  1, -2],
        [ 0, -1,  2],
        [ 0,  2,  1],
        [ 0, -2, -1],
        [ 0,  2, -1],
        [ 0, -2,  1],
        [ 1,  0,  2],
        [-1,  0, -2],
        [ 1,  0, -2],
        [-1,  0,  2],
        [ 1,  2,  0],
        [-1, -2,  0],
        [ 1, -2,  0],
        [-1,  2,  0],
        [ 2,  0,  1],
        [-2,  0, -1],
        [ 2,  0, -1],
        [-2,  0,  1],
        [ 2,  1,  0],
        [-2, -1,  0],
        [ 2, -1,  0],
        [-2,  1,  0],
    ]
    single_passives = []
    single_captures = []
    successive_passives_or_captures = []
    successive_passives = []
    successive_captures = []

class Queen(Figure):
    id = 6
    name = ('q', "Queen", 'Q')
    value = 15 # Subject to change
    single_passives_or_captures = []
    single_passives = []
    single_captures = []
    successive_passives_or_captures = Unicorn.successive_passives_or_captures + Rook.successive_passives_or_captures + Bishop.successive_passives_or_captures
    successive_passives = []
    successive_captures = []

class King(Figure):
    id = 7
    name = ('k', "King", 'K')
    value = sys.maxsize
    single_passives_or_captures = Unicorn.successive_passives_or_captures + Rook.successive_passives_or_captures + Bishop.successive_passives_or_captures
    single_passives = []
    single_captures = []
    successive_passives_or_captures = []
    successive_passives = []
    successive_captures = []


FIGURES = [Pawn, Unicorn, Rook, Bishop, Knight, Queen, King]
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
