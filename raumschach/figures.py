from abc import ABCMeta, abstractmethod
import sys


class Colour(object):
    WHITE = 1
    BLACK = -1


class Figure(metaclass=ABCMeta):
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
    def move_or_capture(self):
        pass

    @property
    @abstractmethod
    def moves(self):
        pass

    @property
    @abstractmethod
    def captures(self):
        pass


class Pawn(Figure):
    id = 1
    name = ("Pawn", 'P', 'p')
    value = 1 # Set as the unit value
    can_jump = False
    move_or_capture = []
    moves = [
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
    name = ("Unicorn", 'U', 'u')
    value = 3 # Subject to change
    can_jump = False
    move_or_capture = [
        lambda x, c: (x, x, x, x+1),
        lambda x, c: (-x, -x, -x, x+1),
        lambda x, c: (x, -x, x, x+1),
        lambda x, c: (-x, x, -x, x+1),
        lambda x, c: (-x, -x, x, x+1),
        lambda x, c: (x, x, -x, x+1),
        lambda x, c: (-x, x, x, x+1),
        lambda x, c: (x, -x, -x, x+1),
    ]
    moves = []
    captures = []

class Rook(Figure):
    id = 3
    name = ("Rook", 'R', 'r')
    value = 5 # Subject to change
    can_jump = False
    move_or_capture = [
        lambda x, c: (x, 0, 0, x+1),
        lambda x, c: (-x, 0, 0, x+1),
        lambda x, c: (0, x, 0, x+1),
        lambda x, c: (0, -x, 0, x+1),
        lambda x, c: (0, 0, x, x+1),
        lambda x, c: (0, 0, -x, x+1),
    ]
    moves = []
    captures = []

class Bishop(Figure):
    id = 4
    name = ("Bishop", 'B', 'b')
    value =  5 # Subject to change
    can_jump = False
    move_or_capture = [
        lambda x, c: (x, x, 0, x+1),
        lambda x, c: (-x, -x, 0, x+1),
        lambda x, c: (0, x, -x, x+1),
        lambda x, c: (0, -x, x, x+1),
        lambda x, c: (-x, x, 0, x+1),
        lambda x, c: (x, -x, 0, x+1),
        lambda x, c: (0, x, x, x+1),
        lambda x, c: (0, -x, -x, x+1),
    ]
    moves = []
    captures = []

class Knight(Figure):
    id = 5
    name = ("Knight", 'N', 'n')
    value = 9 # Subject to change
    can_jump = True
    move_or_capture = [
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
    moves = []
    captures = []

class Queen(Figure):
    id = 6
    name = ("Queen", 'Q', 'q')
    value = 15 # Subject to change
    can_jump = False
    move_or_capture = [
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
    moves = []
    captures = []

class King(Figure):
    id = 7
    name = ("King", 'K', 'k')
    value = sys.maxsize
    can_jump = False
    move_or_capture = [
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
    moves = []
    captures = []


FIGURES = [Pawn, Unicorn, Rook, Bishop, Knight, Queen, King]


def build_figure_maps():
    figure_id_map = {}
    figure_name_map = {}
    for figure in FIGURES:
        figure_id_map[figure.id * Colour.WHITE] = (figure, Colour.WHITE)
        figure_id_map[figure.id * Colour.BLACK] = (figure, Colour.BLACK)
        figure_name_map[figure.name[1]] = (figure, Colour.WHITE)
        figure_name_map[figure.name[2]] = (figure, Colour.BLACK)
    return (figure_id_map, figure_name_map)