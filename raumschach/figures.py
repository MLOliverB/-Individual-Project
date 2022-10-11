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
    def name(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def value(self):
        pass


class Pawn(Figure):
    id = 1
    name = ("Pawn", 'P', 'p')
    value = 1 # Set as the unit value

class Unicorn(Figure):
    id = 2
    name = ("Unicorn", 'U', 'u')
    value = 3 # Subject to change

class Rook(Figure):
    id = 3
    name = ("Rook", 'R', 'r')
    value = 5 # Subject to change

class Bishop(Figure):
    id = 4
    name = ("Bishop", 'B', 'b')
    value =  5 # Subject to change

class Knight(Figure):
    id = 5
    name = ("Knight", 'N', 'n')
    value = 9 # Subject to change

class Queen(Figure):
    id = 6
    name = ("Queen", 'Q', 'q')
    value = 15 # Subject to change

class King(Figure):
    id = 7
    name = ("King", 'K', 'k')
    value = sys.maxsize


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