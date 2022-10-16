import numpy as np
from raumschach.figures import build_figure_maps

INITIAL_5_5_BOARD_SETUP = [
    "r Ea5", "n Eb5", "k Ec5", "n Ed5", "r Ee5",
    "b Da5", "u Db5", "q Dc5", "b Dd5", "u De5",
    "p Ea4", "p Eb4", "p Ec4", "p Ed4", "p Ee4",
    "p Da4", "p Db4", "p Dc4", "p Dd4", "p De4",
    "P Ba2", "P Bb2", "P Bc2", "P Bd2", "P Be2",
    "P Aa2", "P Ab2", "P Ac2", "P Ad2", "P Ae2",
    "B Ba1", "U Bb1", "Q Bc1", "B Bd1", "U Be1",
    "R Aa1", "N Ab1", "K Ac1", "N Ad1", "R Ae1"
]



class ChessBoard:

    def __init__(self, board_size, setup):
        self.size = board_size
        self.cube = np.zeros((self.size, self.size, self.size), dtype=np.byte, order='C')
        self.figure_id_map, self.figure_name_map = build_figure_maps()
        for str in setup:
            fig_name, pos_code = str.split(" ")
            figure, colour = self.figure_name_map[fig_name]
            pos = ChessBoard.get_pos_coord(pos_code)
            self._add(figure, colour, pos)

    def move(self, from_pos, to_pos):
        if self[from_pos] == 0:
            print(f"No piece on {self.get_pos_code(from_pos)}")
            return
        moves, captures = self.get_figure_moves(from_pos)
        if to_pos not in moves and to_pos not in captures:
            print(f"Position {self.get_pos_code(to_pos)} is not reachable by this piece")
            return
        fig = self[from_pos]
        self[to_pos] = fig
        self[from_pos] = 0
        # TODO handle captures and pawn promotions in this function

    def get_figure_moves(self, figure_pos):
        if self[figure_pos] == 0:
            return ([], [])
        figure, colour = self.figure_id_map[self[figure_pos]]
        plane, file, rank = figure_pos
        moves = []
        captures = []

        # iterate over passive moves
        for move in figure.moves:
            x = 1
            pv, fv, rv, next_x = move(x, colour)
            while self._in_bounds(plane+pv, file+fv, rank+rv) and self[plane+pv, file+fv, rank+rv] == 0:
                moves.append((plane+pv, file+fv, rank+rv))
                if not next_x:
                    break
                x = next_x
                pv, fv, rv, next_x = move(x, colour)

        # iterate over capture moves
        for move in figure.captures:
            x = 1
            pv, fv, rv, next_x = move(x, colour)
            while self._in_bounds(plane+pv, file+fv, rank+rv) and self[plane+pv, file+fv, rank+rv] != 0 and self.figure_id_map[self[plane+pv, file+fv, rank+rv]][1] == -colour:
                captures.append((plane+pv, file+fv, rank+rv))
                if not next_x:
                    break
                x = next_x
                pv, fv, rv, next_x = move(x, colour)

        # iterate over passive or capture moves
        for move in figure.move_or_capture:
            x = 1
            pv, fv, rv, next_x = move(x, colour)
            while self._in_bounds(plane+pv, file+fv, rank+rv):
                if self[plane+pv, file+fv, rank+rv] == 0:
                    moves.append((plane+pv, file+fv, rank+rv))
                else:
                    if self.figure_id_map[self[plane+pv, file+fv, rank+rv]][1] == -colour:
                        captures.append((plane+pv, file+fv, rank+rv))
                    break
                x = next_x
                pv, fv, rv, next_x = move(x, colour)
        
        return (moves, captures)

    def _in_bounds(self, x, y, z):
        return (0 <= x <= self.size-1) and (0 <= y <= self.size-1) and (0 <= z <= self.size-1)

    def __getitem__(self, key):
        # TODO implement rigurous input checking for key
        if type(key) == str:
            return self.cube[ChessBoard.get_pos_coord(key)]
        elif type(key) == tuple:
            return self.cube[key]
        elif type(key) == int:
            return self.cube[key]
        else:
            raise KeyError(f"Key must be tuple or str, not {type(key)}")

    def __setitem__(self, key, value):
        # TODO implement rigurous input checking for key & value
        if type(key) == str:
            self.cube[ChessBoard.get_pos_coord(key)] = value
        elif type(key) == tuple:
            self.cube[key] = value
        elif type(key) == int:
            self.cube[key] = value
        else:
            raise KeyError(f"Key must be tuple or str, not {type(key)}")

    def _add(self, figure, colour, pos):
        self[pos] = figure.id * colour

    @staticmethod
    def get_pos_code(pos_coord):
        plane, file, rank = pos_coord
        return f"{ord('A') + plane}{ord('a') + rank}{file+1}"

    @staticmethod
    def get_pos_coord(pos_code):
        # On a chess board, rank(row) and file(column) are used in the inverted way w.r.t array notation
        # Therefore, the array is indexed by [plane, file, rank]
        plane, rank, file = pos_code
        return (ord(plane) - ord('A'), int(file)-1, ord(rank) - ord('a'))
