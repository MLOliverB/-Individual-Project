import numpy as np
from raumschach.figures import FIGURE_ID_MAP, FIGURE_NAME_MAP, Colour, King

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
        for str in setup:
            fig_name, pos_code = str.split(" ")
            figure, colour = FIGURE_NAME_MAP[fig_name]
            pos = ChessBoard.get_pos_coord(pos_code)
            self._add(figure, colour, pos)

    def move(self, from_pos, to_pos):
        if self[from_pos] == 0:
            print(f"No piece on {self.get_pos_code(from_pos)}")
            return
        moves, captures = ChessBoard.generate_figure_moves_captures(self.cube, from_pos)
        if to_pos not in moves and to_pos not in captures:
            print(f"Position {self.get_pos_code(to_pos)} is not reachable by this piece")
            return
        fig = self[from_pos]
        self[to_pos] = fig
        self[from_pos] = 0
        # TODO handle captures and pawn promotions in this function
        
    @staticmethod
    def get_moves_captures(board_a, colour):
        ally_positions = np.asarray((board_a > 0).nonzero()).T if colour == Colour.WHITE else np.asarray((board_a < 0).nonzero()).T
        enemy_positions = np.asarray((board_a > 0).nonzero()).T if colour == Colour.WHITE else np.asarray((board_a < 0).nonzero()).T

        # Generate moves for figures of the same colour
        ally_king_position = None
        ally_moves = {}
        ally_captures = {}
        for figure_pos in [ (p[0], p[1], p[2]) for p in ally_positions ]:
            if not ally_king_position and King == FIGURE_ID_MAP[board_a[figure_pos]][0]:
                ally_king_position = figure_pos
            moves, captures = ChessBoard.generate_figure_moves_captures(board_a, figure_pos)
            if moves:
                ally_moves[figure_pos] = moves
            if captures:
                ally_captures[figure_pos] = captures

        # Generate moves for figures of the opposite colour
        enemy_moves = {}
        enemy_captures = {}
        for figure_pos in [ (p[0], p[1], p[2]) for p in enemy_positions ]:
            moves, captures = ChessBoard.generate_figure_moves_captures(board_a, figure_pos)
            if moves:
                enemy_moves[figure_pos] = moves
            if captures:
                enemy_captures[figure_pos] = captures

        # Delete moves from the King's move set that would make the King check himself
        # We only need to check enemy moves since enemy captures capture on of our ally piece and the King can't move there anyways
        if ally_king_position in ally_moves:
            for enemy in enemy_moves:
                ally_moves[ally_king_position] = [ k_move for k_move in ally_moves[ally_king_position] if k_move not in enemy_moves[enemy] ]
        
        return (ally_moves, ally_captures)

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
    def generate_figure_moves_captures(board_a, figure_pos):
        figure, colour = FIGURE_ID_MAP[board_a[figure_pos]]
        plane, file, rank = figure_pos

        moves = []
        captures = []

        #iterate over passive moves
        for move in figure.moves:
            x = 1
            pv, fv, rv, next_x = move(x, colour)
            while ChessBoard.in_bounds(board_a.shape[0]-1, plane+pv, file+fv, rank+rv) and board_a[plane+pv, file+fv, rank+rv] == 0:
                moves.append((plane+pv, file+fv, rank+rv))
                if not next_x:
                    break
                x = next_x
                pv, fv, rv, next_x = move(x, colour)

        # iterate over capture moves
        for move in figure.captures:
            x = 1
            pv, fv, rv, next_x = move(x, colour)
            while ChessBoard.in_bounds(board_a.shape[0]-1, plane+pv, file+fv, rank+rv) and board_a[plane+pv, file+fv, rank+rv] != 0 and FIGURE_ID_MAP[board_a[plane+pv, file+fv, rank+rv]][1] == -colour:
                captures.append((plane+pv, file+fv, rank+rv))
                if not next_x:
                    break
                x = next_x
                pv, fv, rv, next_x = move(x, colour)

        # iterate over passive or capture moves
        for move in figure.move_or_capture:
            x = 1
            pv, fv, rv, next_x = move(x, colour)
            while ChessBoard.in_bounds(board_a.shape[0]-1, plane+pv, file+fv, rank+rv):
                if board_a[plane+pv, file+fv, rank+rv] == 0:
                    moves.append((plane+pv, file+fv, rank+rv))
                else:
                    if FIGURE_ID_MAP[board_a[plane+pv, file+fv, rank+rv]][1] == -colour:
                        captures.append((plane+pv, file+fv, rank+rv))
                    break
                if not next_x:
                    break
                x = next_x
                pv, fv, rv, next_x = move(x, colour)

        return (moves, captures)

    @staticmethod
    def in_bounds(bound_size, x, y, z):
        return (0 <= x <= bound_size) and (0 <= y <= bound_size) and (0 <= z <= bound_size)

    @staticmethod
    def get_pos_code(pos_coord):
        plane, file, rank = pos_coord
        return f"{chr(ord('A') + plane)}{chr(ord('a') + rank)}{chr(ord('1') + file)}"

    @staticmethod
    def get_pos_coord(pos_code):
        # On a chess board, rank(row) and file(column) are used in the inverted way w.r.t array notation
        # Therefore, the array is indexed by [plane, file, rank]
        plane, rank, file = pos_code
        return (ord(plane) - ord('A'), int(file)-1, ord(rank) - ord('a'))


class BoardState():
    def __init__(self, cube, colour, moves, captures, no_progress_count, state_repetition):
        self.cb = cube
        self.colour = colour
        self.moves = moves
        self.captures = captures
        self.no_progress_count = no_progress_count
        self.state_repetition = state_repetition