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
    def move(board_a, from_pos, to_pos):
        if board_a[from_pos] == 0:
            raise Exception(f"Cannot move piece at {ChessBoard.get_pos_code(from_pos)} since this position is empty")
        fig = board_a[from_pos]
        board_a[to_pos] = fig
        board_a[from_pos] = 0
        # TODO handle captures and pawn promotions in this function

    @staticmethod
    def get_moves_captures(board_a, colour=None):
        white_positions = np.asarray((board_a > 0).nonzero()).T
        black_positions = np.asarray((board_a < 0).nonzero()).T

        white_king_pos = np.where(board_a==(King.id * Colour.WHITE))
        black_king_pos = np.where(board_a==(King.id * Colour.BLACK))

        white_king_position = (white_king_pos[0][0], white_king_pos[1][0], white_king_pos[2][0])
        black_king_position = (black_king_pos[0][0], black_king_pos[1][0], black_king_pos[2][0])

        board_positions = [white_positions, black_positions]
        king_positions = [white_king_position, black_king_position]

        colours = [Colour.WHITE, Colour.BLACK]

        moves = [{}, {}]
        captures = [{}, {}]

        # Generate moves and captures for both colours
        for i in range(2):
            ally_moves = moves[i]
            ally_captures = captures[i]
            for piece_pos in [ (p[0], p[1], p[2]) for p in board_positions[i] ]:
                piece_moves, piece_captures = ChessBoard.generate_figure_moves_captures(board_a, piece_pos)
                if piece_moves:
                    ally_moves[piece_pos] = set(piece_moves)
                if piece_captures:
                    ally_captures[piece_pos] = set(piece_captures)

        for i in (range(2) if not colour else [0] if colour == Colour.WHITE else [1]):
            # Simulate each ally move and capture and delete those actions that directly put the ally king in check
            ally_moves = moves[i]
            enemy_king_pos = white_king_position if colours[i] == Colour.BLACK else black_king_position
            for ally_piece in ally_moves:
                ally_piece_moves = ally_moves[ally_piece]
                safe_moves = set()
                for piece_move in ally_piece_moves:
                    # TODO IMPORTANT - Extend move simulation by checking whether the enemy king is put check-mate -> if we do then we don't even need to simulate the move
                    if piece_move == enemy_king_pos:
                        safe_moves.add(piece_move)
                        continue
                    sim_board_a = np.array(board_a) # Copy the board for the simulation
                    ChessBoard.move(sim_board_a, ally_piece, piece_move) # simulate the move
                    ally_king_pos = piece_move if (King.id*colours[i]) == sim_board_a[piece_move] else king_positions[i]
                    enemy_positions = np.asarray((sim_board_a > 0).nonzero()).T if -1*colours[i] == Colour.WHITE else np.asarray((sim_board_a < 0).nonzero()).T
                    is_ally_king_under_threat = False
                    for enemy_piece_pos in [ (p[0], p[1], p[2]) for p in enemy_positions ]:
                        enemy_piece_captures = ChessBoard.generate_figure_moves_captures(sim_board_a, enemy_piece_pos)[1]
                        if ally_king_pos in enemy_piece_captures:
                            is_ally_king_under_threat = True
                            break
                    if not is_ally_king_under_threat:
                        safe_moves.add(piece_move)
                ally_moves[ally_piece] = safe_moves

            ally_captures = captures[i]
            for ally_piece in ally_captures:
                ally_piece_captures = ally_captures[ally_piece]
                safe_captures = set()
                for piece_capture in ally_piece_captures:
                    sim_board_a = np.array(board_a) # Copy the board for the simulation
                    ChessBoard.move(sim_board_a, ally_piece, piece_capture) # simulate the move
                    ally_king_pos = piece_capture if (King.id*colours[i]) == sim_board_a[piece_capture] else king_positions[i]
                    enemy_positions = np.asarray((sim_board_a > 0).nonzero()).T if -1*colours[i] == Colour.WHITE else np.asarray((sim_board_a < 0).nonzero()).T
                    is_ally_king_under_threat = False
                    for enemy_piece_pos in [ (p[0], p[1], p[2]) for p in enemy_positions ]:
                        enemy_piece_captures = ChessBoard.generate_figure_moves_captures(sim_board_a, enemy_piece_pos)[1]
                        if ally_king_pos in enemy_piece_captures:
                            is_ally_king_under_threat = True
                            break
                    if not is_ally_king_under_threat:
                        safe_captures.add(piece_capture)
                ally_captures[ally_piece] = safe_captures

        # Revert all sets back to lists
        white_moves    = {key: [*value] for (key, value) in moves[0].items()    if value}
        white_captures = {key: [*value] for (key, value) in captures[0].items() if value}
        black_moves    = {key: [*value] for (key, value) in moves[1].items()    if value}
        black_captures = {key: [*value] for (key, value) in captures[1].items() if value}

        
        # Return the appropriate tuple in regard to colour
        if colour:
            if colour == Colour.WHITE:
                return (white_moves, white_captures)
            else:
                return (black_moves, black_captures)
        else:
            return ((white_moves, white_captures), (black_moves, black_captures))

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
            while ChessBoard.in_bounds(board_a.shape[0]-1, plane+pv, file+fv, rank+rv):
                if board_a[plane+pv, file+fv, rank+rv] != 0:
                    if FIGURE_ID_MAP[board_a[plane+pv, file+fv, rank+rv]][1] == -colour:
                        captures.append((plane+pv, file+fv, rank+rv))
                        break
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