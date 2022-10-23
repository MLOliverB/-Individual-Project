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

        # Generate moves for white pieces
        white_moves = {}
        white_captures = {}
        for figure_pos in [ (p[0], p[1], p[2]) for p in white_positions ]:
            moves, captures = ChessBoard.generate_figure_moves_captures(board_a, figure_pos)
            if moves:
                white_moves[figure_pos] = set(moves)
            if captures:
                white_captures[figure_pos] = set(captures)

        # Generate moves for white pieces
        black_moves = {}
        black_captures = {}
        for figure_pos in [ (p[0], p[1], p[2]) for p in black_positions ]:
            moves, captures = ChessBoard.generate_figure_moves_captures(board_a, figure_pos)
            if moves:
                black_moves[figure_pos] = set(moves)
            if captures:
                black_captures[figure_pos] = set(captures)

        # If colour is not specified, delete king moves and captures for each colour, else only do it for the specified colour
        ally_moves_lst = [white_moves, black_moves] if not colour else [white_moves] if colour == Colour.WHITE else [black_moves]
        ally_captures_lst = [white_captures, black_captures] if not colour else [white_captures] if colour == Colour.WHITE else [black_captures]
        ally_king_pos_lst = [white_king_position, black_king_position] if not colour else [white_king_position] if colour == Colour.WHITE else [black_king_position]
        enemy_king_pos_lst = [black_king_position, white_king_position] if not colour else [black_king_position] if colour == Colour.WHITE else [white_king_position]
        enemy_moves_lst = [black_moves, white_moves] if not colour else [black_moves] if colour == Colour.WHITE else [white_moves]
        enemy_captures_lst = [black_captures, white_captures] if not colour else [black_captures] if colour == Colour.WHITE else [white_captures]
        loop_colour_lst = [Colour.WHITE, Colour.BLACK] if not colour else [colour]
        for i in range(len(ally_king_pos_lst)):
            ally_moves = ally_moves_lst[i]
            ally_captures = ally_captures_lst[i]
            ally_king_pos = ally_king_pos_lst[i]
            enemy_king_pos = enemy_king_pos_lst[i]
            enemy_moves = enemy_moves_lst[i]
            enemy_captures = enemy_captures_lst[i]
            loop_colour = loop_colour_lst[i]
            # # Iterate over the moves of all enemy pieces
            # # The ally king must not move on a field on which an enemy piece can move (this would make him check and incidentally mate himself)
            # # It also must not capture a piece to which another enemy piece can move
            # all_enemy_moves_captures = set()
            # for enemy_piece in enemy_moves:
            #     all_enemy_moves_captures += enemy_moves[enemy_piece]
            # for enemy_piece in enemy_captures:

            # if ally_king_pos in ally_moves:
            #     ally_moves[ally_king_pos] -= all_enemy_moves_captures
            # if ally_king_pos in ally_captures:
            #     ally_captures[ally_king_pos] -= all_enemy_moves_captures
            # # if ally_king_pos in ally_moves:
            # #     for enemy_piece in enemy_moves:
            # #         ally_moves[ally_king_pos] = ally_moves[ally_king_pos].difference(enemy_moves[enemy_piece])
            # #         # set.di
            # #         # move_intersection = ally_moves[ally_king_pos].intersection(enemy_moves[enemy_piece])
            # #         # if move_intersection:
            # #         #     ally_moves[ally_king_pos].difference_update(move_intersection)
            # # if ally_king_pos in ally_captures:
            # #     for enemy_piece in enemy_moves:
            # #         ally_captures[ally_king_pos] = ally_captures[ally_king_pos].difference(enemy_captures[enemy_piece])
            # #         # capture_intersection = ally_captures[ally_king_pos].intersection(enemy_moves[enemy_piece])
            # #         # if capture_intersection:
            # #         #     ally_captures[ally_king_pos].difference_update(capture_intersection)

            # Iterate over the captures of enemy pieces and determine whether the ally king is under check
            is_ally_king_check = False
            for enemy_piece in enemy_captures:
                if ally_king_pos in enemy_captures[enemy_piece]:
                    is_ally_king_check = True
                    break

            if True or is_ally_king_check:
                # The king is under attack
                # Restrict all moves / captures to moves that either checkmate the enemy king
                # or to moves / captures that free the king from the check situation
                # This requires us to simulate every move and determine wheher the ally king is still under check
                for sim_move_ally_piece in ally_moves:
                    sim_move_is_king = board_a[sim_move_ally_piece] == King.id * loop_colour
                    saving_moves = set()
                    for move in ally_moves[sim_move_ally_piece]:
                        sim_moves_board_a = np.array(board_a)
                        ChessBoard.move(sim_moves_board_a, sim_move_ally_piece, move)
                        sim_move_ally_king_pos = move if sim_move_is_king else ally_king_pos
                        sim_move_enemy_positions = np.asarray((sim_moves_board_a < 0).nonzero()).T if loop_colour == Colour.WHITE else np.asarray((sim_moves_board_a > 0).nonzero()).T
                        is_saving_move = True
                        for sim_move_figure_pos in [ (p[0], p[1], p[2]) for p in sim_move_enemy_positions ]:
                            sim_gen_captures = ChessBoard.generate_figure_moves_captures(sim_moves_board_a, sim_move_figure_pos)[1]
                            if sim_move_ally_king_pos in sim_gen_captures:
                                is_saving_move = False
                                break
                        if is_saving_move:
                            saving_moves.add(move)
                    ally_moves[sim_move_ally_piece] = saving_moves

                for sim_capture_ally_piece in ally_captures:
                    sim_capture_is_king = board_a[sim_capture_ally_piece] == King.id * loop_colour
                    saving_captures = set()
                    for capture in ally_captures[sim_capture_ally_piece]:
                        if enemy_king_pos == capture: # If this capture checkmates the other king, that works too
                            saving_captures.add(enemy_king_pos)
                        else:
                            sim_capture_board_a = np.array(board_a)
                            ChessBoard.move(sim_capture_board_a, sim_capture_ally_piece, capture)
                            sim_capture_ally_king_pos = capture if sim_capture_is_king else ally_king_pos
                            sim_capture_enemy_positions = np.asarray((sim_capture_board_a < 0).nonzero()).T if loop_colour == Colour.WHITE else np.asarray((sim_capture_board_a > 0).nonzero()).T
                            is_saving_capture = True
                            for sim_capture_figure_pos in [ (p[0], p[1], p[2]) for p in sim_capture_enemy_positions ]:
                                sim_gen_captures = ChessBoard.generate_figure_moves_captures(sim_capture_board_a, sim_capture_figure_pos)[1]
                                if sim_capture_ally_king_pos in sim_gen_captures:
                                    is_saving_capture = False
                                    break
                            if is_saving_capture:
                                saving_captures.add(capture)
                    ally_captures[sim_capture_ally_piece] = saving_captures



        # Revert all sets back to lists
        white_moves    = {key: [*value] for (key, value) in white_moves.items()    if value}
        white_captures = {key: [*value] for (key, value) in white_captures.items() if value}
        black_moves    = {key: [*value] for (key, value) in black_moves.items()    if value}
        black_captures = {key: [*value] for (key, value) in black_captures.items() if value}

        
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