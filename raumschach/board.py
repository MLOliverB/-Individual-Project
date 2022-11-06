import numpy as np
from raumschach.figures import FIGURE_ID_MAP, FIGURE_NAME_MAP, PROMOTABLE_FIGURES, Colour, King, Pawn
from raumschach.render import render_board_ascii

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
        # FEATURE implement rigurous input checking for key
        if type(key) == str:
            return self.cube[ChessBoard.get_pos_coord(key)]
        elif type(key) == tuple:
            return self.cube[key]
        elif type(key) == int:
            return self.cube[key]
        else:
            raise KeyError(f"Key must be tuple or str, not {type(key)}")

    def __setitem__(self, key, value):
        # FEATURE implement rigurous input checking for key & value
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
    def move(board_a, from_action, to_action):
        from_piece, from_coord = from_action
        to_piece, to_coord = to_action
        if board_a[from_coord] == 0:
            raise Exception(f"Cannot move piece at {ChessBoard.get_pos_code(from_coord)} since this position is empty - Move {from_action}, {to_action}")
        if board_a[from_coord] != from_piece:
            raise Exception(f"Internal state error for piece at {ChessBoard.get_pos_code(from_coord)} - actual piece does not match expeceted piece - Move {from_action}, {to_action}")
        board_a[to_coord] = to_piece
        board_a[from_coord] = 0

    @staticmethod
    def get_passives_captures(board_a, colour=None):
        white_positions = np.asarray((board_a > 0).nonzero()).T
        black_positions = np.asarray((board_a < 0).nonzero()).T

        # print(np.where(board_a==(King.id * Colour.WHITE)))
        # print(np.where(board_a==(King.id * Colour.BLACK)))

        white_king_position = (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(board_a==(King.id * Colour.WHITE)))
        black_king_position = (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(board_a==(King.id * Colour.BLACK)))

        board_positions = [white_positions, black_positions]
        king_positions = [white_king_position, black_king_position]

        colours = [Colour.WHITE, Colour.BLACK]

        passives = [{}, {}]
        captures = [{}, {}]

        # Generate passives and captures for both colours
        for i in range(2):
            ally_passives = passives[i]
            ally_captures = captures[i]
            for piece_pos in [ (p[0], p[1], p[2]) for p in board_positions[i] ]:
                piece_passives, piece_captures = ChessBoard._generate_figure_passives_captures(board_a, piece_pos)
                if piece_passives:
                    ally_passives[(board_a[piece_pos], piece_pos)] = set(piece_passives)
                if piece_captures:
                    ally_captures[(board_a[piece_pos], piece_pos)] = set(piece_captures)

        safe_passives = [{}, {}]
        safe_captures = [{}, {}]

        # Simulate each possible move and delete those that directly put the ally king in check (since those are illegal moves)

        for (pw_from_piece_id, pw_from_pos) in passives[0]: # White passives
            safe_w_passives = set()
            for (pw_to_piece_id, pw_to_pos) in passives[0][(pw_from_piece_id, pw_from_pos)]:
                pw_ally_king_pos = white_king_position
                pw_enemy_king_pos = black_king_position
                if pw_to_pos == pw_enemy_king_pos:
                    safe_w_passives.add((pw_to_piece_id, pw_to_pos))
                else:
                    pw_sim_board_a = np.array(board_a)
                    ChessBoard.move(pw_sim_board_a, (pw_from_piece_id, pw_from_pos), (pw_to_piece_id, pw_to_pos))
                    if pw_to_piece_id == King.id * 1: # We are moving the king
                        pw_ally_king_pos = pw_to_pos
                    pw_is_ally_king_under_threat = False
                    pw_enemy_positions = np.asarray((pw_sim_board_a < 0).nonzero()).T
                    for pw_enemy_pos in [ (p[0], p[1], p[2]) for p in pw_enemy_positions ]:
                        pw_enemy_captures = ChessBoard._generate_figure_passives_captures(pw_sim_board_a, pw_enemy_pos)[1]
                        if pw_ally_king_pos in [ x[1] for x in pw_enemy_captures ]:
                            pw_is_ally_king_under_threat = True
                            break
                    if not pw_is_ally_king_under_threat:
                        safe_w_passives.add((pw_to_piece_id, pw_to_pos))
            safe_passives[0][(pw_from_piece_id, pw_from_pos)] = safe_w_passives

        for (cw_from_piece_id, cw_from_pos) in captures[0]: # White captures
            safe_w_captures = set()
            for (cw_to_piece_id, cw_to_pos) in captures[0][(cw_from_piece_id, cw_from_pos)]:
                cw_ally_king_pos = white_king_position
                cw_enemy_king_pos = black_king_position
                if cw_to_pos == cw_enemy_king_pos:
                    safe_w_captures.add((cw_to_piece_id, cw_to_pos))
                else:
                    cw_sim_board_a = np.array(board_a)
                    ChessBoard.move(cw_sim_board_a, (cw_from_piece_id, cw_from_pos), (cw_to_piece_id, cw_to_pos))
                    if cw_to_piece_id == King.id * 1: # We are moving the king
                        cw_ally_king_pos = cw_to_pos
                    cw_is_ally_king_under_threat = False
                    cw_enemy_positions = np.asarray((cw_sim_board_a < 0).nonzero()).T
                    for cw_enemy_pos in [ (p[0], p[1], p[2]) for p in cw_enemy_positions ]:
                        cw_enemy_captures = ChessBoard._generate_figure_passives_captures(cw_sim_board_a, cw_enemy_pos)[1]
                        if cw_ally_king_pos in [ x[1] for x in cw_enemy_captures ]:
                            cw_is_ally_king_under_threat = True
                            break
                    if not cw_is_ally_king_under_threat:
                        safe_w_captures.add((cw_to_piece_id, cw_to_pos))
            safe_captures[0][(cw_from_piece_id, cw_from_pos)] = safe_w_captures

        for (pb_from_piece_id, pb_from_pos) in passives[1]: # Black passives
            safe_b_passives = set()
            for (pb_to_piece_id, pb_to_pos) in passives[1][(pb_from_piece_id, pb_from_pos)]:
                pb_ally_king_pos = black_king_position
                pb_enemy_king_pos = white_king_position
                if pb_to_pos == pb_enemy_king_pos:
                    safe_b_passives.add((pb_to_piece_id, pb_to_pos))
                else:
                    pb_sim_board_a = np.array(board_a)
                    ChessBoard.move(pb_sim_board_a, (pb_from_piece_id, pb_from_pos), (pb_to_piece_id, pb_to_pos))
                    if pb_to_piece_id == King.id * -1: # We are moving the king
                        pb_ally_king_pos = pb_to_pos
                    pb_is_ally_king_under_threat = False
                    pb_enemy_positions = np.asarray((pb_sim_board_a > 0).nonzero()).T
                    for pb_enemy_pos in [ (p[0], p[1], p[2]) for p in pb_enemy_positions ]:
                        pb_enemy_captures = ChessBoard._generate_figure_passives_captures(pb_sim_board_a, pb_enemy_pos)[1]
                        if pb_ally_king_pos in [ x[1] for x in pb_enemy_captures ]:
                            pb_is_ally_king_under_threat = True
                            break
                    if not pb_is_ally_king_under_threat:
                        safe_b_passives.add((pb_to_piece_id, pb_to_pos))
            safe_passives[1][(pb_from_piece_id, pb_from_pos)] = safe_b_passives

        for (cb_from_piece_id, cb_from_pos) in captures[1]: # Black captures
            safe_b_captures = set()
            for (cb_to_piece_id, cb_to_pos) in captures[1][(cb_from_piece_id, cb_from_pos)]:
                cb_ally_king_pos = black_king_position
                cb_enemy_king_pos = white_king_position
                if cb_to_pos == cb_enemy_king_pos:
                    safe_b_captures.add((cb_to_piece_id, cb_to_pos))
                else:
                    cb_sim_board_a = np.array(board_a)
                    ChessBoard.move(cb_sim_board_a, (cb_from_piece_id, cb_from_pos), (cb_to_piece_id, cb_to_pos))
                    if cb_to_piece_id == King.id * -1: # We are moving the king
                        cb_ally_king_pos = cb_to_pos
                    cb_is_ally_king_under_threat = False
                    cb_enemy_positions = np.asarray((cb_sim_board_a > 0).nonzero()).T
                    for cb_enemy_pos in [ (p[0], p[1], p[2]) for p in cb_enemy_positions ]:
                        cb_enemy_captures = ChessBoard._generate_figure_passives_captures(cb_sim_board_a, cb_enemy_pos)[1]
                        if cb_ally_king_pos in [ x[1] for x in cb_enemy_captures ]:
                            cb_is_ally_king_under_threat = True
                            break
                    if not cb_is_ally_king_under_threat:
                        safe_b_captures.add((cb_to_piece_id, cb_to_pos))
            safe_captures[1][(cb_from_piece_id, cb_from_pos)] = safe_b_captures

        # Revert all sets back to lists
        white_passives = {key: [*value] for (key, value) in safe_passives[0].items() if value}
        white_captures = {key: [*value] for (key, value) in safe_captures[0].items() if value}
        black_passives = {key: [*value] for (key, value) in safe_passives[1].items() if value}
        black_captures = {key: [*value] for (key, value) in safe_captures[1].items() if value}

        
        # Return the appropriate tuple in regard to colour
        if colour:
            if colour == Colour.WHITE:
                return (white_passives, white_captures)
            else:
                return (black_passives, black_captures)
        else:
            return ((white_passives, white_captures), (black_passives, black_captures))

    @staticmethod
    def _generate_figure_passives_captures(board_a, figure_pos):
        figure, colour = FIGURE_ID_MAP[board_a[figure_pos]]
        fig_id = figure.id * colour
        plane, file, rank = figure_pos

        passives = []
        captures = []


        #iterate over passive moves
        for passive in figure.passives:
            x = 1
            pv, fv, rv, next_x = passive(x, colour)
            while ChessBoard.in_bounds(board_a.shape[0]-1, plane+pv, file+fv, rank+rv) and board_a[plane+pv, file+fv, rank+rv] == 0:
                passives.append((fig_id, (plane+pv, file+fv, rank+rv)))
                if not next_x:
                    break
                x = next_x
                pv, fv, rv, next_x = passive(x, colour)

        # iterate over capture moves
        for capture in figure.captures:
            x = 1
            pv, fv, rv, next_x = capture(x, colour)
            while ChessBoard.in_bounds(board_a.shape[0]-1, plane+pv, file+fv, rank+rv):
                if board_a[plane+pv, file+fv, rank+rv] != 0:
                    if FIGURE_ID_MAP[board_a[plane+pv, file+fv, rank+rv]][1] == -colour:
                        captures.append((fig_id, (plane+pv, file+fv, rank+rv)))
                        break
                if not next_x:
                    break
                x = next_x
                pv, fv, rv, next_x = capture(x, colour)

        # iterate over passive or capture moves
        for move in figure.passive_or_capture:
            x = 1
            pv, fv, rv, next_x = move(x, colour)
            while ChessBoard.in_bounds(board_a.shape[0]-1, plane+pv, file+fv, rank+rv):
                if board_a[plane+pv, file+fv, rank+rv] == 0:
                    passives.append((fig_id, (plane+pv, file+fv, rank+rv)))
                else:
                    if FIGURE_ID_MAP[board_a[plane+pv, file+fv, rank+rv]][1] == -colour:
                        captures.append((fig_id, (plane+pv, file+fv, rank+rv)))
                    break
                if not next_x:
                    break
                x = next_x
                pv, fv, rv, next_x = move(x, colour)

        # # Handle pawn promotions
        if figure.id == Pawn.id:
            promotion_plane = board_a.shape[0]-1 if colour == Colour.WHITE else 0
            promotion_file  = board_a.shape[1]-1 if colour == Colour.WHITE else 0

            moves = [passives, captures]
            pawn_moves = [[], []]

            for i in range(len(pawn_moves)):
                for move in moves[i]:
                    f_id, (p, f, r) = move
                    if p == promotion_plane and f == promotion_file: # If this move promotes the pawn, create a move for every figure that the pawn can promote to
                        for fig in PROMOTABLE_FIGURES:
                            pawn_moves[i].append((fig.id*colour, (p, f, r)))
                    else:
                        pawn_moves[i].append((f_id, (p, f, r)))
            
            return (pawn_moves[0], pawn_moves[1])
        else:
            return (passives, captures)

        

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
    def __init__(self, cube, colour, passives, captures, no_progress_count, state_repetition):
        self.cb = cube
        self.colour = colour
        self.passives = passives
        self.captures = captures
        self.no_progress_count = no_progress_count
        self.state_repetition = state_repetition