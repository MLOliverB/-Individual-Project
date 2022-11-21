import numpy as np
from raumschach.figures import FIGURE_ID_MAP, FIGURE_NAME_MAP, FIGURES, PROMOTABLE_FIGURES, Colour, King, Pawn
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
    def move(board_a, action):
        if board_a[tuple(action[2:5])] == 0:
            raise Exception(f"Cannot move piece at {ChessBoard.get_pos_code(tuple(action[2:5]))} since this position is empty - Move {action}")
        if board_a[tuple(action[2:5])] != action[0]:
            raise Exception(f"Internal state error for piece at {ChessBoard.get_pos_code(tuple(action[2:5]))} - actual piece does not match expeceted piece - Move {action}")
        board_a[tuple(action[5:8])] = action[1]
        board_a[tuple(action[2:5])] = 0

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
                passives.append((fig_id, fig_id, plane, file, rank, plane+pv, file+fv, rank+rv))
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
                        captures.append((fig_id, fig_id, plane, file, rank, plane+pv, file+fv, rank+rv))
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
                    passives.append((fig_id, fig_id, plane, file, rank, plane+pv, file+fv, rank+rv))
                else:
                    if FIGURE_ID_MAP[board_a[plane+pv, file+fv, rank+rv]][1] == -colour:
                        captures.append((fig_id, fig_id, plane, file, rank, plane+pv, file+fv, rank+rv))
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
                    fig_id, _, p_0, f_0, r_0, p_1, f_1, r_1 = move
                    if p_1 == promotion_plane and f_1 == promotion_file: # If this move promotes the pawn, create a move for every figure that the pawn can promote to
                        for fig in PROMOTABLE_FIGURES:
                            pawn_moves[i].append((fig_id, fig.id*colour, p_0, f_0, r_0, p_1, f_1, r_1))
                    else:
                        pawn_moves[i].append((fig_id, fig_id, p_0, f_0, r_0, p_1, f_1, r_1))
            
            return (np.array(pawn_moves[0]), np.array(pawn_moves[1]))
        else:
            return (np.array(passives), np.array(captures))

    @staticmethod
    def get_passives_captures(board_a, colour=None, simulate_safe_moves=True):
        white_positions = np.asarray((board_a > 0).nonzero()).T
        black_positions = np.asarray((board_a < 0).nonzero()).T

        white_king_position = (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(board_a==(King.id * Colour.WHITE)))
        black_king_position = (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(board_a==(King.id * Colour.BLACK)))

        ally_king_position = [white_king_position, black_king_position]
        enemy_king_position = [black_king_position, white_king_position]

        board_positions = [white_positions, black_positions]
        colours = [Colour.WHITE, Colour.BLACK]

        passives = [[], []]
        captures = [[], []]

        # Generate passives and captures for both colours
        for i in range(2):
            for piece_pos in [ (p[0], p[1], p[2]) for p in board_positions[i] ]:
                piece_passives, piece_captures = ChessBoard._generate_figure_passives_captures(board_a, piece_pos)
                if piece_passives.shape[0] > 0:
                    passives[i].append(piece_passives)
                if piece_captures.shape[0] > 0:
                    captures[i].append(piece_captures)

        for i in range(2):
            passives[i] = np.array([]) if len(passives[i]) == 0 else np.concatenate(passives[i], axis=0)
            captures[i] = np.array([]) if len(captures[i]) == 0 else np.concatenate(captures[i], axis=0)

        # # Simulate each possible move and delete those that directly put the ally king in check (since those are illegal moves)
        if simulate_safe_moves:
            if colour == None or colour == Colour.WHITE:
                passives[0] = ChessBoard.get_safe_moves(board_a, passives[0], Colour.WHITE, white_king_position, black_king_position)
                captures[0] = ChessBoard.get_safe_moves(board_a, captures[0], Colour.WHITE, white_king_position, black_king_position)
            if colour == None or colour == Colour.BLACK:
                passives[1] = ChessBoard.get_safe_moves(board_a, passives[1], Colour.BLACK, black_king_position, white_king_position)
                captures[1] = ChessBoard.get_safe_moves(board_a, captures[1], Colour.BLACK, black_king_position, white_king_position)

        # for col_i in (range(2) if colour == None else range(0, 1) if colour == Colour.WHITE else range(1, 2)):
        #     ally_king_pos = ally_king_position[i]
        #     enemy_king_pos = enemy_king_position[i]
        #     print(ally_king_pos, enemy_king_pos)
        #     for from_tuple in passives[col_i]:
        #         safe_passives[col_i][from_tuple] = ChessBoard.simulate_is_safe_moves(board_a, from_tuple, passives[col_i][from_tuple], colours[col_i], ally_king_pos, enemy_king_pos)
        #     for from_tuple in captures[col_i]:
        #         safe_captures[col_i][from_tuple] = ChessBoard.simulate_is_safe_moves(board_a, from_tuple, captures[col_i][from_tuple], colours[col_i], ally_king_pos, enemy_king_pos)

        # print(safe_passives)
        # print(safe_captures)

        # # TODO Encapsulate this in another method to avoid code duplication

        # for (pw_from_piece_id, pw_from_pos) in passives[0]: # White passives
        #     safe_w_passives = set()
        #     for (pw_to_piece_id, pw_to_pos) in passives[0][(pw_from_piece_id, pw_from_pos)]:
        #         pw_ally_king_pos = white_king_position
        #         pw_enemy_king_pos = black_king_position
        #         if pw_to_pos == pw_enemy_king_pos:
        #             safe_w_passives.add((pw_to_piece_id, pw_to_pos))
        #         else:
        #             pw_sim_board_a = np.array(board_a)
        #             ChessBoard.move(pw_sim_board_a, (pw_from_piece_id, pw_from_pos), (pw_to_piece_id, pw_to_pos))
        #             if pw_to_piece_id == King.id * 1: # We are moving the king
        #                 pw_ally_king_pos = pw_to_pos
        #             pw_is_ally_king_under_threat = False
        #             pw_enemy_positions = np.asarray((pw_sim_board_a < 0).nonzero()).T
        #             for pw_enemy_pos in [ (p[0], p[1], p[2]) for p in pw_enemy_positions ]:
        #                 pw_enemy_captures = ChessBoard._generate_figure_passives_captures(pw_sim_board_a, pw_enemy_pos)[1]
        #                 if pw_ally_king_pos in [ x[1] for x in pw_enemy_captures ]:
        #                     pw_is_ally_king_under_threat = True
        #                     break
        #             if not pw_is_ally_king_under_threat:
        #                 safe_w_passives.add((pw_to_piece_id, pw_to_pos))
        #     safe_passives[0][(pw_from_piece_id, pw_from_pos)] = safe_w_passives

        # for (cw_from_piece_id, cw_from_pos) in captures[0]: # White captures
        #     safe_w_captures = set()
        #     for (cw_to_piece_id, cw_to_pos) in captures[0][(cw_from_piece_id, cw_from_pos)]:
        #         cw_ally_king_pos = white_king_position
        #         cw_enemy_king_pos = black_king_position
        #         if cw_to_pos == cw_enemy_king_pos:
        #             safe_w_captures.add((cw_to_piece_id, cw_to_pos))
        #         else:
        #             cw_sim_board_a = np.array(board_a)
        #             ChessBoard.move(cw_sim_board_a, (cw_from_piece_id, cw_from_pos), (cw_to_piece_id, cw_to_pos))
        #             if cw_to_piece_id == King.id * 1: # We are moving the king
        #                 cw_ally_king_pos = cw_to_pos
        #             cw_is_ally_king_under_threat = False
        #             cw_enemy_positions = np.asarray((cw_sim_board_a < 0).nonzero()).T
        #             for cw_enemy_pos in [ (p[0], p[1], p[2]) for p in cw_enemy_positions ]:
        #                 cw_enemy_captures = ChessBoard._generate_figure_passives_captures(cw_sim_board_a, cw_enemy_pos)[1]
        #                 if cw_ally_king_pos in [ x[1] for x in cw_enemy_captures ]:
        #                     cw_is_ally_king_under_threat = True
        #                     break
        #             if not cw_is_ally_king_under_threat:
        #                 safe_w_captures.add((cw_to_piece_id, cw_to_pos))
        #     safe_captures[0][(cw_from_piece_id, cw_from_pos)] = safe_w_captures

        # for (pb_from_piece_id, pb_from_pos) in passives[1]: # Black passives
        #     safe_b_passives = set()
        #     for (pb_to_piece_id, pb_to_pos) in passives[1][(pb_from_piece_id, pb_from_pos)]:
        #         pb_ally_king_pos = black_king_position
        #         pb_enemy_king_pos = white_king_position
        #         if pb_to_pos == pb_enemy_king_pos:
        #             safe_b_passives.add((pb_to_piece_id, pb_to_pos))
        #         else:
        #             pb_sim_board_a = np.array(board_a)
        #             ChessBoard.move(pb_sim_board_a, (pb_from_piece_id, pb_from_pos), (pb_to_piece_id, pb_to_pos))
        #             if pb_to_piece_id == King.id * -1: # We are moving the king
        #                 pb_ally_king_pos = pb_to_pos
        #             pb_is_ally_king_under_threat = False
        #             pb_enemy_positions = np.asarray((pb_sim_board_a > 0).nonzero()).T
        #             for pb_enemy_pos in [ (p[0], p[1], p[2]) for p in pb_enemy_positions ]:
        #                 pb_enemy_captures = ChessBoard._generate_figure_passives_captures(pb_sim_board_a, pb_enemy_pos)[1]
        #                 if pb_ally_king_pos in [ x[1] for x in pb_enemy_captures ]:
        #                     pb_is_ally_king_under_threat = True
        #                     break
        #             if not pb_is_ally_king_under_threat:
        #                 safe_b_passives.add((pb_to_piece_id, pb_to_pos))
        #     safe_passives[1][(pb_from_piece_id, pb_from_pos)] = safe_b_passives

        

        # Split up the different collections
        white_passives = passives[0]
        white_captures = captures[0]
        black_passives = passives[1]
        black_captures = captures[1]

        
        # Return the appropriate tuple in regard to colour
        if colour:
            if colour == Colour.WHITE:
                return (white_passives, white_captures)
            else:
                return (black_passives, black_captures)
        else:
            return ((white_passives, white_captures), (black_passives, black_captures))

    @staticmethod
    def get_safe_moves(board_a, moves, colour, ally_king_pos=None, enemy_king_pos=None):
        if ally_king_pos == None or enemy_king_pos == None:
            white_king_position = (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(board_a==(King.id * Colour.WHITE)))
            black_king_position = (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(board_a==(King.id * Colour.BLACK)))
            if colour == Colour.WHITE:
                ally_king_pos, enemy_king_pos = white_king_position, black_king_position
            else:
                ally_king_pos, enemy_king_pos = black_king_position, white_king_position

        is_safe_move = np.zeros(moves.shape[0], dtype=np.bool8)
        for i in range(moves.shape[0]):
            move = moves[i]
            if tuple(move[5:8]) == enemy_king_pos:
                is_safe_move[i] = 1
            else:
                sim_board_a = board_a.copy()
                ChessBoard.move(sim_board_a, move)
                if not ChessBoard.is_king_under_check(sim_board_a, colour, ally_king_pos):
                    is_safe_move[i] = 1

        return moves[is_safe_move]

        # @staticmethod
    # def simulate_safe_moves(board_a, from_tuple, to_moves, colour, ally_king_pos=None, enemy_king_pos=None):
    #     if ally_king_pos == None or enemy_king_pos == None:
    #         white_king_position = (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(board_a==(King.id * Colour.WHITE)))
    #         black_king_position = (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(board_a==(King.id * Colour.BLACK)))
    #         if colour == Colour.WHITE:
    #             ally_king_pos, enemy_king_pos = white_king_position, black_king_position
    #         else:
    #             ally_king_pos, enemy_king_pos = black_king_position, white_king_position
        
    #     safe_moves = set()
    #     for to_tuple in to_moves:
    #         if ChessBoard._simulate_is_safe_move(board_a, from_tuple, to_tuple, colour, ally_king_pos, enemy_king_pos):
    #             safe_moves.add(to_tuple)
    #     return safe_moves

    # @staticmethod
    # def _simulate_is_safe_move(board_a, from_tuple, to_tuple, colour, ally_king_pos, enemy_king_pos):
    #     from_id, from_coords = from_tuple
    #     to_id, to_coords = to_tuple
    #     if enemy_king_pos == to_coords:
    #         return True
    #     else:
    #         sim_board_a = board_a.copy()
    #         ChessBoard.move(sim_board_a, from_tuple, to_tuple)
    #         enemy_colour = Colour.BLACK if colour == Colour.WHITE else Colour.WHITE
    #         sim_ally_king_pos = to_coords if from_id == King.id*colour else ally_king_pos

    #         enemy_figures = [ FIGURE_ID_MAP[id][0] for id in np.unique(sim_board_a[sim_board_a < 0 if colour == Colour.WHITE else sim_board_a > 0])] # Get all unique enemy figures
    #         for enemy_figure in enemy_figures:
    #             if ChessBoard._can_capture_target_emulate(sim_board_a, sim_ally_king_pos, enemy_figure, colour, enemy_figure, enemy_colour):
    #                 return False
    #         return True
    


    @staticmethod
    def get_safe_moves_simulated(board_a, moves, colour, ally_king_pos=None, enemy_king_pos=None):
        if ally_king_pos == None or enemy_king_pos == None:
            white_king_position = (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(board_a==(King.id * Colour.WHITE)))
            black_king_position = (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(board_a==(King.id * Colour.BLACK)))
            if colour == Colour.WHITE:
                ally_king_pos, enemy_king_pos = white_king_position, black_king_position
            else:
                ally_king_pos, enemy_king_pos = black_king_position, white_king_position
        # for (pb_from_piece_id, pb_from_pos) in passives[1]: # Black passives
        #     safe_b_passives = set()
        #     for (pb_to_piece_id, pb_to_pos) in passives[1][(pb_from_piece_id, pb_from_pos)]:
        #         pb_ally_king_pos = black_king_position
        #         pb_enemy_king_pos = white_king_position
        #         if pb_to_pos == pb_enemy_king_pos:
        #             safe_b_passives.add((pb_to_piece_id, pb_to_pos))
        #         else:
        #             pb_sim_board_a = np.array(board_a)
        #             ChessBoard.move(pb_sim_board_a, (pb_from_piece_id, pb_from_pos), (pb_to_piece_id, pb_to_pos))
        #             if pb_to_piece_id == King.id * -1: # We are moving the king
        #                 pb_ally_king_pos = pb_to_pos
        #             pb_is_ally_king_under_threat = False
        #             pb_enemy_positions = np.asarray((pb_sim_board_a > 0).nonzero()).T
        #             for pb_enemy_pos in [ (p[0], p[1], p[2]) for p in pb_enemy_positions ]:
        #                 pb_enemy_captures = ChessBoard._generate_figure_passives_captures(pb_sim_board_a, pb_enemy_pos)[1]
        #                 if pb_ally_king_pos in [ x[1] for x in pb_enemy_captures ]:
        #                     pb_is_ally_king_under_threat = True
        #                     break
        #             if not pb_is_ally_king_under_threat:
        #                 safe_b_passives.add((pb_to_piece_id, pb_to_pos))
        #     safe_passives[1][(pb_from_piece_id, pb_from_pos)] = safe_b_passives


    @staticmethod
    def is_king_under_check(board_a, colour, king_coords=None):
        if king_coords == None:
            king_coords = (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(board_a==(King.id * colour)))
        plane, file, rank = king_coords
        enemy_figures_bool_a = board_a < 0 if colour == Colour.WHITE else board_a > 0
        enemy_figures = [ FIGURE_ID_MAP[id][0] for id in np.unique(board_a[enemy_figures_bool_a])] # Get all unique enemy figures
        for enemy_figure in enemy_figures:
            enemy_figure_id = enemy_figure.id*-1*colour
            for move_set in [enemy_figure.captures, enemy_figure.passive_or_capture]:
                for move in move_set:
                    x = 1
                    pv, fv, rv, next_x = move(x, colour)
                    abs_p, abs_f, abs_r = plane+pv, file+fv, rank+rv
                    while ChessBoard.in_bounds(board_a.shape[0]-1, abs_p, abs_f, abs_r):
                        if board_a[abs_p, abs_f, abs_r] == enemy_figure_id:
                            return True
                        if not next_x:
                            break
                        x = next_x
                        pv, fv, rv, next_x = move(x, colour)
                        abs_p, abs_f, abs_r = plane+pv, file+fv, rank+rv

    @staticmethod
    def is_king_checkmate(board_a, colour):
        return not ((King.id*colour) in board_a)


    # @staticmethod
    # def _can_capture_target_emulate(board_a, orig_figure_pos, emu_figure, emu_colour, target_figure, target_colour):
    #     plane, file, rank = orig_figure_pos
    #     target_figure_id = target_figure.id*target_colour

    #     move_sets = [emu_figure.captures, emu_figure.passive_or_capture]
    #     for move_set in move_sets:
    #         for move in move_set:
    #             x = 1
    #             pv, fv, rv, next_x = move(x, emu_colour)
    #             abs_p, abs_f, abs_r = plane+pv, file+fv, rank+rv
    #             while ChessBoard.in_bounds(board_a.shape[0]-1, abs_p, abs_f, abs_r):
    #                 if board_a[abs_p, abs_f, abs_r] == target_figure_id:
    #                     return True
    #                 if not next_x:
    #                     break
    #                 x = next_x
    #                 pv, fv, rv, next_x = move(x, emu_colour)
    #                 abs_p, abs_f, abs_r = plane+pv, file+fv, rank+rv

    #     return False
        

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

    @staticmethod
    def record_move(board_a, move, is_checked, is_checkmate):
        from_figure_id, to_figure_id, from_coord, to_coord = move[0], move[1], tuple(move[2:5]), tuple(move[5:8])
        from_figure, from_colour = FIGURE_ID_MAP[from_figure_id]
        if board_a[to_coord] == 0:
            to_figure, to_colour = None, None
        else:
            to_figure, to_colour = FIGURE_ID_MAP[to_figure_id]
        move_sign = "-" if to_figure == None else "x"
        s = f"{(from_figure.name[1+from_colour])}:{ChessBoard.get_pos_code(from_coord)}{move_sign}{ChessBoard.get_pos_code(to_coord)}"
        if from_figure_id != to_figure_id:
            s += f"={FIGURE_ID_MAP[to_figure_id][0].name[1+from_colour]}"
        if all(is_checked):
            s += "++"
        elif is_checked[0]:
            s += "+w"
        elif is_checked[1]:
            s += "+b"
        elif all(is_checkmate):
            s += " ½-½"
        elif any(is_checkmate):
            s += "#"
            if is_checkmate[0]:
                s += " 0-1"
            else:
                s += " 1-0"
        return s

    # TODO rewrite to return a single move ndarray
    @staticmethod
    def read_recorded_move(record):
        from_figure, from_colour = FIGURE_NAME_MAP[record[0]] 
        from_pos = record[2:5]
        to_pos = record[6:9]
        to_figure, to_colour = from_figure, from_colour
        if '=' in record:
            to_figure, to_colour = FIGURE_NAME_MAP[record[10]]
        from_coord = ChessBoard.get_pos_coord(from_pos)
        to_coord = ChessBoard.get_pos_coord(to_pos)
        from_tuple = (from_figure.id* from_colour, from_coord)
        to_tuple   = (to_figure.id  * to_colour,   to_coord)
        return np.array((from_figure.id* from_colour, to_figure.id  * to_colour, from_coord[0], from_coord[1], from_coord[2], to_coord[0], to_coord[1], to_coord[2]))
        return (from_tuple, to_tuple)


class BoardState():
    def __init__(self, cube, colour, passives, captures, no_progress_count, state_repetition):
        self.cb = cube
        self.colour = colour
        self.passives = passives
        self.captures = captures
        self.no_progress_count = no_progress_count
        self.state_repetition = state_repetition