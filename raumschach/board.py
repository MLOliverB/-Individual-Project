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
        self.relative_figure_moves = self._init_relative_figure_moves()
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

    def _init_relative_figure_moves(self):
        relative_figure_moves = {}
        colours = [Colour.WHITE, Colour.BLACK]
        for c_i in range(len(colours)):
            for figure in FIGURES:
                relative_figure_moves[figure.id*colours[c_i]] = {}
                passives = []
                captures = []
                passives_or_captures = []

                for p_rel_pos in figure.single_passives:
                    passives.append([p_rel_pos])
                for p_rel_vec in figure.successive_passives:
                    p_vec = []
                    for j in range(1, 2+self.size):
                        p_vec.append([ v*j for v in p_rel_vec ])
                    passives.append(p_vec)
                
                for c_rel_pos in figure.single_captures:
                    captures.append([c_rel_pos])
                for c_rel_vec in figure.successive_captures:
                    c_vec = []
                    for j in range(1, 2+self.size):
                        c_vec.append([ v*j for v in c_rel_vec ])
                    captures.append(c_vec)

                for pc_rel_pos in figure.single_passives_or_captures:
                    passives_or_captures.append([pc_rel_pos])
                for pc_rel_vec in figure.successive_passives_or_captures:
                    poc_vec = []
                    for j in range(1, 2+self.size):
                        poc_vec.append([ v*j for v in pc_rel_vec ])
                    passives_or_captures.append(poc_vec)

                relative_figure_moves[figure.id*colours[c_i]]["pas"] = colours[c_i] * np.array(passives)
                relative_figure_moves[figure.id*colours[c_i]]["cap"] = colours[c_i] * np.array(captures)
                relative_figure_moves[figure.id*colours[c_i]]["poc"] = colours[c_i] * np.array(passives_or_captures)

        return relative_figure_moves


    def _generate_figure_passives_captures(self, figure_pos):
        figure, colour = FIGURE_ID_MAP[self.cube[figure_pos]]
        fig_id = figure.id * colour
        plane, file, rank = figure_pos
        a_figure_pos = np.array(figure_pos)

        rel_passives = self.relative_figure_moves[fig_id]["pas"]
        rel_captures = self.relative_figure_moves[fig_id]["cap"]
        rel_passives_or_captures = self.relative_figure_moves[fig_id]["poc"]

        passives = np.array([[0, 0, 0], ], dtype=np.int8)
        captures = np.array([[0, 0, 0], ], dtype=np.int8)

        abs_passives = np.array(rel_passives)
        if abs_passives.shape[0] > 0:
            abs_passives = a_figure_pos + abs_passives
            bool_outside_board_passives = np.repeat(~np.all(np.logical_and(abs_passives >= 0, abs_passives < self.size), axis=2, keepdims=True), 3, axis=2)
            abs_passives[bool_outside_board_passives] = 0
            for split in np.split(abs_passives,abs_passives.shape[0], axis=0):
                if not np.all(split==0):
                    arr = split.squeeze(axis=0)
                    arr = arr[~np.all(arr == 0, axis=1)]
                    move_ids = self.cube[arr[:,0], arr[:,1], arr[:,2]]
                    cumsum = np.cumsum(move_ids)
                    passive_moves = arr[cumsum == 0]
                    passives = np.concatenate((passives, passive_moves), axis=0)

        abs_captures = np.array(rel_captures)
        if abs_captures.shape[0] > 0:
            abs_captures = a_figure_pos + abs_captures
            bool_outside_board_captures = np.repeat(~np.all(np.logical_and(abs_captures >= 0, abs_captures < self.size), axis=2, keepdims=True), 3, axis=2)
            abs_captures[bool_outside_board_captures] = 0
            for split in np.split(abs_captures, abs_captures.shape[0], axis=0):
                if not np.all(split==0):
                    arr = split.squeeze(axis=0)
                    arr = arr[~np.all(arr == 0, axis=1)]
                    move_ids = self.cube[arr[:,0], arr[:,1], arr[:,2]]
                    cumprod = np.cumprod(move_ids)
                    if (cumprod[0] < 0 if colour == Colour.WHITE else cumprod[0] > 0):
                        capture_moves = arr[:1]
                        captures = np.concatenate((captures, capture_moves), axis=0)

        abs_poc = np.array(rel_passives_or_captures)
        if abs_poc.shape[0] > 0:
            abs_poc = a_figure_pos + abs_poc
            bool_outside_board_poc = np.repeat(~np.all(np.logical_and(abs_poc >= 0, abs_poc < self.size), axis=2, keepdims=True), 3, axis=2)
            abs_poc[bool_outside_board_poc] = 0
            for split in np.split(abs_poc, abs_poc.shape[0], axis=0):
                if not np.all(split==0):
                    arr = split.squeeze(axis=0)
                    arr = arr[~np.all(arr == 0, axis=1)]
                    move_ids = self.cube[arr[:,0], arr[:,1], arr[:,2]]
                    cumsum = np.cumsum(move_ids)
                    passive_moves = arr[cumsum == 0]
                    passives = np.concatenate((passives, passive_moves), axis=0)
                    nonzero_cumsum = cumsum != 0
                    if arr[nonzero_cumsum].shape[0] > 0 and (move_ids[nonzero_cumsum][0] < 0 if colour == Colour.WHITE else move_ids[nonzero_cumsum][0] > 0):
                        capture_moves = arr[nonzero_cumsum][:1]
                        captures = np.concatenate((captures, capture_moves), axis=0)

        passives = None if passives.shape[0] == 1 else passives[1:]
        captures = None if captures.shape[0] == 1 else captures[1:]

        passive_actions = None
        capture_actions = None

        if figure.id == Pawn.id: # Handle pawn promotions
            promotion_plane = self.cube.shape[0]-1 if colour == Colour.WHITE else 0
            promotion_file  = self.cube.shape[1]-1 if colour == Colour.WHITE else 0
            promotion_ids = np.array([ figure.id*colour for figure in PROMOTABLE_FIGURES ])
            num_promotion_ids = promotion_ids.shape[0]

            if type(passives) != type(None):
                promotable_positions = (passives == promotion_plane)[:, 0] & (passives == promotion_file)[:, 1]
                promotable_passives = passives[promotable_positions]
                nonpromotable_passives = passives[~promotable_positions]
                num_promotable_passives = promotable_passives.shape[0]
                num_nonpromotable_passives = nonpromotable_passives.shape[0]

                passive_actions = np.full(((num_promotable_passives*num_promotion_ids)+num_nonpromotable_passives, 8), fig_id, dtype=np.int8)
                passive_actions[:, 2:5] = a_figure_pos
                passive_actions[: num_nonpromotable_passives, 5:8] = nonpromotable_passives
                if num_promotable_passives > 0:
                    passive_actions[num_nonpromotable_passives: , 1] = np.tile(promotion_ids, (1, num_promotable_passives))
                    passive_actions[num_nonpromotable_passives: , 5:8] = np.repeat(promotable_passives, promotion_ids.shape[0], axis=0)

            if type(captures) != type(None):
                promotable_positions = (captures == promotion_plane)[:, 0] & (captures == promotion_file)[:, 1]
                promotable_captures = captures[promotable_positions]
                nonpromotable_captures = captures[~promotable_positions]
                num_promotable_captures = promotable_captures.shape[0]
                num_nonpromotable_captures = nonpromotable_captures.shape[0]

                capture_actions = np.full(((num_promotable_captures*num_promotion_ids)+num_nonpromotable_captures, 8), fig_id, dtype=np.int8)
                capture_actions[:, 2:5] = a_figure_pos
                capture_actions[: num_nonpromotable_captures, 5:8] = nonpromotable_captures
                if num_promotable_captures > 0:
                    capture_actions[num_nonpromotable_captures: , 1] = np.tile(promotion_ids, (1, num_promotable_captures))
                    capture_actions[num_nonpromotable_captures: , 5:8] = np.repeat(promotable_captures, promotion_ids.shape[0], axis=0)

        else:
            if type(passives) != type(None):
                passive_actions = np.full((passives.shape[0], 8), fig_id, dtype=np.int8)
                passive_actions[:, 2:5] = a_figure_pos
                passive_actions[:, 5:8] = passives
            if type(captures) != type(None):
                capture_actions = np.full((captures.shape[0], 8), fig_id, dtype=np.int8)
                capture_actions[:, 2:5] = a_figure_pos
                capture_actions[:, 5:8] = captures


        # Actions have the form [from_piece_id, to_piece_id, from_plane, from_file, from_rank, to_plane, to_file, to_rank]
        # from_piece_id and to_piece_id will only ever be different if the piece "changes" during the action (i.e. pawn promotion)
        return (passive_actions, capture_actions)


    def get_moves(self, colour=None):
        board_positions = [np.asarray((self.cube > 0).nonzero()).T, np.asarray((self.cube < 0).nonzero()).T]

        passives = []
        captures = []

        for i in range(len(board_positions)):
            passive_moves = []
            capture_moves = []
            for piece_pos in [ (p[0], p[1], p[2]) for p in board_positions[i] ]:
                piece_passives, piece_captures = self._generate_figure_passives_captures(piece_pos)
                if type(piece_passives) != type(None):
                    passive_moves.append(piece_passives)
                if type(piece_captures) != type(None):
                    capture_moves.append(piece_captures)
            passives.append(np.concatenate(passive_moves, axis=0) if passive_moves else None)
            captures.append(np.concatenate(capture_moves, axis=0) if capture_moves else None)

        moves = passives + captures
        colours = [Colour.WHITE, Colour.BLACK, Colour.WHITE, Colour.BLACK]
        # TODO Move Simulation
        for i in range(len(moves)):
            if type(moves[i]) != type(None):
                board_stack = ChessBoard.move(self.cube, moves[i])
                safe_moves = []
                for j in range(board_stack.shape[0]):
                    if not np.any(board_stack[j] == King.id*(-1*colours[i])): # If there is no enemy king
                        safe_moves.append(moves[i][j])
                    else:
                        ally_king_pos = (lambda x: np.array((x[0][0], x[1][0], x[2][0]))) (np.where(board_stack[j] == King.id*(1*colours[i])))
                        enemy_positions = np.asarray((board_stack[j] < 0).nonzero()).T if colours[i] == Colour.WHITE else np.asarray((board_stack[j] > 0).nonzero()).T
                        under_threat = False
                        for enemy_pos in [ (p[0], p[1], p[2]) for p in enemy_positions ]:
                            enemy_captures = self._generate_figure_passives_captures(enemy_pos)[1]
                            if type(enemy_captures) != type(None):
                                if np.any(np.all(enemy_captures[:, 5:8] == ally_king_pos, axis=1)):
                                    under_threat = True
                                    break
                        if not under_threat:
                            safe_moves.append(moves[i][j])
                moves[i] = np.array(safe_moves)

        if colour:
            if colour == Colour.WHITE:
                return (passives[0], captures[0])
            elif colour == Colour.BLACK:
                return (passives[1], captures[1])
        else:
            return ((passives[0], captures[0]), (passives[1], captures[1]))
        


#         # Simulate each possible move and delete those that directly put the ally king in check (since those are illegal moves)

#         for (pw_from_piece_id, pw_from_pos) in passives[0]: # White passives
#             safe_w_passives = set()
#             for (pw_to_piece_id, pw_to_pos) in passives[0][(pw_from_piece_id, pw_from_pos)]:
#                 pw_ally_king_pos = white_king_position
#                 pw_enemy_king_pos = black_king_position
#                 if pw_to_pos == pw_enemy_king_pos:
#                     safe_w_passives.add((pw_to_piece_id, pw_to_pos))
#                 else:
#                     pw_sim_board_a = np.array(board_a)
#                     ChessBoard.move(pw_sim_board_a, (pw_from_piece_id, pw_from_pos), (pw_to_piece_id, pw_to_pos))
#                     if pw_to_piece_id == King.id * 1: # We are moving the king
#                         pw_ally_king_pos = pw_to_pos
#                     pw_is_ally_king_under_threat = False
#                     pw_enemy_positions = np.asarray((pw_sim_board_a < 0).nonzero()).T
#                     for pw_enemy_pos in [ (p[0], p[1], p[2]) for p in pw_enemy_positions ]:
#                         pw_enemy_captures = ChessBoard._generate_figure_passives_captures(pw_sim_board_a, pw_enemy_pos)[1]
#                         if pw_ally_king_pos in [ x[1] for x in pw_enemy_captures ]:
#                             pw_is_ally_king_under_threat = True
#                             break
#                     if not pw_is_ally_king_under_threat:
#                         safe_w_passives.add((pw_to_piece_id, pw_to_pos))
#             safe_passives[0][(pw_from_piece_id, pw_from_pos)] = safe_w_passives

#         for (cw_from_piece_id, cw_from_pos) in captures[0]: # White captures
#             safe_w_captures = set()
#             for (cw_to_piece_id, cw_to_pos) in captures[0][(cw_from_piece_id, cw_from_pos)]:
#                 cw_ally_king_pos = white_king_position
#                 cw_enemy_king_pos = black_king_position
#                 if cw_to_pos == cw_enemy_king_pos:
#                     safe_w_captures.add((cw_to_piece_id, cw_to_pos))
#                 else:
#                     cw_sim_board_a = np.array(board_a)
#                     ChessBoard.move(cw_sim_board_a, (cw_from_piece_id, cw_from_pos), (cw_to_piece_id, cw_to_pos))
#                     if cw_to_piece_id == King.id * 1: # We are moving the king
#                         cw_ally_king_pos = cw_to_pos
#                     cw_is_ally_king_under_threat = False
#                     cw_enemy_positions = np.asarray((cw_sim_board_a < 0).nonzero()).T
#                     for cw_enemy_pos in [ (p[0], p[1], p[2]) for p in cw_enemy_positions ]:
#                         cw_enemy_captures = ChessBoard._generate_figure_passives_captures(cw_sim_board_a, cw_enemy_pos)[1]
#                         if cw_ally_king_pos in [ x[1] for x in cw_enemy_captures ]:
#                             cw_is_ally_king_under_threat = True
#                             break
#                     if not cw_is_ally_king_under_threat:
#                         safe_w_captures.add((cw_to_piece_id, cw_to_pos))
#             safe_captures[0][(cw_from_piece_id, cw_from_pos)] = safe_w_captures

#         for (pb_from_piece_id, pb_from_pos) in passives[1]: # Black passives
#             safe_b_passives = set()
#             for (pb_to_piece_id, pb_to_pos) in passives[1][(pb_from_piece_id, pb_from_pos)]:
#                 pb_ally_king_pos = black_king_position
#                 pb_enemy_king_pos = white_king_position
#                 if pb_to_pos == pb_enemy_king_pos:
#                     safe_b_passives.add((pb_to_piece_id, pb_to_pos))
#                 else:
#                     pb_sim_board_a = np.array(board_a)
#                     ChessBoard.move(pb_sim_board_a, (pb_from_piece_id, pb_from_pos), (pb_to_piece_id, pb_to_pos))
#                     if pb_to_piece_id == King.id * -1: # We are moving the king
#                         pb_ally_king_pos = pb_to_pos
#                     pb_is_ally_king_under_threat = False
#                     pb_enemy_positions = np.asarray((pb_sim_board_a > 0).nonzero()).T
#                     for pb_enemy_pos in [ (p[0], p[1], p[2]) for p in pb_enemy_positions ]:
#                         pb_enemy_captures = ChessBoard._generate_figure_passives_captures(pb_sim_board_a, pb_enemy_pos)[1]
#                         if pb_ally_king_pos in [ x[1] for x in pb_enemy_captures ]:
#                             pb_is_ally_king_under_threat = True
#                             break
#                     if not pb_is_ally_king_under_threat:
#                         safe_b_passives.add((pb_to_piece_id, pb_to_pos))
#             safe_passives[1][(pb_from_piece_id, pb_from_pos)] = safe_b_passives

#         for (cb_from_piece_id, cb_from_pos) in captures[1]: # Black captures
#             safe_b_captures = set()
#             for (cb_to_piece_id, cb_to_pos) in captures[1][(cb_from_piece_id, cb_from_pos)]:
#                 cb_ally_king_pos = black_king_position
#                 cb_enemy_king_pos = white_king_position
#                 if cb_to_pos == cb_enemy_king_pos:
#                     safe_b_captures.add((cb_to_piece_id, cb_to_pos))
#                 else:
#                     cb_sim_board_a = np.array(board_a)
#                     ChessBoard.move(cb_sim_board_a, (cb_from_piece_id, cb_from_pos), (cb_to_piece_id, cb_to_pos))
#                     if cb_to_piece_id == King.id * -1: # We are moving the king
#                         cb_ally_king_pos = cb_to_pos
#                     cb_is_ally_king_under_threat = False
#                     cb_enemy_positions = np.asarray((cb_sim_board_a > 0).nonzero()).T
#                     for cb_enemy_pos in [ (p[0], p[1], p[2]) for p in cb_enemy_positions ]:
#                         cb_enemy_captures = ChessBoard._generate_figure_passives_captures(cb_sim_board_a, cb_enemy_pos)[1]
#                         if cb_ally_king_pos in [ x[1] for x in cb_enemy_captures ]:
#                             cb_is_ally_king_under_threat = True
#                             break
#                     if not cb_is_ally_king_under_threat:
#                         safe_b_captures.add((cb_to_piece_id, cb_to_pos))
#             safe_captures[1][(cb_from_piece_id, cb_from_pos)] = safe_b_captures


    @staticmethod
    def move(board_a, action):
        if action.shape == (8,):
            # This is a single action, modify the given board in-place
            from_id, to_id, from_coord, to_coord = action[0], action[1], tuple(action[2:5]), tuple(action[5:8])
            if board_a[from_coord] == 0:
                raise Exception(f"Cannot move piece at {ChessBoard.get_pos_code(from_coord)} since this position is empty - {action}")
            if board_a[from_coord] != from_id:
                raise Exception(f"Internal state error for piece at {ChessBoard.get_pos_code(from_coord)} - actual piece does not match expeceted piece - {action}")
            board_a[to_coord] = to_id
            board_a[from_coord] = 0
        else:
            # This is an array of action, return an array of modified boards
            cpy_board_a = np.array(board_a)
            board_stack = np.stack(action.shape[0]*[cpy_board_a], axis=0)
            for i in range(board_stack.shape[0]): # TODO find a way to properly vectorize this operation
                board_stack[i, action[i, 5], action[i, 6], action[i, 7]] = action[i, 1]
                board_stack[i, action[i, 2], action[i, 3], action[i, 4]] = 0
            return board_stack
        

    # @staticmethod
    # def in_bounds(bound_size, x, y, z):
    #     return (0 <= x <= bound_size) and (0 <= y <= bound_size) and (0 <= z <= bound_size)

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