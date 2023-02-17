import numpy as np
from raumschach.figures import FIGURE_ID_MAP, FIGURE_NAME_MAP, PROMOTABLE_FIGURES, Colour, King, Pawn

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

    @staticmethod
    def init(board_size: int, setup: list[str]) -> np.ndarray:
        board_a = np.zeros((board_size, board_size, board_size), dtype=np.byte, order='C')
        for string in setup:
            fig_name, pos_code = string.split(" ")
            figure, colour = FIGURE_NAME_MAP[fig_name]
            pos = ChessBoard.get_pos_coord(pos_code)
            board_a[pos] = figure.id * colour
        return board_a

    @staticmethod
    def move(board_a: np.ndarray, action: np.ndarray) -> np.ndarray:
        move_board_a = board_a.copy()
        if move_board_a[tuple(action[2:5])] == 0:
            raise Exception(f"Cannot move piece at {ChessBoard.get_pos_code(tuple(action[2:5]))} since this position is empty - Move {action}")
        if move_board_a[tuple(action[2:5])] != action[0]:
            raise Exception(f"Internal state error for piece at {ChessBoard.get_pos_code(tuple(action[2:5]))} - actual piece does not match expeceted piece - Move {action}")
        move_board_a[tuple(action[5:8])] = action[1]
        move_board_a[tuple(action[2:5])] = 0
        return move_board_a

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
                target_pos_id = board_a[plane+pv, file+fv, rank+rv]
                if target_pos_id != 0:
                    if FIGURE_ID_MAP[target_pos_id][1] == -colour:
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
                target_pos_id = board_a[plane+pv, file+fv, rank+rv]
                if target_pos_id == 0:
                    passives.append((fig_id, fig_id, plane, file, rank, plane+pv, file+fv, rank+rv))
                else:
                    if FIGURE_ID_MAP[target_pos_id][1] == -colour:
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
            
            passives_a = np.array(pawn_moves[0])
            captures_a = np.array(pawn_moves[1])
            return (np.empty(shape=(0,8), dtype=np.int32) if passives_a.shape[0] == 0 else passives_a, np.empty(shape=(0,8), dtype=np.int32) if captures_a.shape[0] == 0 else captures_a)
        else:
            passives_a = np.array(passives)
            captures_a = np.array(captures)
            return (np.empty(shape=(0,8), dtype=np.int32) if passives_a.shape[0] == 0 else passives_a, np.empty(shape=(0,8), dtype=np.int32) if captures_a.shape[0] == 0 else captures_a)

    @staticmethod
    def get_passives_captures(board_a, colour=None, simulate_safe_moves=True) -> tuple[np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        white_positions = np.asarray((board_a > 0).nonzero()).T
        black_positions = np.asarray((board_a < 0).nonzero()).T

        white_king_position = ChessBoard.get_king_coords(board_a, Colour.WHITE)
        black_king_position = ChessBoard.get_king_coords(board_a, Colour.BLACK)

        board_positions = [white_positions, black_positions]

        passives = [[], []]
        captures = [[], []]

        # Generate passives and captures for both colours
        for i in range(2):
            for piece_pos in [ (p[0], p[1], p[2]) for p in board_positions[i] ]:
                piece_passives, piece_captures = ChessBoard._generate_figure_passives_captures(board_a, piece_pos)
                passives[i].append(piece_passives)
                captures[i].append(piece_captures)

        for i in range(2):
            passives[i] = np.concatenate(passives[i], axis=0)
            captures[i] = np.concatenate(captures[i], axis=0)

        # # Simulate each possible move and delete those that directly put the ally king in check (since those are illegal moves)
        if simulate_safe_moves:
            if colour == None or colour == Colour.WHITE:
                passives[0] = ChessBoard.get_safe_moves(board_a, passives[0], Colour.WHITE, white_king_position, black_king_position)
                captures[0] = ChessBoard.get_safe_moves(board_a, captures[0], Colour.WHITE, white_king_position, black_king_position)
            if colour == None or colour == Colour.BLACK:
                passives[1] = ChessBoard.get_safe_moves(board_a, passives[1], Colour.BLACK, black_king_position, white_king_position)
                captures[1] = ChessBoard.get_safe_moves(board_a, captures[1], Colour.BLACK, black_king_position, white_king_position)        

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
    def get_safe_moves(board_a, unsafe_moves, colour, ally_king_pos=None, enemy_king_pos=None):
        if ally_king_pos == None or enemy_king_pos == None:
            ally_king_pos, enemy_king_pos = ChessBoard.get_ally_enemy_king_pos(board_a, colour)

        is_safe_move = np.zeros(unsafe_moves.shape[0], dtype=np.bool8)
        for i in range(unsafe_moves.shape[0]):
            move = unsafe_moves[i]
            if ChessBoard.is_safe_move(board_a, move, colour, ally_king_pos, enemy_king_pos):
                is_safe_move[i] = 1

        return unsafe_moves[is_safe_move]


    def get_safe_moves_simulated(board_a: np.ndarray, unsafe_moves: np.ndarray, colour, ally_king_pos=None, enemy_king_pos=None):
        if ally_king_pos == None or enemy_king_pos == None:
            ally_king_pos, enemy_king_pos = ChessBoard.get_ally_enemy_king_pos(board_a, colour)

        simulated_moves = {}
        safe_moves_bool_a = np.zeros(unsafe_moves.shape[0], dtype=np.bool8)

        for i in range(unsafe_moves.shape[0]):
            move = unsafe_moves[i]
            is_safe_move, sim_moves = ChessBoard.is_safe_move_simulated(board_a, move, colour, ally_king_pos, enemy_king_pos)
            if is_safe_move:
                safe_moves_bool_a[i] = 1
                simulated_moves[tuple(move)] = sim_moves

        safe_moves = unsafe_moves[safe_moves_bool_a]
        return (safe_moves, simulated_moves)

    
    @staticmethod
    def is_safe_move(board_a: np.ndarray, unsafe_move: np.ndarray, colour, ally_king_pos=None, enemy_king_pos=None):
        if ally_king_pos == None or enemy_king_pos == None:
            ally_king_pos, enemy_king_pos = ChessBoard.get_ally_enemy_king_pos(board_a, colour)
        if tuple(unsafe_move[5:8]) == enemy_king_pos:
            return True
        else:
            sim_board_a = ChessBoard.move(board_a, unsafe_move)
            sim_ally_king_pos = tuple(unsafe_move[5:8]) if unsafe_move[0] == King.id*colour else ally_king_pos
            if not ChessBoard.is_king_under_check(sim_board_a, colour, sim_ally_king_pos):
                return True
            else:
                return False



    @staticmethod
    def is_safe_move_simulated(board_a: np.ndarray, unsafe_move: np.ndarray, colour, ally_king_pos=None, enemy_king_pos=None):
        if ally_king_pos == None or enemy_king_pos == None:
            ally_king_pos, enemy_king_pos = ChessBoard.get_ally_enemy_king_pos(board_a, colour)

        tuple_unsafe_move = tuple(unsafe_move[5:8])

        if tuple_unsafe_move == enemy_king_pos:
            return (True, np.empty(shape=(0,8), dtype=np.int32))
        else:
            sim_board_a = ChessBoard.move(board_a, unsafe_move)
            sim_ally_king_pos = tuple_unsafe_move if unsafe_move[0] == King.id*colour else ally_king_pos
            sim_moves = []
            enemy_positions = np.asarray((sim_board_a < 0).nonzero()).T if colour == Colour.WHITE else np.asarray((sim_board_a > 0).nonzero()).T
            for enemy_pos in [ (p[0], p[1], p[2]) for p in enemy_positions ]:
                sim_enemy_passives, sim_enemy_captures = ChessBoard._generate_figure_passives_captures(sim_board_a, enemy_pos)
                sim_moves.append(sim_enemy_passives)
                sim_moves.append(sim_enemy_captures)
                if sim_enemy_captures.shape[0] > 0 and np.any(np.all(sim_ally_king_pos == sim_enemy_captures[:, 5:8], axis=1)):
                        return (False, np.empty(shape=(0,8), dtype=np.int32))
            return (True, np.concatenate(sim_moves, axis=0))


    @staticmethod
    def is_king_under_check(board_a, colour, king_coords=None):
        if king_coords == None:
            king_coords = ChessBoard.get_king_coords(board_a, colour)
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
                        elif board_a[abs_p, abs_f, abs_r] != 0:
                            break
                        if not next_x:
                            break
                        x = next_x
                        pv, fv, rv, next_x = move(x, colour)
                        abs_p, abs_f, abs_r = plane+pv, file+fv, rank+rv

    @staticmethod
    def get_ally_enemy_king_pos(board_a, colour):
        return (ChessBoard.get_king_coords(board_a, colour), ChessBoard.get_king_coords(board_a, colour*-1))

    @staticmethod
    def get_king_coords(board_a, colour):
        if colour == Colour.WHITE:
            return (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(board_a==(King.id * Colour.WHITE)))
        else:
            return (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(board_a==(King.id * Colour.BLACK)))

    @staticmethod
    def is_king_checkmate(board_a, colour):
        return not ((King.id*colour) in board_a)
        

    @staticmethod
    def in_bounds(bound_size, x, y, z):
        return (0 <= x <= bound_size) and (0 <= y <= bound_size) and (0 <= z <= bound_size)

    @staticmethod
    def get_piece_passives_captures(passives, captures, piece_pos):
        piece_passives = np.array([])
        piece_captures = np.array([])
        passives_piece_bool_a = piece_pos == passives[:, 2:5]
        captures_piece_bool_a = piece_pos == captures[:, 2:5]
        if passives_piece_bool_a.shape[0] > 0:
            piece_passives = passives[np.all(passives_piece_bool_a, axis=1)]
        if captures_piece_bool_a.shape[0] > 0:
            piece_captures = captures[np.all(captures_piece_bool_a, axis=1)]
        return piece_passives, piece_captures

    @staticmethod
    def get_move_from_passives_captures(passives, captures, from_coord, to_coord):
        from_coord = np.array(from_coord)
        to_coord = np.array(to_coord)
        piece_passives, piece_captures = ChessBoard.get_piece_passives_captures(passives, captures, from_coord)
        if piece_passives.shape[0] == 0 and piece_captures.shape[0] == 0:
            return None
        elif to_coord not in piece_passives[:, 5:8] and to_coord not in piece_captures[:, 5:8]:
            return None
        else:
            if to_coord in piece_passives[:, 5:8]:
                return piece_passives[np.all(to_coord == piece_passives[:, 5:8], axis=1)][0]
            else:
                return piece_captures[np.all(to_coord == piece_captures[:, 5:8], axis=1)][0]


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
    def record_move(prev_board_a, board_a, move, is_checked, is_checkmate):
        from_figure_id, to_figure_id, from_coord, to_coord = move[0], move[1], tuple(move[2:5]), tuple(move[5:8])
        from_figure, from_colour = FIGURE_ID_MAP[from_figure_id]
        move_sign = "-" if prev_board_a[to_coord] == 0 else "x"
        s = f"{(from_figure.name[1+from_colour])}:{ChessBoard.get_pos_code(from_coord)}{move_sign}{ChessBoard.get_pos_code(to_coord)}"
        if from_figure_id != to_figure_id:
            s += f"={FIGURE_ID_MAP[to_figure_id][0].name[1+from_colour]}"
        if all(is_checked):
            s += "++"
        elif is_checked[0]:
            s += "+"
        elif is_checked[1]:
            s += "+"
        elif all(is_checkmate):
            s += " ½-½"
        elif any(is_checkmate):
            s += "#"
            if is_checkmate[0]:
                s += " 0-1"
            else:
                s += " 1-0"
        return s

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