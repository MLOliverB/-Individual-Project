from raumschach.board_state import BoardState
from raumschach.players.player import Player

from raumschach.board import ChessBoard
from raumschach.figures import FIGURE_ID_MAP, Colour, King

import numpy as np

class MiniMaxPlayer(Player):
    def __init__(self, search_depth=1, rand_seed=None):
        super().__init__()
        self.search_depth = search_depth
        self.inf = np.inf
        self.neg_inf = -1*np.inf

        if rand_seed == None:
            rand_seed = np.random.default_rng().integers(-(2**63), high=(2**63 - 1))
        self.seed = rand_seed
        self.rng = np.random.default_rng(np.abs(rand_seed))

        self.vectorized_value_transform = np.vectorize((lambda x: 0 if x == 0 else FIGURE_ID_MAP[x][0].value*(x/np.abs(x))), otypes='f')

    def send_action(self, board_state: BoardState):
        moves = np.concatenate([board_state.passives, board_state.captures], axis=0)
        best_move, best_value = self._recursive_minimax(0, self.search_depth, board_state.board_a, moves, board_state.colour)
        return best_move

    def receive_reward(self, reward_value, move_history):
        return super().receive_reward(reward_value, move_history)

    def _recursive_minimax(self, depth, max_depth, board_a: np.ndarray, unsafe_moves, current_colour):
        if King.id*Colour.WHITE not in board_a: # Black has won - terminal condition
            return (None, self.neg_inf)
        if King.id*Colour.BLACK not in board_a: # White has won - terminal condition
            return (None, self.inf)
        if depth == max_depth: # End of search tree - terminal condition
            value_board = self.vectorized_value_transform(board_a)
            value_board[np.isinf(value_board)] = 0
            return (None, np.sum(value_board))

        safe_moves, next_unsafe_moves = ChessBoard.get_safe_moves_simulated(board_a, unsafe_moves, current_colour)

        if safe_moves.shape[0] == 0: # No moves to compute - terminal conditions
            return (None, 0) # TODO: What should be the right board value for this?

        # recursive case
        move_values = np.full(safe_moves.shape[0], (self.neg_inf if current_colour == Colour.WHITE else self.inf))

        for i in range(safe_moves.shape[0]):
            move = safe_moves[i]
            sim_board_a = ChessBoard.move(board_a, move)
            move_values[i] = self._recursive_minimax(depth+1, max_depth, sim_board_a, next_unsafe_moves[tuple(move)], current_colour*-1)[1]

        best_move_value = None
        best_moves_bool_a = None
        if current_colour == Colour.WHITE:
            best_move_value = np.max(move_values)
            best_moves_bool_a = np.isclose(move_values, best_move_value) #move_values == best_move_value
        else:
            best_move_value = np.min(move_values)
            best_moves_bool_a = np.isclose(move_values, best_move_value) #move_values == best_move_value

        best_moves = safe_moves[best_moves_bool_a]
        single_best_move = best_moves[self.rng.integers(0, best_moves.shape[0])]
        return (single_best_move, best_move_value)


class AlphaBetaPlayer(Player):
    def __init__(self, search_depth=1, rand_seed=None):
        super().__init__()
        self.search_depth = search_depth
        self.inf = np.inf
        self.neg_inf = -1*np.inf

        if rand_seed == None:
            rand_seed = np.random.default_rng().integers(-(2**63), high=(2**63 - 1))
        self.seed = rand_seed
        self.rng = np.random.default_rng(np.abs(rand_seed))

        self.vectorized_value_transform = np.vectorize((lambda x: 0 if x == 0 else FIGURE_ID_MAP[x][0].value*(x/np.abs(x))), otypes='f')

    def send_action(self, board_state: BoardState):
        moves = np.concatenate([board_state.captures, board_state.passives], axis=0)
        best_move, best_value = self._recursive_alphabeta(0, self.search_depth, self.neg_inf, self.inf, board_state.board_a, moves, board_state.colour)
        return best_move

    def receive_reward(self, reward_value, move_history):
        return super().receive_reward(reward_value, move_history)

    def _recursive_alphabeta(self, depth, max_depth, alpha, beta, board_a: np.ndarray, unsafe_moves, current_colour):
        # White is alpha (maximizing)
        # Black is beta (minimizing)
        if King.id*Colour.WHITE not in board_a: # Black has won - terminal condition
            return (None, self.neg_inf)
        if King.id*Colour.BLACK not in board_a: # White has won - terminal condition
            return (None, self.inf)
        if depth == max_depth: # End of search tree - terminal condition
            value_board = self.vectorized_value_transform(board_a)
            value_board[np.isinf(value_board)] = 0
            return (None, np.sum(value_board))

        moves = unsafe_moves

        if moves.shape[0] == 0: # No moves to compute - terminal conditions
            return (None, 0) # TODO: What should be the right board value for this?

        # recursive case
        best_moves, best_value = [], None
        ally_king_pos, enemy_king_pos = ChessBoard.get_ally_enemy_king_pos(board_a, current_colour)

        if current_colour == Colour.WHITE:
            best_value = self.neg_inf
            for i in range(moves.shape[0]):
                is_safe_move, next_unsafe_moves = None, None
                if depth + 1 == max_depth:
                    is_safe_move, next_unsafe_moves = ChessBoard.is_safe_move(board_a, moves[i], current_colour, ally_king_pos, enemy_king_pos), np.empty(shape=(0,8), dtype=np.int32)
                else:
                    is_safe_move, next_unsafe_moves = ChessBoard.is_safe_move_simulated(board_a, moves[i], current_colour, ally_king_pos, enemy_king_pos)
                if is_safe_move:
                    move = moves[i]
                    sim_board_a = ChessBoard.move(board_a, move)
                    value = self._recursive_alphabeta(depth+1, max_depth, alpha, beta, sim_board_a, next_unsafe_moves, current_colour*-1)[1]
                    if value > best_value:
                        best_value = value
                        best_moves = [move]
                    elif value == best_value:
                        best_moves.append(move)
                    if best_value >= beta:
                        break # beta cutoff
                    alpha = max(alpha, best_value)
        else:
            best_value = self.inf
            for i in range(moves.shape[0]):
                is_safe_move, next_unsafe_moves = None, None
                if depth + 1 == max_depth:
                    is_safe_move, next_unsafe_moves = ChessBoard.is_safe_move(board_a, moves[i], current_colour, ally_king_pos, enemy_king_pos), np.empty(shape=(0,8), dtype=np.int32)
                else:
                    is_safe_move, next_unsafe_moves = ChessBoard.is_safe_move_simulated(board_a, moves[i], current_colour, ally_king_pos, enemy_king_pos)
                if is_safe_move:
                    move = moves[i]
                    sim_board_a = ChessBoard.move(board_a, move)
                    value = self._recursive_alphabeta(depth+1, max_depth, alpha, beta, sim_board_a, next_unsafe_moves, current_colour*-1)[1]
                    if value < best_value:
                        best_value = value
                        best_moves = [move]
                    elif value == best_value:
                        best_moves.append(move)
                    if best_value <= alpha:
                        break # alpha cutoff
                    beta = min(beta, best_value)

        if not best_moves:
            return (None, best_value)
        else:
            best_move = best_moves[self.rng.integers(0, len(best_moves))]
            return (best_move, best_value)


class CachedAlphaBetaPlayer(Player):
    def __init__(self, search_depth=1, rand_seed=None):
        super().__init__()
        self.cache = dict()
        self.search_depth = search_depth
        self.inf = np.inf
        self.neg_inf = -1*np.inf

        if rand_seed == None:
            rand_seed = np.random.default_rng().integers(-(2**63), high=(2**63 - 1))
        self.seed = rand_seed
        self.rng = np.random.default_rng(np.abs(rand_seed))

        self.vectorized_value_transform = np.vectorize((lambda x: 0 if x == 0 else FIGURE_ID_MAP[x][0].value*(x/np.abs(x))), otypes='f')

    def send_action(self, board_state: BoardState):
        # moves = np.concatenate([board_state.captures, board_state.passives], axis=0)
        best_move, best_value = self._recursive_alphabeta(0, self.search_depth, self.neg_inf, self.inf, board_state.board_a, board_state.colour)
        return best_move

    def receive_reward(self, reward_value, move_history):
        return super().receive_reward(reward_value, move_history)

    def _recursive_alphabeta(self, depth, max_depth, alpha, beta, board_a: np.ndarray, current_colour):
        # White is alpha (maximizing)
        # Black is beta (minimizing)
        if King.id*Colour.WHITE not in board_a: # Black has won - terminal condition
            return (None, self.neg_inf)
        if King.id*Colour.BLACK not in board_a: # White has won - terminal condition
            return (None, self.inf)
        if depth == max_depth: # End of search tree - terminal condition
            value_board = self.vectorized_value_transform(board_a)
            value_board[np.isinf(value_board)] = 0
            return (None, np.sum(value_board))

        ally_king_pos, enemy_king_pos = ChessBoard.get_ally_enemy_king_pos(board_a, current_colour)

        # safe_moves = None
        # board_hash = board_a.data.tobytes()
        # if board_hash in self.cache:
        #     safe_moves = self.cache[board_hash]
        # else:
        #     unsafe_passives, unsafe_captures = ChessBoard.get_passives_captures(board_a, current_colour, simulate_safe_moves=False)
        #     unsafe_moves = np.concatenate((unsafe_captures, unsafe_passives), axis=0)
        #     safe_moves = ChessBoard.get_safe_moves_generator(board_a, unsafe_moves, current_colour, ally_king_pos, enemy_king_pos)
        #     print(unsafe_moves.shape[0], len(list(safe_moves)))
        #     print(list(unsafe_moves))
        #     print(list(safe_moves))
        #     self.cache[board_hash] = safe_moves
        safe_passives, safe_captures = ChessBoard.get_passives_captures(board_a, current_colour, simulate_safe_moves=True)
        # unsafe_moves = np.concatenate((unsafe_captures, unsafe_passives), axis=0)
        safe_moves = np.concatenate((safe_captures, safe_passives), axis=0)
        # move_list = list(safe_moves)
        # print(unsafe_moves.shape[0], len(move_list))
        # print("unsafe", list(unsafe_moves))
        # print("safe", move_list)

        # if not safe_moves: # No moves to compute - terminal conditions
        #     print("No safe moves")
        #     return (None, 0) # TODO: What should be the right board value for this?

        # recursive case
        best_moves, best_value = [], None

        if current_colour == Colour.WHITE:
            best_value = self.neg_inf
            for move in safe_moves:
                # is_safe_move, next_unsafe_moves = None, None
                # if depth + 1 == max_depth:
                #     is_safe_move, next_unsafe_moves = ChessBoard.is_safe_move(board_a, move, current_colour, ally_king_pos, enemy_king_pos), np.empty(shape=(0,8), dtype=np.int32)
                # else:
                #     is_safe_move, next_unsafe_moves = ChessBoard.is_safe_move_simulated(board_a, move, current_colour, ally_king_pos, enemy_king_pos)
                # if is_safe_move:
                sim_board_a = ChessBoard.move(board_a, move)
                value = self._recursive_alphabeta(depth+1, max_depth, alpha, beta, sim_board_a, current_colour*-1)[1]
                if value > best_value:
                    best_value = value
                    best_moves = [move]
                elif value == best_value:
                    best_moves.append(move)
                if best_value >= beta:
                    break # beta cutoff
                alpha = max(alpha, best_value)
        else:
            best_value = self.inf
            for move in safe_moves:
                # is_safe_move, next_unsafe_moves = None, None
                # if depth + 1 == max_depth:
                #     is_safe_move, next_unsafe_moves = ChessBoard.is_safe_move(board_a, move, current_colour, ally_king_pos, enemy_king_pos), np.empty(shape=(0,8), dtype=np.int32)
                # else:
                #     is_safe_move, next_unsafe_moves = ChessBoard.is_safe_move_simulated(board_a, move, current_colour, ally_king_pos, enemy_king_pos)
                # if is_safe_move:
                sim_board_a = ChessBoard.move(board_a, move)
                value = self._recursive_alphabeta(depth+1, max_depth, alpha, beta, sim_board_a, current_colour*-1)[1]
                if value < best_value:
                    best_value = value
                    best_moves = [move]
                elif value == best_value:
                    best_moves.append(move)
                if best_value <= alpha:
                    break # alpha cutoff
                beta = min(beta, best_value)

        if not best_moves:
            return (None, 0) # TODO: What should be the right board value for this?
            # return (None, best_value)
        else:
            best_move = best_moves[self.rng.integers(0, len(best_moves))]
            return (best_move, best_value)
