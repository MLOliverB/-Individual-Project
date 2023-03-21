from typing import Callable
from raumschach.board_state import BoardState, SimpleBoardState
from raumschach.players.player import Player

from raumschach.board import ChessBoard
from raumschach.figures import FIGURE_ID_MAP, Colour, King

import numpy as np

from raumschach.reinforcement_learn.const import DRAW_STATE_VALUE

class MiniMaxPlayer(Player):
    def __init__(self, search_depth=1, rand_seed=None, memory=None):
        super().__init__(memory=memory)
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
        super().step_memory(board_state, best_move)
        return best_move

    def receive_reward(self, reward_value, move_history):
        return super().commit_memory(reward_value)

    def __str__(self) -> str:
        return f"MiniMax Player (Piece-Value-Function, search_depth={self.search_depth})"

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
    def __init__(self, search_depth=1, rand_seed=None, play_to_lose=False, memory=None):
        super().__init__(memory=memory)
        self.search_depth = search_depth
        self.play_to_lose = play_to_lose
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
        super().step_memory(board_state, best_move)
        return best_move

    def receive_reward(self, reward_value, move_history):
        return super().commit_memory(reward_value)

    def __str__(self) -> str:
        return f"AlphaBeta Player (Piece-Value-Function, search_depth={self.search_depth}, play_to_lose={self.play_to_lose})"

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

        if (not self.play_to_lose and current_colour == Colour.WHITE) or (self.play_to_lose and current_colour == Colour.BLACK):
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

class MiniMaxTreeSearchPlayer(Player):
    def __init__(self, search_depth=1, branching_factor=500, random_action_p=0.0, rand_seed=None, play_to_lose=False, memory=None, value_function : Callable[['SimpleBoardState', np.ndarray], np.ndarray] = None, value_function_name=""):
        super().__init__(memory)

        self.value_function_name = value_function_name
        self.play_to_lose = play_to_lose
        self.search_depth = search_depth
        self.inf = np.inf
        self.neg_inf = -1*np.inf

        self.branching_factor = branching_factor
        self.random_selection_chance = random_action_p

        if rand_seed == None:
            rand_seed = np.random.default_rng().integers(-(2**63), high=(2**63 - 1))
        self.seed = rand_seed
        self.rng = np.random.default_rng(np.abs(rand_seed))

        if value_function == None:
            self.value_function = MiniMaxTreeSearchPlayer._get_std_val_func()
        else:
            self.value_function = value_function

    def send_action(self, board_state: BoardState):
        moves = np.concatenate([board_state.passives, board_state.captures], axis=0)

        random_move = self.rng.choice([True, False], p=[self.random_selection_chance, 1-self.random_selection_chance])
        if random_move:
            best_move = moves[self.rng.integers(0, moves.shape[0])]
        else:
            best_move, best_value = self._tree_search(0, self.search_depth, board_state, player_colour=board_state.colour)
        super().step_memory(board_state, best_move)
        return best_move

    def receive_reward(self, reward_value, move_history):
        return super().commit_memory(reward_value)

    def __str__(self) -> str:
        return f"MiniMax TreeSearch Player (Value Function={self.value_function_name}, search_depth={self.search_depth}, branching_factor={self.branching_factor}, random_action_p={self.random_selection_chance}, play_to_lose={self.play_to_lose})"

    def _tree_search(self, depth, max_depth, simple_board_state: 'SimpleBoardState', player_colour):
        if King.id*player_colour not in simple_board_state.board_a: # Player has lost - terminal condition
            return (None, self.neg_inf)
        if King.id*player_colour*-1 not in simple_board_state.board_a: # Player has won - terminal condition
            return (None, self.inf)
        
        passives, captures = ChessBoard.get_passives_captures(simple_board_state.board_a, simple_board_state.colour, simulate_safe_moves=True)
        safe_moves = np.concatenate((passives, captures), axis=0)

        if safe_moves.shape[0] == 0: # No moves to compute - terminal conditions
            return (None, DRAW_STATE_VALUE) # TODO: What should be the right board value for this?

        move_values = self.value_function(simple_board_state, safe_moves)

        selected_vals = None
        selected_moves = None
        if player_colour == simple_board_state.colour and self.play_to_lose: # Minimize
            sorted_ixs = np.argsort(move_values)[0:self.branching_factor]
            selected_vals = move_values[sorted_ixs]
            selected_moves = safe_moves[sorted_ixs]
        else: # Maximize
            sorted_ixs = np.argsort(move_values)[::-1][0:self.branching_factor]
            selected_vals = move_values[sorted_ixs]
            selected_moves = safe_moves[sorted_ixs]

        if depth+1 == max_depth:
            ixs = None
            if player_colour == simple_board_state.colour and self.play_to_lose:
                ixs = np.argmin(selected_vals, keepdims=True)
            else:
                ixs = np.argmax(selected_vals, keepdims=True)
            rng_ix = self.rng.choice(ixs)
            return (selected_moves[rng_ix], selected_vals[rng_ix])
        else:
            vals = np.zeros(selected_vals.shape)
            for i in range(selected_moves.shape[0]):
                move = selected_moves[i]
                vals[i] = self._tree_search(depth+1, max_depth, BoardState.move(simple_board_state, move, simple=True), player_colour)[1]
            if player_colour == simple_board_state.colour and self.play_to_lose:
                ixs = np.argmin(vals, keepdims=True)
            else:
                ixs = np.argmax(vals, keepdims=True)

            ixs = np.argmax(vals, keepdims=True) if player_colour == simple_board_state.colour else np.argmin(vals, keepdims=True)
            rng_ix = self.rng.choice(ixs)
            return (selected_moves[rng_ix], vals[rng_ix])

    @staticmethod
    def _get_std_val_func():
        vectorized_value_transform = np.vectorize((lambda x: 0 if x == 0 else FIGURE_ID_MAP[x][0].value*(x/np.abs(x))), otypes='f')

        def _std_val_func(simple_board_state: 'SimpleBoardState', moves: np.ndarray) -> np.ndarray:
            vals = []
            for i in range(moves.shape[0]):
                move = moves[i]
                sbs = BoardState.move(simple_board_state, move, simple=True)
                value_board = vectorized_value_transform(sbs.board_a)
                value_board[np.isinf(value_board)] = 0
                vals.append(np.sum(value_board))
            vals_a = np.array(vals)
            if simple_board_state.colour == Colour.BLACK:
                return vals_a * -1
            else:
                return vals_a

        return _std_val_func

class AlphaBetaTreeSearchPlayer(Player):
    def __init__(self, search_depth=1, rand_seed=None, play_to_lose=False, memory=None, value_function : Callable[['SimpleBoardState'], int] = None, value_function_name=""):
        super().__init__(memory)
        self.value_function_name = value_function_name
        self.search_depth = search_depth
        self.play_to_lose = play_to_lose

        self.inf = np.inf
        self.neg_inf = -1*np.inf

        if rand_seed == None:
            rand_seed = np.random.default_rng().integers(-(2**63), high=(2**63 - 1))
        self.seed = rand_seed
        self.rng = np.random.default_rng(np.abs(rand_seed))

        if value_function == None:
            self.value_function = AlphaBetaTreeSearchPlayer._get_std_val_func()
        else:
            self.value_function = value_function

    def send_action(self, board_state: BoardState) -> np.ndarray:
        moves = np.concatenate([board_state.captures, board_state.passives], axis=0)
        best_move, best_value = self._tree_search(0, self.search_depth, self.neg_inf, self.inf, board_state.simplify(), moves)
        super().step_memory(board_state, best_move)
        return best_move

    def receive_reward(self, reward_value: int, move_history: list):
        return super().commit_memory(reward_value)

    def __str__(self) -> str:
        return f"AlphaBeta TreeSearch Player (Value Function={self.value_function_name}, search_depth={self.search_depth}, play_to_lose={self.play_to_lose})"

    def _tree_search(self, depth, max_depth, alpha, beta, simple_board_state: 'SimpleBoardState', unsafe_moves):
        # White is alpha (maximizing)
        # Black is beta (minimizing)
        if King.id*Colour.WHITE not in simple_board_state.board_a: # Black has won - terminal condition
            return (None, self.neg_inf)
        if King.id*Colour.BLACK not in simple_board_state.board_a: # White has won - terminal condition
            return (None, self.inf)
        
        if depth == max_depth: # End of search tree - terminal condition
            return (None, self.value_function(simple_board_state))

        moves = unsafe_moves

        if moves.shape[0] == 0:
            return (None, DRAW_STATE_VALUE) # TODO: What should be the right board value for this?

        # recursive case
        best_moves, best_value = [], None
        ally_king_pos, enemy_king_pos = ChessBoard.get_ally_enemy_king_pos(simple_board_state.board_a, simple_board_state.colour)

        maximize = False
        if (not self.play_to_lose and simple_board_state.colour == Colour.WHITE) or (self.play_to_lose and simple_board_state.colour == Colour.BLACK):
            maximize = True

        best_value = self.neg_inf if maximize else self.inf
        for i in range(moves.shape[0]):
            is_safe_move, next_unsafe_moves = None, None
            if depth + 1 == max_depth:
                is_safe_move, next_unsafe_moves = ChessBoard.is_safe_move(simple_board_state.board_a, moves[i], simple_board_state.colour, ally_king_pos, enemy_king_pos), np.empty(shape=(0,8), dtype=np.int32)
            else:
                is_safe_move, next_unsafe_moves = ChessBoard.is_safe_move_simulated(simple_board_state.board_a, moves[i], simple_board_state.colour, ally_king_pos, enemy_king_pos)
            if is_safe_move:
                move = moves[i]
                sim_board_state = BoardState.move(simple_board_state, move, simple=True)
                _, value = self._tree_search(depth+1, max_depth, alpha, beta, sim_board_state, next_unsafe_moves)
                if maximize:
                    if value > best_value:
                        best_value = value
                        best_moves = [move]
                    elif value == best_value:
                        best_moves.append(move)
                    if best_value >= beta:
                        break # beta cutoff
                    alpha = max(alpha, best_value)
                else:
                    value = value * -1
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

    @staticmethod
    def _get_std_val_func():
        vectorized_value_transform = np.vectorize((lambda x: 0 if x == 0 else FIGURE_ID_MAP[x][0].value*(x/np.abs(x))), otypes='f')

        def _std_val_func(simple_board_state: 'SimpleBoardState') -> int:
            value_board = vectorized_value_transform(simple_board_state.board_a)
            value_board[np.isinf(value_board)] = 0
            if simple_board_state.colour == Colour.BLACK:
                return -1 * np.sum(value_board)
            else:
                return np.sum(value_board)

        return _std_val_func