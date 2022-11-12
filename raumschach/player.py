from abc import ABCMeta, abstractmethod
import numpy as np
from raumschach.board import ChessBoard
from raumschach.figures import FIGURE_ID_MAP, Colour, King
from raumschach.game import ChessGame
from raumschach.render import render_board_ascii, render_figure_moves_ascii


class Player(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def receive_observation(self, board_state):
        pass

    def send_action(self, observation):
        pass

    @abstractmethod
    def receive_reward(self, reward_value, move_history):
        pass


class DummyPlayer(Player):
    script = None
    def __init__(self):
        super().__init__()

    def receive_observation(self, board_state):
        return board_state

    def send_action(self, observation):
        render_board_ascii(observation.cb)
        return ChessGame.read_recorded_move(self.script.pop(0))

    def receive_reward(self, reward_value, move_history):
        return super().receive_reward(reward_value, move_history)


class RandomPlayer(Player):
    def __init__(self, rand_seed=None):
        super().__init__()
        if rand_seed == None:
            rand_seed = np.random.default_rng().integers(-(2**63), high=(2**63 - 1))
        self.seed = rand_seed
        self.rng = np.random.default_rng(np.abs(rand_seed))

    def receive_observation(self, board_state):
        return board_state

    def send_action(self, observation):
        pieces = [*set([*observation.passives] + [*observation.captures])]
        from_pos = pieces[self.rng.integers(0, len(pieces))]
        # from_pos = pieces[random.randint(0, len(pieces)-1)] # random.choice([*set([*observation.passives] + [*observation.captures])])
        moves = observation.passives[from_pos] if from_pos in observation.passives else []
        captures = observation.captures[from_pos] if from_pos in observation.captures else []
        while not [*set(moves + captures)]:
            from_pos = pieces[self.rng.integers(0, len(pieces))]
            # from_pos = random.choice([*set([*observation.passives] + [*observation.captures])])
            moves = observation.passives[from_pos] if from_pos in observation.passives else []
            captures = observation.captures[from_pos] if from_pos in observation.captures else []
        moves_captures = [*set(moves + captures)]
        to_pos = moves_captures[self.rng.integers(0, len(moves_captures))]
        # to_pos = moves_captures[random.randint(0, len(moves_captures)-1)] # random.choice([*set(moves + captures)]) - but choice somehow breaks after some time
        # render_board_ascii(observation.cb)
        return (from_pos, to_pos)

    def receive_reward(self, reward_value, move_history):
        return super().receive_reward(reward_value, move_history)



class MiniMaxPlayer(Player):
    def __init__(self, search_depth=1, discount_factor=1, rand_seed=None):
        super().__init__()
        self.search_depth = search_depth
        self.gamma = discount_factor

        if rand_seed == None:
            rand_seed = np.random.default_rng().integers(-(2**63), high=(2**63 - 1))
        self.seed = rand_seed
        self.rng = np.random.default_rng(np.abs(rand_seed))

        self.vectorized_value_transform = np.vectorize((lambda x: 0 if x == 0 else FIGURE_ID_MAP[x][0].value*(x/np.abs(x))), otypes='f')


    def receive_observation(self, board_state):
        return board_state

    def _passives_captures_dict_to_move_list(self, passives, captures):
        moves = []
        for key in passives.keys():
            for val in passives[key]:
                moves.append((key, val))
        for key in captures.keys():
            for val in captures[key]:
                moves.append((key, val))
        return moves

    def send_action(self, observation):
        board_a = observation.cb
        moves = self._passives_captures_dict_to_move_list(observation.passives, observation.captures)
        best_move_ix, best_move_value = self._resursive_best_move_values(1, self.search_depth, moves, board_a, observation.colour, observation.colour)
        return moves[best_move_ix]

    def _resursive_best_move_values(self, depth, max_depth, moves, board_a, player_colour, current_colour):
        # if depth > max_depth:
        #     return 0, 0
        # elif King.id*player_colour not in board_a: # The enemy player has won
        #     return np.inf*-1, np.inf*-1
        # elif King.id*-1*player_colour not in board_a: # This player has won
        #     return np.inf, np.inf
        # else:
        len_moves = len(moves)
        move_values = np.full(len_moves, -1*np.inf, dtype=np.float32)
        for i in range(len_moves):
            from_move, to_move = moves[i]
            sim_board_a = np.array(board_a)
            ChessBoard.move(sim_board_a, from_move, to_move)
            sim_passives, sim_captures = ChessBoard.get_passives_captures(sim_board_a, current_colour)
            sim_moves = self._passives_captures_dict_to_move_list(sim_passives, sim_captures)

            best_sim_move_ix, best_sim_value = None, None
            if depth+1 > max_depth:
                best_sim_move_ix, best_sim_value = 0, 0
            elif King.id*player_colour not in sim_board_a: # The enemy player has won
                best_sim_move_ix, best_sim_value =  np.inf*-1, np.inf*-1
            elif King.id*-1*player_colour not in sim_board_a: # This player has won
                best_sim_move_ix, best_sim_value =  np.inf, np.inf
            else:
                best_sim_move_ix, best_sim_value = self._resursive_best_move_values(depth+1, max_depth, sim_moves, sim_board_a, player_colour, current_colour*-1)

            sim_board_a = sim_board_a if current_colour == Colour.WHITE else sim_board_a*-1
            sim_value_board = self.vectorized_value_transform(sim_board_a)
            if np.inf in sim_value_board and -1*np.inf in sim_value_board:
                sim_value_board[sim_value_board == np.inf] = 0
                sim_value_board[sim_value_board == -1*np.inf] = 0
            sim_board_value = np.sum(sim_value_board)
            move_values[i] = sim_board_value + np.power(self.gamma, depth)*best_sim_value # Value of current board state plus best possible values of future moves (assuming enemy has the same policy)
        if current_colour == player_colour:
            move_values[move_values != np.max(move_values)] = 0
        else:
            move_values[move_values != np.min(move_values)] = 0
        max_move_values = move_values.nonzero()[0]
        best_move_ix = max_move_values[self.rng.integers(0, max_move_values.shape[0])]
        return (best_move_ix, move_values[best_move_ix])


    def receive_reward(self, reward_value, move_history):
        return super().receive_reward(reward_value, move_history)



# TODO Refactor random player to fit new moves style
class ConsolePlayer(Player):
    def __init__(self):
        super().__init__()

    def receive_observation(self, board_state):
        return board_state

    def send_action(self, observation):
        colour = "White" if observation.colour == Colour.WHITE else "Black"
        action_input = ""
        action = None
        render_board_ascii(observation.cb)
        while True:
            action_input = input(f"(Move {colour}) >> ")
            if len(action_input.split(' ')) == 1:
                try:
                    render_figure_moves_ascii(observation.cb, ChessBoard.get_pos_coord(action_input))
                    fig, c = FIGURE_ID_MAP[observation.cb[ChessBoard.get_pos_coord(action_input)]]
                    col = "white" if c == Colour.WHITE else "black"
                    print(f"Possible moves by {action_input} ({fig.name[1]} [{col}])")
                except:
                    print("Invalid input - Either input a single board position (e.g. Aa1) to display the moves of that chess piece")
                    print("              - Or input two board positions (e.g. Aa1 Ab1) to move the piece from the first position to the second")
                    render_board_ascii(observation.cb)
            elif len(action_input.split(' ')) == 2:
                pos1, pos2 = action_input.split(' ')
                coord1 = ChessBoard.get_pos_coord(pos1)
                coord2 = ChessBoard.get_pos_coord(pos2)
                if observation.cb[coord1] == 0:
                    print("Invalid input - You can only move your own chess pieces")
                elif (coord1 not in observation.passives) and (coord1 not in observation.captures):
                    print("Invalid input - You cannot move the piece at the specified position")
                    print("input a single board position (e.g. Aa1) to display the moves of that chess piece")
                elif (coord2 not in observation.passives.get(coord1, [])) and (coord2 not in observation.captures.get(coord1, [])):
                    print("Invalid input - You cannot move/capture to the specified location")
                    print("input a single board position (e.g. Aa1) to display the moves of that chess piece")
                else:
                    action = (coord1, coord2)
                    break
                render_board_ascii(observation.cb)
            else:
                print("Invalid input - Either input a single board position (e.g. Aa1) to display the moves of that chess piece (Too many arguments)")
                print("              - Or input two board positions (e.g. Aa1 Ab1) to move the piece from the first position to the second")
                render_board_ascii(observation.cb)
        return action

    def receive_reward(self, reward_value, move_history):
        return super().receive_reward(reward_value, move_history)