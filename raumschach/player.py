from abc import ABCMeta, abstractmethod
import random
from raumschach.board import ChessBoard
from raumschach.figures import FIGURE_ID_MAP, Colour
from raumschach.game import ChessGame

from raumschach.render import render_board_ascii, render_figure_moves_ascii
import numpy as np


class Player(object, metaclass=ABCMeta):
    def __init__(self, name):
        self.name = name

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
    def __init__(self, name):
        super().__init__(name)

    def receive_observation(self, board_state):
        return board_state

    def send_action(self, observation):
        render_board_ascii(observation.cb)
        return ChessGame.read_recorded_move(self.script.pop(0))

    def receive_reward(self, reward_value, move_history):
        return super().receive_reward(reward_value, move_history)


class RandomPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def receive_observation(self, board_state):
        return board_state

    def send_action(self, observation):
        moves = None
        if type(observation.passives) == type(None) and type(observation.captures) == type(None):
            raise Exception("No moves available")
        elif type(observation.passives) == type(None):
            moves = observation.captures
        elif type(observation.captures) == type(None):
            moves = observation.passives
        else:
            moves = np.concatenate((observation.passives, observation.captures))
        rand = np.random.randint(0, moves.shape[0])
        return moves[rand]
        raise Exception()
        pieces = [*set([*observation.passives] + [*observation.captures])]
        from_pos = pieces[random.randint(0, len(pieces)-1)] # random.choice([*set([*observation.passives] + [*observation.captures])])
        moves = observation.passives[from_pos] if from_pos in observation.passives else []
        captures = observation.captures[from_pos] if from_pos in observation.captures else []
        while not [*set(moves + captures)]:
            from_pos = random.choice([*set([*observation.passives] + [*observation.captures])])
            moves = observation.passives[from_pos] if from_pos in observation.passives else []
            captures = observation.captures[from_pos] if from_pos in observation.captures else []
        moves_captures = [*set(moves + captures)]
        to_pos = moves_captures[random.randint(0, len(moves_captures)-1)] # random.choice([*set(moves + captures)]) - but choice somehow breaks after some time
        # render_board_ascii(observation.cb)
        return (from_pos, to_pos)

    def receive_reward(self, reward_value, move_history):
        return super().receive_reward(reward_value, move_history)


class ConsolePlayer(Player):
    def __init__(self, name):
        super().__init__(name)

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