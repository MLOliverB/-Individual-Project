from abc import ABCMeta, abstractmethod
from raumschach.board import ChessBoard
from raumschach.figures import FIGURE_ID_MAP, Colour

from raumschach.render import render_board_ascii, render_figure_moves_ascii


class Player(object, metaclass=ABCMeta):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def receive_observation(self, board_state):
        pass

    def send_action(self, observation):
        pass

    @abstractmethod
    def receive_reward(self, action, reward_value):
        pass


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
                try:
                    coord1 = ChessBoard.get_pos_coord(pos1)
                    coord2 = ChessBoard.get_pos_coord(pos2)
                    if observation.cb[coord1] == 0:
                        print("Invalid input - You can only move your own chess pieces")
                    elif coord2 not in observation.moves[coord1] and coord2 not in observation.captures[coord1]:
                        print("Invalid input - You tried to move to an illegal position")
                        print("input a single board position (e.g. Aa1) to display the moves of that chess piece")
                    else:
                        action = (coord1, coord2)
                        break
                except:
                    print("Invalid input - Either input a single board position (e.g. Aa1) to display the moves of that chess piece")
                    print("              - Or input two board positions (e.g. Aa1 Ab1) to move one of your pieces from the first position to the second")
                render_board_ascii(observation.cb)
            else:
                print("Invalid input - Either input a single board position (e.g. Aa1) to display the moves of that chess piece")
                print("              - Or input two board positions (e.g. Aa1 Ab1) to move the piece from the first position to the second")
                render_board_ascii(observation.cb)
        return action

    def receive_reward(self, action, reward_value):
        return super().receive_reward(action, reward_value)