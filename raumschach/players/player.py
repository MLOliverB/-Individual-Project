from abc import ABCMeta, abstractmethod
import numpy as np

from raumschach.board_state import REWARD_DRAW, REWARD_LOSS, REWARD_WIN, BoardState
# from raumschach.game import REWARD_DRAW, REWARD_LOSS, REWARD_WIN



class Player(object, metaclass=ABCMeta):
    def __init__(self, memory=None):
        self.memory = memory
        self.memory_buffer = []

    def step_memory(self, board_state: BoardState, move: np.ndarray):
        if self.memory != None:
            self.memory_buffer.append(BoardState.move(board_state, move, simple=True))

    def commit_memory(self, reward_value: int):
        if self.memory_buffer:
            win_count = 0
            draw_count = 0
            loss_count = 0
            if reward_value == REWARD_WIN:
                win_count = 1
            elif reward_value == REWARD_DRAW:
                draw_count = 1
            elif reward_value == REWARD_LOSS:
                loss_count = 1

            for simple_board_state in self.memory_buffer:
                self.memory.push(simple_board_state, win_count=win_count, draw_count=draw_count, loss_count=loss_count)

            self.memory_buffer = []

    @abstractmethod
    def send_action(self, board_state: BoardState) -> np.ndarray:
        pass

    @abstractmethod
    def receive_reward(self, reward_value: int, move_history: list):
        pass