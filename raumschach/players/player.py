from abc import ABCMeta, abstractmethod
import numpy as np

from raumschach.board_state import BoardState


class Player(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def send_action(self, board_state: BoardState) -> np.ndarray:
        pass

    @abstractmethod
    def receive_reward(self, reward_value: int, move_history: list):
        pass