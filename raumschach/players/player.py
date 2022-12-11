from abc import ABCMeta, abstractmethod


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