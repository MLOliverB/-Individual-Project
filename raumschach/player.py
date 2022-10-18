from abc import ABCMeta, abstractmethod


class Player(metaclass=ABCMeta):
    def __init__(self):
        # ...
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def move(self, board_state):
        pass