from raumschach.board_state import BoardState
from raumschach.players.player import Player


class MoveValueClassifierPlayer(Player):
    def __init__(self):
        super().__init__()

    def send_action(self, board_state: BoardState):
        pass

    def receive_reward(self, reward_value, move_history):
        pass