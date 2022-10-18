
from raumschach.board import INITIAL_5_5_BOARD_SETUP, ChessBoard


SIZE_TO_SETUP_MAP = {
    5: INITIAL_5_5_BOARD_SETUP
}

class ChessGame():

    def __init__(self, player1, player2, board_size):
        setup = SIZE_TO_SETUP_MAP[board_size] if board_size in SIZE_TO_SETUP_MAP else []
        self.board = ChessBoard(board_size, setup)
        self.player1 = player1
        self.player2 = player2

    def play(self):
        pass