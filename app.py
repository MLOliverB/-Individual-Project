from raumschach.board import INITIAL_5_5_BOARD_SETUP, ChessBoard
from raumschach.figures import Colour, Pawn, Queen
from raumschach.game import ChessGame
from raumschach.player import ConsolePlayer, DummyPlayer, RandomPlayer
from raumschach.render import render_board_ascii, render_figure_moves_ascii

# board = ChessBoard(5, INITIAL_5_5_BOARD_SETUP)
# render_board_ascii(board)

# board._add(Pawn, Colour.BLACK, ChessBoard.get_pos_coord("Bc3"))
# render_board_ascii(board)
# m = board.get_figure_moves(ChessBoard.get_pos_coord("Bb2"))
# print(m[0], m[1])

# board = ChessBoard(5, INITIAL_5_5_BOARD_SETUP)
# render_board_ascii(board)
# render_figure_moves_ascii(board, ChessBoard.get_pos_coord("Bc1"))
# board.move(ChessBoard.get_pos_coord("Bc1"), ChessBoard.get_pos_coord("Ec4"))
# render_figure_moves_ascii(board, ChessBoard.get_pos_coord("Ec4"))
# render_figure_moves_ascii(board, ChessBoard.get_pos_coord("Db5"))
# board.move(ChessBoard.get_pos_coord("Db5"), ChessBoard.get_pos_coord("Ec4"))
# render_board_ascii(board)
# print(board.cube.data.tobytes())

# game = ChessGame(ConsolePlayer("P1"), ConsolePlayer("P2"), 5)
# game.play()

game = ChessGame(RandomPlayer("P1"), RandomPlayer("P2"), 5)
game.play()

# threefold_repetition_script = ['N:Ab1-Aa3', 'n:Eb5-Ea3', 'N:Aa3-Ab1', 'n:Ea3-Eb5', 'N:Ab1-Aa3', 'n:Eb5-Ea3', 'N:Aa3-Ab1', 'n:Ea3-Eb5', 'N:Ab1-Aa3']
# game = ChessGame(DummyPlayer("P1"), DummyPlayer("P2"), 5)
# game.play_script(threefold_repetition_script)