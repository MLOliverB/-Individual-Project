import numpy as np

INITIAL_BOARD_SETUP = [
    "r Ea5", "n Eb5", "k Ec5", "n Ed5", "r Ee5",
    "b Da5", "u Db5", "q Dc5", "b Dd5", "u De5",
    "p Ea4", "p Eb4", "p Ec4", "p Ed4", "p Ee4",
    "p Da4", "p Db4", "p Dc4", "p Dd4", "p De4",
    "P Ba2", "P Bb2", "P Bc2", "P Bd2", "P Be2",
    "P Aa2", "P Ab2", "P Ac2", "P Ad2", "P Ae2",
    "B Ba1", "U Bb1", "Q Bc1", "B Bd1", "U Be1",
    "R Aa1", "N Ab1", "K Ac1", "N Ad1", "R Ae1"
]



class ChessBoard:

    def __init__(self, setup):
        self.cube = np.zeros((5, 5, 5), dtype=np.byte, order='C')
        for str in setup:
            fig_name, pos_code = str.split(" ")
            self._add(fig_name, ChessBoard.get_pos_coord(pos_code))
        # self.positions = {} # encodes a char into a 3-tuple of board positions (i.e. a look-up table)
        # print(self.cube)

    def _add(self, figure_name, pos):
        self.cube[pos] = ord(figure_name)

    def get_figure(self, pos):
        chr(self.cube[pos])

    @staticmethod
    def get_pos_code(pos_coord):
        plane, file, rank = pos_coord
        return f"{ord('A') + plane}{ord('a') + rank}{file+1}"

    @staticmethod
    def get_pos_coord(pos_code):
        plane, rank, file = pos_code
        return (ord(plane) - ord('A'), int(file)-1, ord(rank) - ord('a'))


ChessBoard(INITIAL_BOARD_SETUP)