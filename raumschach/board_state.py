import numpy as np

from raumschach.board import ChessBoard
from raumschach.figures import FIGURE_ID_MAP, Colour, Pawn

REWARD_WIN = 1
REWARD_DRAW = 0
REWARD_LOSS = -1

class BoardState():

    def __init__(self, turn_no: int, colour, board_a: np.ndarray, passives: np.ndarray, captures: np.ndarray, no_progress_count: int, state_repetition_count: int, state_repetition_map: dict[bytes, int]) -> None:
        self.turn = turn_no
        self.colour = colour
        self.board_a = board_a
        self.passives = passives
        self.captures = captures
        self.no_progress_count = no_progress_count
        self.state_repetition_count = state_repetition_count
        self.state_repetition_map = state_repetition_map

    def simplify(self):
        return SimpleBoardState(self.turn, self.board_a.copy(), self.colour, self.no_progress_count, self.state_repetition_count, self.state_repetition_map)

    @staticmethod
    def game_setup(board_size: int, setup: list[str]) -> 'BoardState':
        board_a = ChessBoard.init(board_size, setup)
        passives, captures = ChessBoard.get_passives_captures(board_a, Colour.WHITE)
        state_repetition_map = dict()
        state_repetition_map[board_a.data.tobytes()] = 1
        return BoardState(0, Colour.WHITE, board_a, passives, captures, 0, 1, state_repetition_map)

    @staticmethod
    def move(board_state: 'BoardState', move: np.ndarray, simple=False) -> 'BoardState':
        board_a = ChessBoard.move(board_state.board_a, move)
        colour = board_state.colour*-1

        # Update no progress rule
        no_progress_count = board_state.no_progress_count + 1
        if FIGURE_ID_MAP[move[0]][0] == Pawn: # A pawn was moved
            no_progress_count = 0
        elif board_state.board_a[move[5], move[6], move[7]] != 0: # A piece is captured through this move
            no_progress_count = 0
        # if simple:
        #     if FIGURE_ID_MAP[move[0]][0] == Pawn: # A pawn was moved
        #         no_progress_count = 0
        #     elif board_state.board_a[move[5], move[6], move[7]] != 0: # A piece is captured through this move
        #         no_progress_count = 0
        # else:
        #     if FIGURE_ID_MAP[move[0]][0] == Pawn: # A pawn was moved
        #         no_progress_count = 0
        #     elif np.any(np.all(move == board_state.captures, axis=1)): # A piece was captured
        #         no_progress_count = 0

        # Update state repetition rule
        state_repetition_map = board_state.state_repetition_map.copy()
        hash_val = board_a.data.tobytes()
        if hash_val in state_repetition_map:
            state_repetition_map[hash_val] += 1
        else:
            state_repetition_map[hash_val] = 1

        if simple:
            return SimpleBoardState(board_state.turn+1, board_a, colour, no_progress_count, state_repetition_map[hash_val], state_repetition_map)
        else:
            passives, captures = ChessBoard.get_passives_captures(board_a, colour)
            return BoardState(board_state.turn+1, colour, board_a, passives, captures, no_progress_count, state_repetition_map[hash_val], state_repetition_map)

    @staticmethod
    def is_legal_move(board_state: 'BoardState', move: np.ndarray) -> bool:
        is_in_passives = np.any(np.all(move == board_state.passives, axis=1))
        is_in_captures = np.any(np.all(move == board_state.captures, axis=1))

        if is_in_passives or is_in_captures:
            return True
        else:
            return False

    @staticmethod
    def get_matching_passives_captures(board_state: 'BoardState', from_coord, to_coord=None):
        if to_coord:
            is_in_passives = np.any(np.all(np.array(from_coord) == board_state.passives[:, 2:5], axis=1) & np.all(np.array(to_coord) == board_state.passives[:, 5:8], axis=1))
            is_in_captures = np.any(np.all(np.array(from_coord) == board_state.captures[:, 2:5], axis=1) & np.all(np.array(to_coord) == board_state.captures[:, 5:8], axis=1))

            if is_in_passives or is_in_captures:
                arr = board_state.passives if is_in_passives else board_state.captures
                return arr[np.all(np.array(from_coord) == arr[:, 2:5], axis=1) & np.all(np.array(to_coord) == arr[:, 5:8], axis=1)][0]
            else:
                return None
        else:
            passive_matches = board_state.passives[np.all(np.array(from_coord) == board_state.passives[:, 2:5], axis=1)]
            capture_matches = board_state.captures[np.all(np.array(from_coord) == board_state.captures[:, 2:5], axis=1)]

            if passive_matches.shape[0] == 0 and capture_matches.shape[0] == 0:
                return None
            else:
                return passive_matches, capture_matches

class SimpleBoardState(BoardState):
    def __init__(self, turn_no, cube, colour, no_progress_count, state_repetition, state_repetition_map):
        super().__init__(turn_no, colour, cube, None, None, no_progress_count, state_repetition, state_repetition_map)