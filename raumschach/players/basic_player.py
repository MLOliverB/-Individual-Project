from raumschach.board_state import BoardState
from raumschach.players.player import Player

from raumschach.render import render_board_ascii, render_figure_moves_ascii
from raumschach.board import ChessBoard
from raumschach.figures import Colour

import numpy as np


class DummyPlayer(Player):
    script = None
    def __init__(self):
        super().__init__()

    def send_action(self, board_state: BoardState):
        render_board_ascii(board_state.board_a)
        return ChessBoard.read_recorded_move(self.script.pop(0))

    def receive_reward(self, reward_value, move_history):
        return super().receive_reward(reward_value, move_history)


class RandomPlayer(Player):
    def __init__(self, rand_seed=None, memory=None):
        super().__init__(memory=memory)
        if rand_seed == None:
            rand_seed = np.random.default_rng().integers(-(2**63), high=(2**63 - 1))
        self.seed = rand_seed
        self.rng = np.random.default_rng(np.abs(rand_seed))

    def send_action(self, board_state: BoardState):
        len_passives = board_state.passives.shape[0]
        len_captures = board_state.captures.shape[0]
        rand_ix = self.rng.integers(0, len_passives+len_captures)

        move = None
        if rand_ix < len_passives:
            move = board_state.passives[rand_ix]
        else:
            move = board_state.captures[len_passives - rand_ix]

        super().step_memory(board_state, move)

        return move

    def receive_reward(self, reward_value, move_history):
        return super().commit_memory(reward_value)

class ConsolePlayer(Player):
    def __init__(self, memory=None):
        super().__init__(memory=memory)

    def send_action(self, board_state: BoardState):
        render_board_ascii(board_state.board_a)
        move = self._get_input(board_state)
        super().step_memory(board_state, move)
        return move

    def _get_input(self, board_state: BoardState):
        txt_col = "White" if board_state.colour == Colour.WHITE else "Black"
        action_input = input(f"(Move {txt_col}) >> ")
        action_input_split = action_input.split(' ')

        coord1 = None
        coord2 = None

        if len(action_input_split) == 1:
            try:
                coord1 = ChessBoard.get_pos_coord(action_input_split[0])
            except:
                coord1 = -1
        elif len(action_input_split) == 2:
            try:
                coord1 = ChessBoard.get_pos_coord(action_input_split[0])
            except:
                coord1 = -1
            try:
                coord2 = ChessBoard.get_pos_coord(action_input_split[1])
            except:
                coord2 = -1

        if coord1 == -1 or coord2 == -1:
            # One of the positions could not be parsed
            print("Invalid input - Either input a single board position (e.g. Aa1) to display the moves of a chess piece")
            print("              - Or input two board positions (e.g. Aa1 Ab1) to move the piece from the first position to the second")
            render_board_ascii(board_state.board_a)
            return self._get_input(board_state)
        else:
            # Check if coord1 points to ally piece
            if not board_state.board_a[coord1]*board_state.colour > 0:
                print("The given position does not belong to an ally piece")
                render_board_ascii(board_state.board_a)
                return self._get_input(board_state)
            else:
                if coord2 == None:
                    # Only one coord is given - display available moves for that piece
                    matches = BoardState.get_matching_passives_captures(board_state, coord1)
                    if isinstance(matches, tuple):
                        render_figure_moves_ascii(board_state.board_a, matches)
                        return self._get_input(board_state)
                    else:
                        print("The given piece does not have any available moves")
                        render_board_ascii(board_state.board_a)
                        return self._get_input(board_state)
                else:
                    # Check if given move exists and return it
                    move = BoardState.get_matching_passives_captures(board_state, coord1, coord2)
                    if isinstance(move, np.ndarray):
                        return move
                    else:
                        print("Invalid input - You cannot move/capture to the specified location")
                        print("input a single board position (e.g. Aa1) to display the moves of a chess piece")
                        render_board_ascii(board_state.board_a)
                        return self._get_input(board_state)
        #     pass
        # elif coord2 == None:
        #     matches = BoardState.get_matching_passives_captures(board_state, coord1)
        #     if isinstance(matches, tuple):
        #         pass
        #     else:
        #         print("I")
        #         print("input a single board position (e.g. Aa1) to display the moves of a chess piece")
        #     # Only one position given - Check if player owns position then display moves
        # else:
        #     # Both positions given - Check if valid move
        #     BoardState.get_matching_passives_captures(board_state, coord1, coord2)


        #     print("Invalid input - You cannot move the piece at the specified position")
        #     print("input a single board position (e.g. Aa1) to display the moves of a chess piece")

        #     # if not BoardState.is_legal_move(board_state)
        #     print("Invalid input - You cannot move/capture to the specified location")
        #     print("input a single board position (e.g. Aa1) to display the moves of a chess piece")



        # return None

    # def _get_input(self, board_state: BoardState, colour_str):
    #     render_board_ascii(board_state.board_a)
    #     action_input = input(f"(Move {colour_str}) >> ")
    #     action_input_split = action_input.split(' ')
    #     len_action_input_split = len(action_input_split)
    #     if len_action_input_split == 1:
    #         pos_coord = None
    #         try:
    #             pos_coord = ChessBoard.get_pos_coord(action_input_split[0])
    #         except:
    #             print("Invalid input - Either input a single board position (e.g. Aa1) to display the moves of a chess piece")
    #             print("              - Or input two board positions (e.g. Aa1 Ab1) to move the piece from the first position to the second")
    #         if pos_coord != None:
    #             render_figure_moves_ascii(board_state.board_a, ChessBoard.get_pos_coord(action_input))                
    #         return self._get_input(board_state, colour_str)
    #     elif len_action_input_split == 2:
    #         pos_coord1 = None
    #         pos_coord2 = None
    #         try:
    #             pos_coord1 = ChessBoard.get_pos_coord(action_input_split[0])
    #             pos_coord2 = ChessBoard.get_pos_coord(action_input_split[1])
    #         except:
    #             print("Invalid input - Either input a single board position (e.g. Aa1) to display the moves of a chess piece")
    #             print("              - Or input two board positions (e.g. Aa1 Ab1) to move the piece from the first position to the second")
    #         if pos_coord1 != None and pos_coord2 != None:
    #             pos_coord1_a = np.array(pos_coord1)
    #             pos_coord2_a = np.array(pos_coord2)
    #             if pos_coord1_a not in board_state.passives[:, 2:5] and pos_coord1_a not in board_state.captures[:, 2:5]:
    #                 print("Invalid input - You cannot move the piece at the specified position")
    #                 print("input a single board position (e.g. Aa1) to display the moves of a chess piece")
    #                 return self._get_input(board_state, colour_str)
    #             else:
    #                 pos_coord1_passives, pos_coord1_captures = ChessBoard.get_piece_passives_captures(board_state.passives, board_state.captures, pos_coord1)
    #                 if pos_coord2 not in pos_coord1_passives[:, 5:8] and pos_coord2 not in pos_coord1_captures[:, 5:8]:
    #                     print("Invalid input - You cannot move/capture to the specified location")
    #                     print("input a single board position (e.g. Aa1) to display the moves of a chess piece")
    #                     return self._get_input(board_state, colour_str)
    #                 else:
    #                     return ChessBoard.get_move_from_passives_captures(pos_coord1_passives, pos_coord1_captures, pos_coord1_a, pos_coord2_a)
    #         else:
    #             return self._get_input(board_state, colour_str)
    #     else:
    #         print("Invalid input - Either input a single board position (e.g. Aa1) to display the moves of a chess piece (Too many arguments)")
    #         print("              - Or input two board positions (e.g. Aa1 Ab1) to move the piece from the first position to the second")
    #         return self._get_input(board_state, colour_str)

    def receive_reward(self, reward_value, move_history):
        return super().commit_memory(reward_value)