import numpy as np

from raumschach.board import INITIAL_5_5_BOARD_SETUP, ChessBoard
from raumschach.board_state import BoardState
from raumschach.figures import FIGURE_ID_MAP, FIGURE_NAME_MAP, Colour, King, Pawn
from raumschach.players.player import Player
from raumschach.render import render_board_ascii

class IllegalActionException(Exception):
    """ illegal action exception class """


SIZE_TO_SETUP_MAP = {
    5: INITIAL_5_5_BOARD_SETUP
}

class ChessGame():

    def __init__(self, player1: Player, player2: Player, board_size):
        setup = SIZE_TO_SETUP_MAP[board_size] if board_size in SIZE_TO_SETUP_MAP else []
        # self.chess_board = ChessBoard(board_size, setup)
        self.board_state = BoardState.game_setup(board_size, setup)
        self.players = ((player1, Colour.WHITE), (player2, Colour.BLACK))
        self.move_history = []

        self.no_progress = 0 # 50-Move rule (if no captures happened or no pawn has been moved for the last 50 moves, the game ends in a draw)
        self.state_repetition_map = {} # Threefold-repetition rule (if the same board state with the same moves occurs for the third time, the game ends in a draw)
        self.is_checked = [False, False]
        self.is_checkmate = [False, False]
        self.next_player_passives_captures = None

    def play_script(self, script):
        for i in range(len(self.players)):
            self.players[i][0].script = script
        self.play()

    def play(self):
        player_num = 0
        message = "-/-"
        while not any(self.is_checkmate):
            message = self.turn(player_num)
            player_num = (player_num + 1) % len(self.players)
        render_board_ascii(self.board_state.board_a)
        print('\n' + message)
        winner = 0
        if all(self.is_checkmate):
            print("Draw!")
            winner = 0
        elif self.is_checkmate[0]:
            print("Black wins!")
            winner = -1
        elif self.is_checkmate[1]:
            print("White wins!")
            winner = 1
        print(f"\nMove history: ({len(self.move_history)})")
        print(self.move_history)
        return winner

    def turn(self, player_num):
        player, colour = self.players[player_num]
        enemy_player_num = (player_num + 1) % len(self.players)
        enemy_player, enemy_colour = self.players[enemy_player_num]
        message = None

        # Generate all possible moves for the pieces of this player
        # We need to generate the moves of all colours however since the King cannot put himself into check

        # passives, captures = self.next_player_passives_captures if self.next_player_passives_captures != None else ChessBoard.get_passives_captures(self.current_board_state.board_a, colour)

        # send the observation to the player and receive the corresponding action
        # The action is only allowed to be a move
        # hash_val = self.chess_board.cube.data.tobytes()
        # state_repetition = self.state_repetition_map[hash_val] if hash_val in self.state_repetition_map else 0
        action = player.send_action(self.board_state)

        # TODO Negative reward for doing illegal move
        # check if the action is legal
        if not BoardState.is_legal_move(self.board_state, action):
            raise IllegalActionException("The given action is not part of the currently legal moves or captures.")
        

        # Implement the action on the chess board
        prev_board_state = self.board_state
        self.board_state = BoardState.move(self.board_state, action)


        # prev_board_a = self.chess_board.cube.copy()
        # ChessBoard.move(self.chess_board.cube, action)

        # Update the no progress rule
        if not message: # The game has not yet ended
            if self.board_state.no_progress_count >= 50:
                self.is_checked[player_num] = False
                self.is_checked[enemy_player_num] = False
                self.is_checkmate[player_num] = True
                self.is_checkmate[enemy_player_num] = True
                player.receive_reward(0, self.move_history)
                enemy_player.receive_reward(0, self.move_history)
                message = "No progress has been achieved the last 50 turns (No pawn moved & no piece captured)." # Draw


        # If the same state was repeated three times, then this will result in an automatic draw
        # Update the board state repetition map
        if not message: # The game has not yet ended
            # Check if threefold repetition rule applies
            if self.board_state.state_repetition_count >= 3:
                self.is_checked[player_num] = False
                self.is_checked[enemy_player_num] = False
                self.is_checkmate[player_num] = True
                self.is_checkmate[enemy_player_num] = True
                player.receive_reward(0, self.move_history)
                enemy_player.receive_reward(0, self.move_history)
                message = "The same board position has been repeated for the third time." # Draw


        if not message: # The game has not yet ended
            # Determine whether we have a checkmate
            if True or self.is_checked[enemy_player_num]: # TODO Ideally this True is obsolete
                # The enemy player is under check
                # Simply check whether the enemy king has been captured i.e. if the king still stands on the board
                if ChessBoard.is_king_checkmate(self.board_state.board_a, enemy_colour):
                    self.is_checked[player_num] = False
                    self.is_checked[enemy_player_num] = False
                    self.is_checkmate[enemy_player_num] = True
                    player.receive_reward(1, self.move_history)
                    enemy_player.receive_reward(-1, self.move_history)
                    message = f"Checkmate - ({Colour.string(colour)}) has captured the enemy's king"


        # # Generate next moves of white and black pieces and assign them to either this player or the enemy player based on colour
        # if not message: # The game has not yet ended
        #     white_next_moves, black_next_moves = ChessBoard.get_passives_captures(self.chess_board.cube)
        #     this_p_moves = [black_next_moves, None, white_next_moves][1+colour]
        #     next_p_moves = [black_next_moves, None, white_next_moves][1+enemy_colour]
        #     self.next_player_passives_captures = next_p_moves


        # Determine whether we (still) have a check situation
        if not message: # The game has not yet ended
            self.is_checked[player_num] = ChessBoard.is_king_under_check(self.board_state.board_a, colour)
            self.is_checked[enemy_player_num] = ChessBoard.is_king_under_check(self.board_state.board_a, enemy_colour)

        # Check if the enemy player is able to do any moves on their next turn
        if not message: # The game has not yet ended
            if self.board_state.passives.shape[0] == 0 and self.board_state.captures.shape[0] == 0:
                if self.is_checked[enemy_player_num]:
                    self.is_checked[enemy_player_num] = False
                    self.is_checkmate[enemy_player_num] = True
                    player.receive_reward(1, self.move_history)
                    enemy_player.receive_reward(-1, self.move_history)
                    message = f"({Colour.string(enemy_colour)}) is checked and does not have any available moves"
                else:
                    self.is_checkmate[player_num]       = True
                    self.is_checkmate[enemy_player_num] = True
                    player.receive_reward(0, self.move_history)
                    enemy_player.receive_reward(0, self.move_history)
                    message = f"({Colour.string(enemy_colour)}) is not checked and does not have any available moves - automatic stalemate"

        # Record the move in the move history
        self.move_history.append(ChessBoard.record_move(prev_board_state.board_a, self.board_state.board_a, action, self.is_checked, self.is_checkmate))

        render_board_ascii(self.board_state.board_a)
        print(f"Total Moves: {('('+str(len(self.move_history))+')').ljust(5)} | Most recent moves: ", " <-- ".join([hist.center(15, ' ') for hist in self.move_history[-1: -6: -1]]))

        if message:
            return message


        # Reward of moves generally is 0 - Reward of win is +1 - Reward of loss is -1 - Reward of draw is 0
        player.receive_reward(0, self.move_history)