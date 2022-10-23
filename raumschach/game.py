import numpy as np

from raumschach.board import INITIAL_5_5_BOARD_SETUP, BoardState, ChessBoard
from raumschach.figures import FIGURE_ID_MAP, Colour, King, Pawn
from raumschach.render import render_board_ascii

class IllegalActionException(Exception):
    """ illegal action exception class """


SIZE_TO_SETUP_MAP = {
    5: INITIAL_5_5_BOARD_SETUP
}

class ChessGame():

    def __init__(self, player1, player2, board_size):
        setup = SIZE_TO_SETUP_MAP[board_size] if board_size in SIZE_TO_SETUP_MAP else []
        self.chess_board = ChessBoard(board_size, setup)
        self.players = [(player1, Colour.WHITE), (player2, Colour.BLACK)]
        self.move_history = []

        self.no_progress = 0 # 50-Move rule (if no captures happened or no pawn has been moved for the last 50 moves, the game ends in a draw)
        self.state_repetition_map = {} # Threefold-repetition rule (if the same board state with the same moves occurs for the third time, the game ends in a draw)
        self.is_checked = [False, False]
        self.is_checkmate = [False, False]

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
        render_board_ascii(self.chess_board.cube)
        print('\n' + message)
        if all(self.is_checkmate):
            print("Draw!")
        elif self.is_checkmate[0]:
            print("Black wins!")
        elif self.is_checkmate[1]:
            print("White wins!")
        # else:
        #     for i in range(len(self.is_checkmate)):
        #         if self.is_checkmate[i]:
        #             if self.players[i][1] == Colour.WHITE:
        #                 print("White wins!")
        #                 break
        #             elif self.players[i][1] == Colour.BLACK:
        #                 print("Black wins!")
        #                 break
        print("\nMove history: ")
        print(self.move_history)

    def turn(self, player_num):
        player, colour = self.players[player_num]
        enemy_player_num = (player_num + 1) % len(self.players)
        enemy_player, enemy_colour = self.players[enemy_player_num]
        message = None

        # Generate all possible moves for the pieces of this player
        # We need to generate the moves of all colours however since the King cannot put himself into check
        moves, captures = ChessBoard.get_moves_captures(self.chess_board.cube, colour)

        # If this player is under check, they can only do moves that get them out of the check situation
        if self.is_checked[player_num]:
            pass


        # for move in moves:
        #     print(f"{ChessBoard.get_pos_code(move)} -> {[ChessBoard.get_pos_code(m) for m in moves[move]]}")
        # for capture in captures:
        #     print(f"{ChessBoard.get_pos_code(capture)} -> {[ChessBoard.get_pos_code(c) for c in captures[capture]]}")


        # send the observation to the player and receive the corresponding action
        # The action is only allowed to be a move
        hash_val = self.chess_board.cube.data.tobytes()
        state_repetition = self.state_repetition_map[hash_val] if hash_val in self.state_repetition_map else 0
        action = player.send_action(player.receive_observation(BoardState(self.chess_board.cube, colour, moves, captures, self.no_progress, state_repetition)))
        if (type(action) == str):
            action = self._read_recorded_move(action)
        from_pos, to_pos = action

        # check if the action is legal
        if (from_pos not in moves) and (from_pos not in captures):
            raise IllegalActionException("This piece does not exist")
        elif (from_pos in moves and to_pos not in moves[from_pos]) and (from_pos in captures and to_pos not in captures[from_pos]):
            raise IllegalActionException("The given action is not part of the currently legal moves or captures.")

        

        # Implement the action on the chess board
        from_figure_colour = FIGURE_ID_MAP[self.chess_board[from_pos]] # Get the figure standing at the from position
        to_figure_colour = FIGURE_ID_MAP[self.chess_board[to_pos]] if self.chess_board[to_pos] != 0 else None # Get the figure standing at the to position
        ChessBoard.move(self.chess_board.cube, from_pos, to_pos)


        # render_board_ascii(self.chess_board.cube)


        # Update the no progress rule
        if not message: # The game has not yet ended
            self.no_progress += 1
            if from_figure_colour[0] == Pawn:
                self.no_progress = 0
            elif from_pos in captures and to_pos in captures[from_pos]:
                self.no_progress = 0
            elif self.no_progress > 50:
                self.is_checkmate[player_num] = True
                self.is_checkmate[enemy_player_num] = True
                player.receive_reward(0, self.move_history)
                enemy_player.receive_reward(0, self.move_history)
                message = "No progress has been achieved the last 50 turns (No pawn moved & no piece captured)." # Draw


        # If the same state was repeated three times, then this will result in an automatic draw
        # Update the board state repetition map
        if not message: # The game has not yet ended
            hash_val = self.chess_board.cube.data.tobytes()
            if hash_val in self.state_repetition_map:
                self.state_repetition_map[hash_val] += 1
            else:
                self.state_repetition_map[hash_val] = 1
            # Check if threefold repetition rule applies
            if self.state_repetition_map[hash_val] >= 3:
                self.is_checkmate[player_num] = True
                self.is_checkmate[enemy_player_num] = True
                player.receive_reward(0, self.move_history)
                enemy_player.receive_reward(0, self.move_history)
                message = "The same board position has been repeated for the third time." # Draw


        # Generate next moves of white and black pieces and assign them to either this player or the enemy player based on colour
        white_next_moves_captures, black_next_moves_captures = ChessBoard.get_moves_captures(self.chess_board.cube)
        this_player_next_moves_captures = [black_next_moves_captures, None, white_next_moves_captures][1+colour]
        enemy_player_next_moves_captures = [black_next_moves_captures, None, white_next_moves_captures][1+enemy_colour]

        if not message: # The game has not yet ended
            # Determine whether we have a checkmate
            if self.is_checked[enemy_player_num]:
                # The enemy player is under check
                # Simply check whether the enemy king has been captured i.e. if the king still stands on the board
                if not np.any(self.chess_board.cube == (King.id * enemy_colour)):
                    self.is_checked[player_num] = False
                    self.is_checkmate[player_num] = True
                    player.receive_reward(1, self.move_history)
                    enemy_player.receive_reward(-1, self.move_history)
                    message = f"Checkmate - '{player.name}' ({Colour.string(colour)}) has captured the enemy's king"


        # Determine whether we (still) have a check situation
        if not message: # The game has not yet ended
            this_player_next_moves, this_player_next_captures = this_player_next_moves_captures
            enemy_king_position = np.where(self.chess_board.cube==(King.id * enemy_colour))
            try:
                enemy_king_position = (enemy_king_position[0][0], enemy_king_position[1][0], enemy_king_position[2][0])
                is_enemy_under_check = False
                for ally_piece in this_player_next_captures:
                    if enemy_king_position in this_player_next_captures[ally_piece]:
                        is_enemy_under_check = True
                        break
                self.is_checked[enemy_player_num] = is_enemy_under_check
            except:
                print("FUUUUUUUUCK")
                render_board_ascii(self.chess_board.cube)
            

        # Check if the enemy player is able to do any moves on their next turn
        if not message: # The game has not yet ended
            if not enemy_player_next_moves_captures[0] and not enemy_player_next_moves_captures[1]:
                if self.is_checked[enemy_player_num]:
                    self.is_checked[enemy_player_num] = False
                    self.is_checkmate[enemy_player_num] = True
                    player.receive_reward(1, self.move_history)
                    enemy_player.receive_reward(-1, self.move_history)
                    message = f"'{enemy_player.name}' ({Colour.string(enemy_colour)}) is checked and does not have any available moves"
                else:
                    self.is_checkmate[player_num]       = True
                    self.is_checkmate[enemy_player_num] = True
                    player.receive_reward(0, self.move_history)
                    enemy_player.receive_reward(0, self.move_history)
                    message = f"'{enemy_player.name}' ({Colour.string(enemy_colour)}) is not checked and does not have any available moves - automatic stalemate"


        # Record the move in the move history
        self._record_move(action, from_figure_colour, to_figure_colour)
        print(f"Total Moves: {('('+str(len(self.move_history))+')').ljust(5)} | Most recent moves: ", " <-- ".join([hist.center(14, ' ') for hist in self.move_history[-1: -6: -1]]))

        if message:
            return message


        # Reward of moves generally is 0 - Reward of win is +1 - Reward of loss is -1 - Reward of draw is 0
        player.receive_reward(0, self.move_history)


    def _record_move(self, action, from_figure_id, to_figure_id):
        from_figure, from_colour = from_figure_id
        move_capture_sign = "-" if to_figure_id == None else "x"
        s = f"{(from_figure.name[1+from_colour])}:{ChessBoard.get_pos_code(action[0])}{move_capture_sign}{ChessBoard.get_pos_code(action[1])}"
        if all(self.is_checked):
            s += "++"
        elif any(self.is_checked):
            s += "+"
        elif all(self.is_checkmate):
            s += "= ½-½"
        elif any(self.is_checkmate):
            s += "#"
            if self.is_checkmate[0]:
                s += " 0-1"
            else:
                s += " 1-0"
        self.move_history.append(s)

    def _read_recorded_move(self, record):
        figure_name, action = record.split(':')
        from_pos = action[:3]
        to_pos = action[4:7]
        return (ChessBoard.get_pos_coord(from_pos), ChessBoard.get_pos_coord(to_pos))
