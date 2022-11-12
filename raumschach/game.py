import numpy as np

from raumschach.board import INITIAL_5_5_BOARD_SETUP, BoardState, ChessBoard
from raumschach.figures import FIGURE_ID_MAP, FIGURE_NAME_MAP, Colour, King, Pawn
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
        render_board_ascii(self.chess_board.cube)
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

        passives, captures = self.next_player_passives_captures if self.next_player_passives_captures != None else ChessBoard.get_passives_captures(self.chess_board.cube, colour)

        # send the observation to the player and receive the corresponding action
        # The action is only allowed to be a move
        hash_val = self.chess_board.cube.data.tobytes()
        state_repetition = self.state_repetition_map[hash_val] if hash_val in self.state_repetition_map else 0
        action = player.send_action(player.receive_observation(BoardState(self.chess_board.cube, colour, passives, captures, self.no_progress, state_repetition)))
        
        from_action, to_action = action
        (from_piece, from_coord), (to_piece, to_coord) = action

        # TODO Negative reward for doing illegal move
        # check if the action is legal
        if (from_action not in passives) and (from_action not in captures):
            raise IllegalActionException("This piece does not exist")
        elif (from_action not in passives or (from_action in passives and to_action not in passives[from_action])):
            if (from_action not in captures or (from_action in captures and to_action not in captures[from_action])):
                raise IllegalActionException("The given action is not part of the currently legal moves or captures.")

        

        # Implement the action on the chess board
        from_figure, from_colour = FIGURE_ID_MAP[self.chess_board[from_coord]] # Get the figure standing at the from position
        to_figure, to_colour = FIGURE_ID_MAP[self.chess_board[to_coord]] if self.chess_board[to_coord] != 0 else (None, None) # Get the figure standing at the to position
        ChessBoard.move(self.chess_board.cube, from_action, to_action)

        # Update the no progress rule
        if not message: # The game has not yet ended
            self.no_progress += 1
            if from_figure == Pawn:
                self.no_progress = 0
            elif from_action in captures and to_action in captures[from_action]:
                self.no_progress = 0
            elif self.no_progress > 50:
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
            hash_val = self.chess_board.cube.data.tobytes()
            if hash_val in self.state_repetition_map:
                self.state_repetition_map[hash_val] += 1
            else:
                self.state_repetition_map[hash_val] = 1
            # Check if threefold repetition rule applies
            if self.state_repetition_map[hash_val] >= 3:
                self.is_checked[player_num] = False
                self.is_checked[enemy_player_num] = False
                self.is_checkmate[player_num] = True
                self.is_checkmate[enemy_player_num] = True
                player.receive_reward(0, self.move_history)
                enemy_player.receive_reward(0, self.move_history)
                message = "The same board position has been repeated for the third time." # Draw


        if not message: # The game has not yet ended
            # Determine whether we have a checkmate
            if True or self.is_checked[enemy_player_num]:
                # The enemy player is under check
                # Simply check whether the enemy king has been captured i.e. if the king still stands on the board
                if not np.any(self.chess_board.cube == (King.id * enemy_colour)):
                    self.is_checked[player_num] = False
                    self.is_checked[enemy_player_num] = False
                    self.is_checkmate[enemy_player_num] = True
                    player.receive_reward(1, self.move_history)
                    enemy_player.receive_reward(-1, self.move_history)
                    message = f"Checkmate - '{player.name}' ({Colour.string(colour)}) has captured the enemy's king"


        # Generate next moves of white and black pieces and assign them to either this player or the enemy player based on colour
        if not message: # The game has not yet ended
            white_next_moves, black_next_moves = ChessBoard.get_passives_captures(self.chess_board.cube)
            this_p_moves = [black_next_moves, None, white_next_moves][1+colour]
            next_p_moves = [black_next_moves, None, white_next_moves][1+enemy_colour]
            self.next_player_passives_captures = next_p_moves


        # Determine whether we (still) have a check situation
        if not message: # The game has not yet ended
            this_p_captures = this_p_moves[1]
            next_p_king_pos = (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(self.chess_board.cube==(King.id * enemy_colour)))
            next_p_checked = False
            for this_p_from_idcoord in this_p_captures:
                if next_p_king_pos in [ x[1] for x in this_p_captures[this_p_from_idcoord] ]:
                    next_p_checked = True
                    break
            self.is_checked[enemy_player_num] = next_p_checked

            next_p_captures = next_p_moves[1]
            this_p_king_pos = (lambda x: (x[0][0], x[1][0], x[2][0])) (np.where(self.chess_board.cube==(King.id * colour)))
            this_p_checked = False
            for next_p_from_idcoord in next_p_captures:
                if this_p_king_pos in [ x[1] for x in next_p_captures[next_p_from_idcoord] ]:
                    this_p_checked = True
                    break
            self.is_checked[player_num] = this_p_checked
            

        # Check if the enemy player is able to do any moves on their next turn
        if not message: # The game has not yet ended
            if not next_p_moves[0] and not next_p_moves[1]:
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
        self._record_move(action, (from_figure, from_colour), (to_figure, to_colour))

        render_board_ascii(self.chess_board.cube)
        print(f"Total Moves: {('('+str(len(self.move_history))+')').ljust(5)} | Most recent moves: ", " <-- ".join([hist.center(15, ' ') for hist in self.move_history[-1: -6: -1]]))

        if message:
            return message


        # Reward of moves generally is 0 - Reward of win is +1 - Reward of loss is -1 - Reward of draw is 0
        player.receive_reward(0, self.move_history)


    def _record_move(self, action, from_figure_id, to_figure_id):
        from_figure, from_colour = from_figure_id
        to_figure, to_colour = to_figure_id
        (from_piece, from_coord), (to_piece, to_coord) = action
        move_sign = "-" if to_figure == None else "x"
        s = f"{(from_figure.name[1+from_colour])}:{ChessBoard.get_pos_code(from_coord)}{move_sign}{ChessBoard.get_pos_code(to_coord)}"
        if from_piece != to_piece:
            s += f"={FIGURE_ID_MAP[to_piece][0].name[1+from_colour]}"
        if all(self.is_checked):
            s += "++"
        elif self.is_checked[0]:
            s += "+"
        elif self.is_checked[1]:
            s += "+"
        elif all(self.is_checkmate):
            s += " ½-½"
        elif any(self.is_checkmate):
            s += "#"
            if self.is_checkmate[0]:
                s += " 0-1"
            else:
                s += " 1-0"
        self.move_history.append(s)

    @staticmethod
    def read_recorded_move(record):
        from_figure, from_colour = FIGURE_NAME_MAP[record[0]] 
        from_pos = record[2:5]
        to_pos = record[6:9]
        to_figure, to_colour = from_figure, from_colour
        if '=' in record:
            to_figure, to_colour = FIGURE_NAME_MAP[record[10]]
        from_coord = ChessBoard.get_pos_coord(from_pos)
        to_coord = ChessBoard.get_pos_coord(to_pos)
        from_tuple = (from_figure.id* from_colour, from_coord)
        to_tuple   = (to_figure.id  * to_colour,   to_coord)
        return (from_tuple, to_tuple)