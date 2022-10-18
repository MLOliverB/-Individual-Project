
from raumschach.board import INITIAL_5_5_BOARD_SETUP, BoardState, ChessBoard
from raumschach.figures import FIGURE_ID_MAP, Colour, Pawn

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
        self.is_check = [False, False]
        self.is_checkmate = [False, False]

    def play(self):
        player_num = 0
        while not any(self.is_checkmate):
            self.turn(player_num)
            player_num = (player_num + 1) % len(self.players)
        if all(self.is_checkmate):
            print("Draw!")
        else:
            for i in range(len(self.is_checkmate)):
                if self.players[i][1] == Colour.WHITE:
                    print("White wins!")
                    break
                else:
                    print("Black wins!")
                    break

    def turn(self, player_num):
        player, colour = self.players[player_num]

        # generate all possible moves for the figures of that colour
        # We need to generate the moves of all colours however since the King cannot put himself into check
        moves, captures = ChessBoard.get_moves_captures(self.chess_board.cube, colour)

        # send the observation to the player and receive the corresponding action
        # The action is only allowed to be a move
        hash_val = self.chess_board.cube.data.tobytes()
        state_repetition = self.state_repetition_map[hash_val] if hash_val in self.state_repetition_map else 0
        action = player.send_action(player.receive_observation(BoardState(self.chess_board.cube, colour, moves, captures, self.no_progress, state_repetition)))
        from_pos, to_pos = action

        # check if the action is legal
        if (from_pos not in moves) and (from_pos not in captures):
            raise IllegalActionException("This piece does not exist")
        elif (from_pos in moves and to_pos not in moves[from_pos]) and (from_pos in captures and to_pos not in captures[from_pos]):
            raise IllegalActionException("The given action is not part of the currently legal moves or captures.")

        # Update the no progress rule
        self.no_progress += 1
        if FIGURE_ID_MAP[self.chess_board[from_pos]][0] == Pawn:
            self.no_progress == 0
        elif from_pos in captures and to_pos in captures[from_pos]:
            self.no_progress == 0
        elif self.no_progress > 50:
            self.is_checkmate[player_num] = True
            self.is_checkmate[(player_num + 1) % len(self.players)] = True
            self.players[player_num][0].receive_reward(0, self.move_history[-1])
            self.players[(player_num + 1) % len(self.players)][0].receive_reward(0, self.move_history[-1])
            return "No progress has been achieved the last 50 turns (No pawn moved & no piece captured)." # Draw

        # Implement the action on the chess board
        from_figure_id = FIGURE_ID_MAP[self.chess_board[from_pos]] # Get the figure standing at the from position
        to_figure_id = FIGURE_ID_MAP[self.chess_board[to_pos]] if self.chess_board[to_pos] != 0 else None # Get the figure standing at the to position
        self.chess_board.move(from_pos, to_pos)

        # Update the board state repetition map
        hash_val = self.chess_board.cube.data.tobytes()
        if hash_val in self.state_repetition_map:
            self.state_repetition_map[hash_val] += 1
        else:
            self.state_repetition_map[hash_val] = 1
        # Check if threefold repetition rule applies
        if self.state_repetition_map[hash_val] >= 3:
            self.is_checkmate[player_num] = True
            self.is_checkmate[(player_num + 1) % len(self.players)] = True
            self.players[player_num][0].receive_reward(0, self.move_history[-1])
            self.players[(player_num + 1) % len(self.players)][0].receive_reward(0, self.move_history[-1])
            return "The same board position has been repeated for the third time." # Draw
        
        # If the same state was repeated three times, then this will result in an automatic draw

        # Determine whether the game is over (either checkmate or stalemate)

        # Determine whether we have a check situation

        # Record the move in the move history
        self._record_move(action, from_figure_id, to_figure_id)
        print(self.move_history)

        # Reward of moves generally is 0 - Reward of win is +1 - Reward of loss is -1 - Reward of draw is 0
        reward = 0

        player.receive_reward(0, self.move_history[-1])


    def _record_move(self, action, from_figure_id, to_figure_id):
        from_figure, from_colour = from_figure_id
        move_capture_sign = "-" if to_figure_id == None else "x"
        s = f"{(from_figure.name[1+from_colour])}:{ChessBoard.get_pos_code(action[0])}{move_capture_sign}{ChessBoard.get_pos_code(action[1])}"
        if all(self.is_check):
            s += "++"
        elif any(self.is_check):
            s += "+"
        elif any(self.is_checkmate):
            s += "#"
            if self.is_checkmate[0]:
                s += " 1-0"
            else:
                s += " 0-1"
        self.move_history.append(s)