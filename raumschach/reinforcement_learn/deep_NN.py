import torch
from torch import nn
import numpy as np

from raumschach.board_state import BoardState, SimpleBoardState
from raumschach.figures import Colour

class ValueNN(nn.Module):

    def __init__(self, cb_size: int, num_piece_types: int, piece_id_lst: list) -> None:
        super().__init__()

        self.cb_size = cb_size
        self.piece_ids = piece_id_lst
        self.piece_id_index_map = dict()
        for i in range(len(self.piece_ids)):
            self.piece_id_index_map[self.piece_ids[i]] = i
        self.num_piece_types = num_piece_types

        board_features = (self.num_piece_types*2)*(self.cb_size**3)

        self.convolve = nn.Sequential(
            nn.Conv3d(num_piece_types*2, num_piece_types*2, 1),
            nn.LeakyReLU(),
            nn.Conv3d(num_piece_types*2, num_piece_types**2, cb_size, padding=cb_size//2),
            nn.LeakyReLU(),
            nn.Conv3d(num_piece_types**2, num_piece_types*2, cb_size, padding=cb_size//2),
            nn.LeakyReLU()
        )

        self.flatten = nn.Flatten()

        self.combine = nn.Bilinear(board_features, 2, board_features)

        self.linear_layers = nn.Sequential(
            nn.Linear(board_features, self.cb_size**3),
            nn.LeakyReLU(),
            nn.Linear(self.cb_size**3, self.cb_size**3),
            nn.LeakyReLU(),
            # nn.Linear(self.cb_size**3, self.cb_size**3),
            # nn.LeakyReLU(),
            nn.Linear(self.cb_size**3, 1)
        )

    def forward(self, sparse_board, meta_data):
        conv = self.convolve(sparse_board)
        flat_board = self.flatten(conv)
        x = self.combine(flat_board, meta_data)
        return self.linear_layers(x)

    def get_board_state_value_function(self, device):

        def get_board_state_value(simple_board_state: 'SimpleBoardState'):
            self.eval()
            sparse_board, meta_data = self.sparsify_board_state(simple_board_state.board_a, simple_board_state.colour, simple_board_state.state_repetition_count, simple_board_state.no_progress_count)
            # print(sparse_board.shape, meta_data.shape)
            sparse_board, meta_data = sparse_board[None, :].to(device).float(), meta_data[None, :].to(device).float()
            # print(sparse_board.shape, meta_data.shape)
            return self(sparse_board, meta_data).item()
    
        return get_board_state_value

    def get_board_state_moves_value_function(self, device):
        
        def get_values(simple_board_state: 'SimpleBoardState', moves: np.ndarray):
            self.eval()
            board_input, meta_data_input = self.sparsify_moves(simple_board_state, moves)
            board_input, meta_data_input = board_input.to(device).float(), meta_data_input.to(device).float()
            vals = self(board_input, meta_data_input)
            return vals[:, 0].detach().cpu().numpy()

        return get_values



    def sparsify_moves(self, board_state: 'BoardState', moves):
        bitmap_shape = (self.num_piece_types, self.cb_size, self.cb_size, self.cb_size)

        # ally_bitmap = np.zeros(bitmap_shape, dtype=np.single)
        # enemy_bitmap = np.zeros(bitmap_shape, dtype=np.single)


        # sparse_flattened_arrays = np.zeros((moves.shape[0], np.product(ally_bitmap.shape) + np.product(enemy_bitmap.shape) + 2), dtype=np.single)
        sparse_board_a = np.zeros((moves.shape[0], self.num_piece_types*2, self.cb_size, self.cb_size, self.cb_size), dtype=np.single)
        meta_data_a = np.zeros((moves.shape[0], 2), dtype=np.single)
        for i in range(moves.shape[0]):
            move = moves[i]
            post_move_board_state = BoardState.move(board_state, move, simple=True)
            board_a = post_move_board_state.board_a
            state_repetition = post_move_board_state.state_repetition_count
            no_progress = post_move_board_state.no_progress_count
            sparse_board_a[i], meta_data_a[i] = self.sparsify_board_state(board_a, board_state.colour, state_repetition, no_progress)
            # sparse_flattened_arrays[0] = self.sparsify_board_state(board_a, board_state.colour, state_repetition, no_progress)

        return torch.from_numpy(sparse_board_a), torch.from_numpy(meta_data_a)

    def sparsify_board_state(self, board_a, colour, state_repetition, no_progress):
        if colour == Colour.BLACK:
            board_a = np.flip(board_a, axis=(0, 1)) # Flipping the array to achieve symmetry for both black and white pieces
        ally_colour = colour
        enemy_colour = ally_colour*-1
        bitmap_shape = (self.num_piece_types, self.cb_size, self.cb_size, self.cb_size)

        ally_bitmap = np.zeros(bitmap_shape)
        enemy_bitmap = np.zeros(bitmap_shape)

        for j in (range(len(self.piece_ids))):
            id = self.piece_ids[j]
            ally_bitmap[j][board_a == id*ally_colour] = 1
            enemy_bitmap[j][board_a == id*enemy_colour] = 1
        flat_ally_bitmap = ally_bitmap.flatten()
        flat_enemy_bitmap = enemy_bitmap.flatten()
        return torch.from_numpy(np.concatenate((ally_bitmap, enemy_bitmap))), torch.from_numpy(np.array([state_repetition, no_progress]))
        # return torch.from_numpy(np.concatenate((flat_ally_bitmap, flat_enemy_bitmap, [state_repetition, ], [no_progress, ])))

    def move_sparse(self, simple_board_state: 'SimpleBoardState', move, sparse_board_a, meta_data: torch.Tensor):
        new_simple_board_state = BoardState.move(simple_board_state, move, simple=True)

        old_ally_a, old_enemy_a = torch.split(sparse_board_a.clone(), sparse_board_a.shape[0]//2, dim=0)
        old_enemy_a[:, move[5], move[6], move[7]] = 0 # It doesn't really matter if this is a capture move or not, there cannot be an enemy piece where the piece moves to
        old_ally_a[self.piece_id_index_map[abs(move[0])], move[2], move[3], move[4]] = 0
        old_ally_a[self.piece_id_index_map[abs(move[1])], move[5], move[6], move[7]] = 1

        moved_sparse_board_a = torch.cat((old_enemy_a, old_ally_a), dim=0).flip(dims=(1, 2))

        moved_meta_data = torch.from_numpy(np.array([new_simple_board_state.state_repetition_count, new_simple_board_state.no_progress_count]))

        return new_simple_board_state, moved_sparse_board_a, moved_meta_data

    @staticmethod
    def reconstruct_chessboard(cb_size, board_bytes):
        return np.reshape(np.frombuffer(board_bytes, dtype=np.byte), (cb_size, cb_size, cb_size))