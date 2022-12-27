from torch import cuda
from torch import nn
import numpy as np
import torch

from raumschach.board_state import BoardState
from raumschach.figures import FIGURES
from raumschach.players.player import Player


class MoveValueClassifierPlayer(Player):

    train: bool
    device: str
    model: nn.Module
    fig_id_map: dict

    def __init__(self, cb_size: int, num_piece_types: int, is_training: bool=True, use_cuda: bool=True):
        super().__init__()
        self.train = is_training
        self.device = "cuda" if (use_cuda and cuda.is_available()) else "cpu"
        self.model = CombinedValueClassifierNN(cb_size, num_piece_types)
        self.model.to(self.device)
        self.fig_id_map = dict()
        for i in range(len(FIGURES)):
            self.fig_id_map[FIGURES[i].id] = i

    def send_action(self, board_state: BoardState):
        move = None
        moves = None
        if board_state.passives.shape[0] > 0:
            move = board_state.passives[0]
            moves = board_state.passives[0:1]
        else:
            move = board_state.captures[0]
            moves = board_state.captures[0:1]
        
        tensor = self.sparsify(board_state, moves)
        # print(tensor)
        # print(tensor.shape)
        self.model.forward(tensor)

        return move

    def receive_reward(self, reward_value, move_history):
        pass

    def sparsify(self, board_state: BoardState, moves: np.ndarray):
        len_figures = len(FIGURES)
        ally_colour = board_state.colour
        enemy_colour = ally_colour*-1
        cb_arrays_shape = tuple([len_figures] + list(board_state.board_a.shape))

        sparse_arrays = []

        ally_arrays = np.zeros(cb_arrays_shape, dtype=np.single)
        enemy_arrays = np.zeros(cb_arrays_shape, dtype=np.single)
        board_state_arrays = np.zeros(tuple([2] + list(board_state.board_a.shape)), np.single)
        from_move_arrays = np.zeros(cb_arrays_shape, dtype=np.single)
        to_move_arrays = np.zeros(cb_arrays_shape, dtype=np.single)

        for figure in FIGURES:
            # print(ally_arrays.shape, self.fig_id_map[figure.id], ally_arrays[self.fig_id_map[figure.id]].shape, figure.id*ally_colour, (board_state.board_a == figure.id*ally_colour).shape)
            ally_arrays[self.fig_id_map[figure.id]][board_state.board_a == figure.id*ally_colour] = 1
            enemy_arrays[self.fig_id_map[figure.id]][board_state.board_a == figure.id*enemy_colour] = 1

        board_state_arrays[0] = board_state.state_repetition_count/3
        board_state_arrays[1] = board_state.no_progress_count/50

        for move in moves:
            from_move_arrays = np.zeros(cb_arrays_shape, dtype=np.single)
            to_move_arrays = np.zeros(cb_arrays_shape, dtype=np.single)
            from_move_arrays[self.fig_id_map[move[0]]][move[2]][move[3]][move[4]] = 1.0
            to_move_arrays[self.fig_id_map[move[1]]][move[5]][move[6]][move[7]] = 1.0
            sparse_array = np.concatenate((ally_arrays, enemy_arrays, board_state_arrays, from_move_arrays, to_move_arrays), axis=0)
            sparse_arrays.append(sparse_array)

        
        return torch.from_numpy(np.stack(sparse_arrays, axis=0)).to(self.device)

class CombinedValueClassifierNN(nn.Module):

    cb_size: int
    num_piece_types: int
    
    def __init__(self, cb_size: int, num_piece_types: int):
        super().__init__()
        self.cb_size = cb_size
        self.num_piece_types = num_piece_types

        in_features = ((self.num_piece_types*4)+2)*(self.cb_size**3)

        self.flatten = nn.Flatten()
        self.preSplit = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Tanh(),
            nn.Linear(in_features, self.num_piece_types*(self.cb_size**3)),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.num_piece_types*(self.cb_size**3), self.num_piece_types*(self.cb_size**3)),
            nn.Tanh(),
            nn.Linear(self.num_piece_types*(self.cb_size**3), self.cb_size**3),
            nn.Sigmoid(),
            nn.Linear(self.cb_size**3, self.cb_size),
            nn.Sigmoid(),
            nn.Linear(self.cb_size, 1),
            nn.Sigmoid()
        )

        self.value_function = nn.Sequential(
            nn.Linear(self.num_piece_types*(self.cb_size**3), self.num_piece_types*(self.cb_size**3)),
            nn.Tanh(),
            nn.Linear(self.num_piece_types*(self.cb_size**3), self.cb_size**3),
            nn.ReLU(),
            nn.Linear(self.cb_size**3, self.cb_size),
            nn.ReLU(),
            nn.Linear(self.cb_size, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.preSplit(x)
        class_proba = self.classifier(x)
        move_val = self.value_function(x)
        return (move_val, class_proba)