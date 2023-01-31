import os
from raumschach.figures import FIGURE_ID_MAP, FIGURES, Colour
from raumschach.game import ChessGame
from raumschach.players.algorithmic_player import AlphaBetaPlayer
from raumschach.players.basic_player import RandomPlayer
from raumschach.players.neural_net_player import NNPlayer
from raumschach.reinforcement_learn.deep_NN import ValueNN
from raumschach.reinforcement_learn.replay_memory import State_Stats, StateMemory

import torch
from torch import optim
from torch import nn
import numpy as np


# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

LOSS_STATE_VALUE = 0.0
DRAW_STATE_VALUE = 0.5
WIN_STATE_VALUE  = 1.0

def network_setup(cb_size, rand_seed=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(rand_seed)

    LR = 1e-4 # learning rate

    value_net = ValueNN(cb_size, len(FIGURES), [ fig.id for fig in FIGURES ]).to(device)
    optimizer = optim.Adam(value_net.parameters(), lr=LR, amsgrad=True)
    memory = StateMemory()

    return (rng, device, value_net, optimizer, memory)

def learn_simple_value_function(cb_size):
    rng, device, model, optimizer, memory = network_setup(cb_size)
    
    vectorized_value_transform = np.vectorize((lambda x: 0 if x == 0 else FIGURE_ID_MAP[x][0].value*(x/np.abs(x))), otypes='f')

    for i in range(0, 50000):
        print(f"\n\n\nIteration {i}")
        white_player, black_player = None, None
        if i%3 == 0:
            white_player, black_player = RandomPlayer(memory=memory), RandomPlayer(memory=memory)
        elif i%3 == 1:
            white_player, black_player = AlphaBetaPlayer(search_depth=2, play_to_lose=False, memory=memory), AlphaBetaPlayer(search_depth=2, play_to_lose=True, memory=memory)
        else:
            white_player, black_player = AlphaBetaPlayer(search_depth=2, play_to_lose=True, memory=memory), AlphaBetaPlayer(search_depth=2, play_to_lose=False, memory=memory)

        game = ChessGame(white_player, black_player, cb_size)
        win_player = game.play()

        if i%50 == 0:
            input_target_lst = []
            for k, stats in memory.iter():
                b = stats.board_state
                input_tensor = model.sparsify_board_state(b.board_a, b.colour, b.state_repetition_count, b.no_progress_count).float().to(device)
                value_board = vectorized_value_transform(b.board_a)
                if np.sum(np.isinf(value_board)) > 1:
                    value_board[np.isinf(value_board)] = 0
                target_val = np.sum(value_board) * b.colour
                input_target_lst.append((input_tensor, torch.tensor(target_val, dtype=input_tensor.dtype)))

            optimize(model, optimizer, device, input_target_lst)

        if i%1000 == 0:
            dir_path = "res/NN_simple_val_func/"
            fn = f"model_{i%1000}.ptm"
            os.makedirs(dir_path, exist_ok=True)
            torch.save(model, dir_path + fn)

# def train_RL_model(cb_size, player_choice_function):

#     rng, device, value_net, optimizer, memory = network_setup(cb_size)

#     training_cycles = 1000
#     consecutive_training_games = 50
#     test_games = 30


#     counter = [0, 0, 0]
#     episode_counter = [0, 0, 0]
#     consec_train_counter = 0
#     for i in range(1000):

#         white_player, black_player = player_choice_function(rng, value_net, memory, device)
#         game = ChessGame(white_player, black_player, cb_size)
            
#         win_player = game.play()

#         if ally_colour == Colour.WHITE:
#             counter[1+win_player] += 1
#             episode_counter[1+win_player] += 1
#         else:
#             counter[2-(1+win_player)] += 1
#             episode_counter[2-(1+win_player)] += 1

#         print(f"Iteration  : {i:5,d} | Wins: {counter[2]} Draws: {counter[1]} Losses: {counter[0]}")
#         print(f"Intra epoch:       | Wins: {episode_counter[2]} Draws: {episode_counter[1]} Losses: {episode_counter[0]}")
#         consec_train_counter += 1
#         if consec_train_counter >= consecutive_training_games:
#             consec_train_counter = 0
#             episode_counter = [0, 0, 0]
#             # Save the current network so it can be compared to later
#             previous_networks.append(value_net.state_dict())
#             print("Optimizing...")
#             optimize_RL_model(value_net, optimizer, memory, device, cb_size)

def optimize(model, optimizer: optim.Optimizer, device, input_target_lst, batch_size=128):
    model.train()

    rng = np.random.default_rng()
    rng.shuffle(input_target_lst)

    while input_target_lst:
        batch, input_target_lst = input_target_lst[:batch_size], input_target_lst[batch_size:]

        inputs = torch.stack([t[0] for t in batch]).to(device)
        targets = torch.stack([t[1] for t in batch], dim=-1)[:, None].to(device)

        vals = model(inputs)

        criterion = nn.SmoothL1Loss()
        loss = criterion(vals, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# def optimize_RL_model(model, optimizer: optim.Optimizer, memory, device, cb_size, batch_size=128):
#     model.train()
#     input_target_lst = []
#     # input_lst = []
#     # target_val_lst = []
#     for (board_hash, colour, state_repetition, no_progress), stats in memory.iter():
#         board_a = ValueNN.reconstruct_chessboard(cb_size, board_hash)
#         input_tensor = model.sparsify_board_state(board_a, colour, state_repetition, no_progress).float().to(device)

#         win_count = stats.win_count
#         draw_count = stats.draw_count
#         loss_count = stats.loss_count
#         total_count = win_count + draw_count + loss_count
#         # print(win_count, draw_count, loss_count)
#         target_val = (LOSS_STATE_VALUE * loss_count + DRAW_STATE_VALUE * draw_count + WIN_STATE_VALUE * win_count) / total_count

#         input_target_lst.append((input_tensor, torch.tensor(target_val, dtype=input_tensor.dtype)))

#     rng = np.random.default_rng()
#     rng.shuffle(input_target_lst)
#     while(input_target_lst):
#         batch, input_target_lst = input_target_lst[:batch_size], input_target_lst[batch_size:]

#         inputs = torch.stack([t[0] for t in batch])
#         targets = torch.stack([t[1] for t in batch], dim=-1)[:, None]
#         # print(targets[0])

#         vals = model(inputs)

#         # print(targets.shape)
#         # print(vals.shape)

#         criterion = nn.SmoothL1Loss()
#         loss = criterion(vals, targets)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     memory.clear()


def neural_net_player_random_weighted_enemy(rng, value_net, memory, device):
    ally_player = NNPlayer(value_net, memory, device)

    enemy_players = [
        AlphaBetaPlayer(search_depth=2, play_to_lose=True),
        AlphaBetaPlayer(search_depth=1, play_to_lose=True),
        RandomPlayer(),
        AlphaBetaPlayer(search_depth=1),
        AlphaBetaPlayer(search_depth=2),
        "NNPlayer"
        ]
    enemy_player_weights = np.array([
        0.17,
        0.17,
        0.27,
        0.17,
        0.17,
        0.05
    ])
    enemy_player_weights = enemy_player_weights / np.sum(enemy_player_weights) # Sum to 1
    enemy_player = rng.choice(enemy_players, p=enemy_player_weights)
    if enemy_player == "NNPlayer":
        enemy_player = NNPlayer(value_net, memory, device)

    ally_colour = rng.choice(np.array([Colour.WHITE, Colour.BLACK]))

    if ally_colour == Colour.WHITE:
        return (ally_player, enemy_player)
    else:
        return (enemy_player, ally_player)
