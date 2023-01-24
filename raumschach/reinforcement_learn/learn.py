from raumschach.figures import FIGURES, Colour
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

## TODO sparsify needs to flip the tensor if the player is not white since we need to keep the symmetry

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

LOSS_STATE_VALUE = 0.0
DRAW_STATE_VALUE = 0.5
WIN_STATE_VALUE  = 1.0

def train_RL_model(cb_size):
    torch.autograd.set_detect_anomaly(True)
    # num_piece_types = len(FIGURES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LR = 1e-4 # learning rate

    consecutive_training_games = 30
    test_games = 1

    previous_networks = []
    value_net = ValueNN(cb_size, len(FIGURES), [ fig.id for fig in FIGURES ]).to(device)
    optimizer = optim.Adam(value_net.parameters(), lr=LR, amsgrad=True)
    memory = StateMemory()
    rng = np.random.default_rng()

    counter = [0, 0, 0]
    episode_counter = [0, 0, 0]
    consec_train_counter = 0
    for i in range(1000):

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
            0.15,
            0.15,
            0.25,
            0.15,
            0.15,
            0.15
        ])
        enemy_player_weights = enemy_player_weights / np.sum(enemy_player_weights) # Sum to 1
        enemy_player = rng.choice(enemy_players, p=enemy_player_weights)
        if enemy_player == "NNPlayer":
            if not previous_networks:
                enemy_player = RandomPlayer()
            else:
                enemy_net = ValueNN(cb_size, len(FIGURES), [ fig.id for fig in FIGURES ]).to(device)
                probas = np.linspace(0.01, 0.99, len(previous_networks))
                probas = probas / np.sum(probas)
                enemy_net.load_state_dict(rng.choice(previous_networks, p=probas))
                enemy_player = NNPlayer(enemy_net, memory, device)

        # enemy_player = AlphaBetaPlayer(search_depth=2, play_to_lose=True)

        ally_colour = rng.choice(np.array([Colour.WHITE, Colour.BLACK]))

        print("\n\n")
        print(f"Ally  ({Colour.string(ally_colour)}):", ally_player)
        print(f"Enemy ({Colour.string(ally_colour*-1)}):", enemy_player)

        game = None
        if ally_colour == Colour.WHITE:
            game = ChessGame(ally_player, enemy_player, cb_size)
        else:
            game = ChessGame(enemy_player, ally_player, cb_size)
            
        print("Start game...")
        win_player = game.play()
        print("Finished game...")

        if ally_colour == Colour.WHITE:
            counter[1+win_player] += 1
            episode_counter[1+win_player] += 1
        else:
            counter[2-(1+win_player)] += 1
            episode_counter[2-(1+win_player)] += 1

        print(f"Iteration  : {i:5,d} | Wins: {counter[2]} Draws: {counter[1]} Losses: {counter[0]}")
        print(f"Intra epoch:       | Wins: {episode_counter[2]} Draws: {episode_counter[1]} Losses: {episode_counter[0]}")
        consec_train_counter += 1
        if consec_train_counter >= consecutive_training_games:
            consec_train_counter = 0
            episode_counter = [0, 0, 0]
            # Save the current network so it can be compared to later
            previous_networks.append(value_net.state_dict())
            print("Optimizing...")
            optimize_RL_model(value_net, optimizer, memory, device, cb_size)


def optimize_RL_model(model, optimizer: optim.Optimizer, memory, device, cb_size, batch_size=128):
    model.train()
    input_target_lst = []
    # input_lst = []
    # target_val_lst = []
    for (board_hash, colour, state_repetition, no_progress), stats in memory.iter():
        board_a = np.reshape(np.frombuffer(board_hash, dtype=np.byte), (cb_size, cb_size, cb_size))
        input_tensor = model.sparsify_board_state(board_a, colour, state_repetition, no_progress).float().to(device)

        win_count = stats.win_count
        draw_count = stats.draw_count
        loss_count = stats.loss_count
        total_count = win_count + draw_count + loss_count
        # print(win_count, draw_count, loss_count)
        target_val = (LOSS_STATE_VALUE * loss_count + DRAW_STATE_VALUE * draw_count + WIN_STATE_VALUE * win_count) / total_count

        input_target_lst.append((input_tensor, torch.tensor(target_val, dtype=input_tensor.dtype)))

        # vals = torch.cat(stats.val_list)
        # target_vals = torch.full_like(vals, target_val)
        # val_lst += stats.val_list
        # target_val_lst += ([target_val] * len(stats.val_list))

        # criterion = nn.SmoothL1Loss()
        # loss = criterion(vals, target_vals)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()


        # for val in state_stats.val_list:
        #     win_count = state_stats.win_count
        #     draw_count = state_stats.draw_count
        #     loss_count = state_stats.loss_count
        #     total_count = win_count + draw_count + loss_count
        #     target_val = LOSS_STATE_VALUE * loss_count / total_count + DRAW_STATE_VALUE * draw_count / total_count + WIN_STATE_VALUE * win_count / total_count
    # vals = torch.cat(val_lst)
    # targets = torch.tensor(target_val_lst, dtype=vals.dtype)
    # criterion = nn.SmoothL1Loss()
    # loss = criterion(vals, targets)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # print(loss_count, draw_count, win_count, target_val)

    rng = np.random.default_rng()
    rng.shuffle(input_target_lst)
    while(input_target_lst):
        batch, input_target_lst = input_target_lst[:batch_size], input_target_lst[batch_size:]

        inputs = torch.stack([t[0] for t in batch])
        targets = torch.stack([t[1] for t in batch], dim=-1)[:, None]
        # print(targets[0])

        vals = model(inputs)

        # print(targets.shape)
        # print(vals.shape)

        criterion = nn.SmoothL1Loss()
        loss = criterion(vals, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    memory.clear()

# BATCH_SIZE = 16384 # sample size from replay buffer
# GAMMA = 0.99 # Discount factor
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 1000
# TAU = 0.005 # the update rate of the target network



# target_value_net = ValueNN(5, 10).to(device)
# target_value_net.load_state_dict(value_net.state_dict())

# # optimizer = optim.AdamW(value_net.parameters(), lr=LR, amsgrad=True)
# optimizer = optim.Adam(value_net.parameters(), lr=LR, amsgrad=True)
# memory = ReplayMemory()

# def optimize_model():
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)

#     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation). This converts batch-array of Transitions
#     # to Transition of batch-arrays.
#     batch = Transition(*zip(*transitions))

#     # Compute a mask of non-final states and concatenate the batch elements
#     # (a final state would've been the one after which simulation ended)
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                           batch.next_state)), device=device, dtype=torch.bool)
#     non_final_next_states = torch.cat([s for s in batch.next_state
#                                                 if s is not None])
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat(batch.reward)

#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken. These are the actions which would've been taken
#     # for each batch state according to policy_net
#     state_action_values = policy_net(state_batch).gather(1, action_batch)

#     # Compute V(s_{t+1}) for all next states.
#     # Expected values of actions for non_final_next_states are computed based
#     # on the "older" target_net; selecting their best reward with max(1)[0].
#     # This is merged based on the mask, such that we'll have either the expected
#     # state value or 0 in case the state was final.
#     next_state_values = torch.zeros(BATCH_SIZE, device=device)
#     with torch.no_grad():
#         next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
#     # Compute the expected Q values
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch

#     # Compute Huber loss
#     criterion = nn.SmoothL1Loss()
#     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     # In-place gradient clipping
#     # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) # Hopefully don't need it
#     optimizer.step()