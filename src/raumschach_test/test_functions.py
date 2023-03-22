import os
from raumschach.game import ChessGame
from raumschach.players.algorithmic_player import MiniMaxTreeSearchPlayer
from raumschach.players.basic_player import RandomPlayer
from raumschach.players.neural_net_player import NNPlayer
from raumschach.reinforcement_learn.learn import network_setup

import torch

def load_model(disk_path):
    rng, device, _, optimizer, memory = network_setup(5)
    model = torch.load(disk_path, map_location=torch.device('cpu')).to(device)
    return model, device

def test_players(player1, player2, save_dir, fn, num_test=100):
    test_results1 = [0, 0, 0]
    test_results2 = [0, 0, 0]

    for i in range(num_test):
        game = ChessGame(player1, player2, 5)
        win_player = game.play()
        test_results1[1+win_player] += 1

    for i in range(num_test):
        game = ChessGame(player2, player1, 5)
        win_player = game.play()
        test_results2[1+win_player] += 1

    lines = [
        "White: " + str(player1),
        "Black: " + str(player2),
        "White Wins, Draws, Losses",
        f"{test_results1[2]}, {test_results1[1]}, {test_results1[0]}",
        "Black Wins, Draws, Losses",
        f"{test_results1[0]}, {test_results1[1]}, {test_results1[2]}",
        "",
        "",
        "White: " + str(player2),
        "Black: " + str(player1),
        "White Wins, Draws, Losses",
        f"{test_results2[2]}, {test_results2[1]}, {test_results2[0]}",
        "Black Wins, Draws, Losses",
        f"{test_results2[0]}, {test_results2[1]}, {test_results2[2]}",
    ]

    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir + fn, 'w') as fn:
        fn.write("\n".join(lines) + "\n")

# def test_network(disk_path, cb_size, num_test=25, tree_search=False):
#     rng, device, _, optimizer, memory = network_setup(cb_size)
#     model = torch.load(disk_path, map_location=torch.device('cpu')).to(device)

#     test_random = [0, 0, 0]

#     for i in range(num_test):

#         nn_player = None
#         if tree_search:
#             # nn_player = AlphaBetaTreeSearchPlayer(search_depth=2, value_function=model.get_board_state_value_function(device))
#             nn_player = MiniMaxTreeSearchPlayer(search_depth=2, branching_factor=10, value_function=model.get_board_state_moves_value_function(device))
#             # nn_player = MiniMaxTreeSearchPlayer(search_depth=2, value_function=None)
#         else:
#             nn_player = NNPlayer(model, device)
#         enemy = RandomPlayer()
#         # enemy = MiniMaxTreeSearchPlayer(search_depth=2, value_function=model.get_board_state_moves_value_function(device))

#         if i%2 == 0:
#             white_player, black_player = nn_player, enemy
#         else:
#             black_player, white_player = enemy, nn_player

#         game = ChessGame(white_player, black_player, cb_size)
#         win_player = game.play()

#         if i%2 == 0:
#             test_random[1+win_player] += 1
#         else:
#             test_random[2-(1+win_player)] += 1

#         print(f"Testing against random - Iteration  : {i:5,d} | Wins: {test_random[2]} Draws: {test_random[1]} Losses: {test_random[0]}")