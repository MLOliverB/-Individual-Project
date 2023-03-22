import os
import torch
from raumschach.raumschach_game.engine.game import ChessGame

from raumschach.raumschach_learn.learn import network_setup

def load_model(disk_path):
    rng, device, model, optimizer, memory = network_setup(5)
    model.load_state_dict(torch.load(disk_path, weights_only=True, map_location=torch.device('cpu')).to(device))
    # model = torch.load(disk_path, map_location=torch.device('cpu')).to(device)
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