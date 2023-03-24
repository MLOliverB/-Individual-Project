import sys
import datetime
import numpy as np
import torch
import os

from raumschach.raumschach_game.engine.game import ChessGame
from raumschach.raumschach_game.players.algorithmic_player import AlphaBetaPlayer, MiniMaxTreeSearchPlayer
from raumschach.raumschach_game.players.basic_player import ConsolePlayer, RandomPlayer
from raumschach.raumschach_test.test_functions import load_model

def game_simulator():
    print("[This is only a prototype demonstration - errors and bugs may occur if used improperly]")
    print("Select the following options via number input\n\n")

    white = player_choice("|-| Choose the White player:")
    print("\n\n")
    black = player_choice("|-| Choose the Black Player:")

    ChessGame(white, black, 5).play()

def player_choice(text):
    print(text)
    print("[1] Interactive Console Player")
    print("[2] Random Player")
    print("[3] Tree Search using a simple Piece-Value-Function")
    print("[4] Tree Search using a Neural Network value function")
    choice = int(input(">> "))

    if choice == 1:
        return ConsolePlayer()
    elif choice == 2:
        return RandomPlayer()
    elif choice == 3:
        return alphabeta_choice()
    elif choice == 4:
        return network_choice()
    else:
        print("Error: Input not recognized - please try again\n")
        return player_choice(text)

def search_depth_choice():
    print("|-| Choose the desired search depth, e.g. 1, 2, ...")
    print("    (Warning: anything over 3 can take very long to compute!)")
    depth = int(input(">> "))
    if depth < 1:
        print("Error: The search depth must be at least 1!\n")
        return search_depth_choice()
    else:
        return depth

def alphabeta_choice():
    depth = search_depth_choice()

    print("|-| Should this player have the objective to lose the game?")
    is_play_to_lose = input("(yes/*) >> ")
    if is_play_to_lose == "yes":
        return AlphaBetaPlayer(search_depth=depth, play_to_lose=True)
    else:
        return AlphaBetaPlayer(search_depth=depth, play_to_lose=False)

def network_choice():
    print("|-| Specify the path to the saved Network that you wish to use")
    net_path = input(">> ")
    model, device = load_model(net_path)
    depth = search_depth_choice()
    return MiniMaxTreeSearchPlayer(search_depth=depth, branching_factor=500, random_action_p=0, value_function=model.get_board_state_moves_value_function(device), value_function_name=net_path)

if __name__ == "__main__":
    game_simulator()