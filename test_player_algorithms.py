import sys
from raumschach.raumschach_game.players.algorithmic_player import AlphaBetaPlayer, MiniMaxTreeSearchPlayer
from raumschach.raumschach_game.players.basic_player import RandomPlayer

from raumschach.raumschach_test.test_functions import load_model, test_players

test_runs = 100
res_dir = "/res/"
save_dir = "/res/test/"

def run_tests(i):
    if i == 1:
        test_players(RandomPlayer(), RandomPlayer(), save_dir, "random_random.txt", num_test=test_runs)


    elif i == 2:
        test_players(RandomPlayer(), AlphaBetaPlayer(search_depth=1), save_dir, "random_alphabeta-sd1.txt", num_test=test_runs)
    elif i == 3:
        test_players(RandomPlayer(), AlphaBetaPlayer(search_depth=2), save_dir, "random_alphabeta-sd2.txt", num_test=test_runs)
    elif i == 4:
        test_players(RandomPlayer(), AlphaBetaPlayer(search_depth=3), save_dir, "random_alphabeta-sd3.txt", num_test=test_runs)


    elif i == 5:
        model, device = load_model(res_dir + "NN_RL/reward-train-1/model_365.ptm")
        plr = MiniMaxTreeSearchPlayer(search_depth=1, branching_factor=10, random_action_p=0, value_function=model.get_board_state_moves_value_function(device), value_function_name="reward-train-var-1_model-365")
        test_players(RandomPlayer(), plr, save_dir, "random_train-1-model-365-sd1-b10.txt", num_test=test_runs)
    elif i == 6:
        model, device = load_model(res_dir + "NN_RL/reward-train-1/model_365.ptm")
        plr = MiniMaxTreeSearchPlayer(search_depth=2, branching_factor=10, random_action_p=0, value_function=model.get_board_state_moves_value_function(device), value_function_name="reward-train-var-1_model-365")
        test_players(RandomPlayer(), plr, save_dir, "random_train-1-model-365-sd2-b10.txt", num_test=test_runs)
    elif i == 7:
        model, device = load_model(res_dir + "NN_RL/reward-train-1/model_365.ptm")
        plr = MiniMaxTreeSearchPlayer(search_depth=3, branching_factor=10, random_action_p=0, value_function=model.get_board_state_moves_value_function(device), value_function_name="reward-train-var-1_model-365")
        test_players(RandomPlayer(), plr, save_dir, "random_train-1-model-365-sd3-b10.txt", num_test=test_runs)


    elif i == 8:
        model, device = load_model(res_dir + "NN_RL/reward-train-2/model_206.ptm")
        plr = MiniMaxTreeSearchPlayer(search_depth=1, branching_factor=10, random_action_p=0, value_function=model.get_board_state_moves_value_function(device), value_function_name="reward-train-var-1_model-206")
        test_players(RandomPlayer(), plr, save_dir, "random_train-2-model-206-sd1-b10.txt", num_test=test_runs)
    elif i == 9:
        model, device = load_model(res_dir + "NN_RL/reward-train-2/model_206.ptm")
        plr = MiniMaxTreeSearchPlayer(search_depth=2, branching_factor=10, random_action_p=0, value_function=model.get_board_state_moves_value_function(device), value_function_name="reward-train-var-1_model-206")
        test_players(RandomPlayer(), plr, save_dir, "random_train-2-model-206-sd2-b10.txt", num_test=test_runs)
    elif i == 10:
        model, device = load_model(res_dir + "NN_RL/reward-train-2/model_206.ptm")
        plr = MiniMaxTreeSearchPlayer(search_depth=3, branching_factor=10, random_action_p=0, value_function=model.get_board_state_moves_value_function(device), value_function_name="reward-train-var-1_model-206")
        test_players(RandomPlayer(), plr, save_dir, "random_train-2-model-206-sd3-b10.txt", num_test=test_runs)


    elif i == 11:
        model, device = load_model(res_dir + "NN_RL/reward-train-3/model_163.ptm")
        plr = MiniMaxTreeSearchPlayer(search_depth=1, branching_factor=10, random_action_p=0, value_function=model.get_board_state_moves_value_function(device), value_function_name="reward-train-var-3_model-163")
        test_players(RandomPlayer(), plr, save_dir, "random_train-3-model-163-sd1-b10.txt", num_test=test_runs)
    elif i == 12:
        model, device = load_model(res_dir + "NN_RL/reward-train-3/model_163.ptm")
        plr = MiniMaxTreeSearchPlayer(search_depth=2, branching_factor=10, random_action_p=0, value_function=model.get_board_state_moves_value_function(device), value_function_name="reward-train-var-3_model-163")
        test_players(RandomPlayer(), plr, save_dir, "random_train-3-model-163-sd2-b10.txt", num_test=test_runs)
    elif i == 13:
        model, device = load_model(res_dir + "NN_RL/reward-train-3/model_163.ptm")
        plr = MiniMaxTreeSearchPlayer(search_depth=3, branching_factor=10, random_action_p=0, value_function=model.get_board_state_moves_value_function(device), value_function_name="reward-train-var-3_model-163")
        test_players(RandomPlayer(), plr, save_dir, "random_train-3-model-163-sd3-b10.txt", num_test=test_runs)


    elif i == 14:
        test_players(RandomPlayer(), AlphaBetaPlayer(search_depth=1, play_to_lose=True), save_dir, "random_alphabeta-sd1-lose.txt", num_test=test_runs)
    elif i == 15:
        test_players(RandomPlayer(), AlphaBetaPlayer(search_depth=2, play_to_lose=True), save_dir, "random_alphabeta-sd2-lose.txt", num_test=test_runs)
    elif i == 16:
        test_players(RandomPlayer(), AlphaBetaPlayer(search_depth=3, play_to_lose=True), save_dir, "random_alphabeta-sd3-lose.txt", num_test=test_runs)

if __name__ == "__main__":
    run_tests(int(sys.argv[1]))