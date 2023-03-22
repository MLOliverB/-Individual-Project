# Individual Project - Building Effective Player Agents to Play Raumschach

*Individual Project (Level 4) as part of the Computing Science Bsc Hons course*

## Code Description

The code has been structured with the goal to appropriately separate concerns. Although most low-level packages are self-contained, there exists some tight coupling between certain classes and functions. This was necessary to avoid code duplication and ensure efficient execution of the code.

The codebase contains a single python package, `raumschach`. The file structure is as follows:

* `raumschach_game` - Contains all files necessary to run the Raumschach game
  * `data` - Static data used in the game engine. The `figures.py` module stores information about each chess piece along with defining constants for the two colours black and white
  * `engine` - The core game engine code. The modules within contain functions and classes to manage the game flow
    * The main entry point to start the engine is `game.py` with the `ChessGame` class
  * `players` - Stores the created player agents in different modules, separated by role and complexity
  * `vis` - Contains the module to handle visualisations. The only currently implemented visualisation is printing a given chess board to stdout
* `raumschach_learn` - Contains modules related to training the neural networks, either through self-supervision or self-play
  * Entry point functions are stored in `learn.py`
  * Since this code was created for the experimentation on a prototype, there is a noticeable amount of duplicate and redundant functions
* `raumschach_test` - Code to test two player agents against each other. The win, draw, and loss counts are recorded to file once for the players playing for either colour

## Build Instructions

### Requirements

* Python 3.10
* Packages listed in 'requirements.txt'
* Optionally, appropriate pytorch packages for cuda support need to be installed (recommended process is through anaconda)
* Tested on Windows 10 and Ubuntu

### Build steps

#### Using a Python Virtual Environment:

* Open the directory in which this README is located
* Run `python -m pip install -r requirements.txt`
* For Cuda support, machine specific packages need to be downloaded from https://pytorch.org/get-started/locally/

#### Using an Anaconda Virtual Environment:

* Check if the given `conda` environment includes `numpy` and `torch` by default
* If not, open the directory in which this README is located
* Run `conda install numpy`
* To install pytorch, use the appropriate command specified in https://pytorch.org/get-started/locally/

### Test steps

* Testing the Game Engine and Players
  * The game engine can best be tested by observing games and their correctness
  * In order to attempt to force errors, the Console Player agent is very useful
  * In the directory of this README file, run `python app.py`
  * Select the desired player agents from the console menu and observer the moves made and the move history at the end of the game
* Testing the Player Performance against a baseline
  * The networks are evaluated against the Random Player agent as a baseline
  * Edit the `test_player_algorithms.py` file
    * Set the desired number of test games through the `test_runs` variable
    * Ensure that the specified models are present in the within the `res/` directory
    * Run the desired test using `python test_player_algorithms.py {i}`, where `{i}` is a number from 1 to 16 depending on the desired test
  * **It will take a considerable amount of time to run these test games**