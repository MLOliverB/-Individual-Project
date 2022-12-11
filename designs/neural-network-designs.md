# Design Options for Neural Networks and their implementations

## Option 1: NN that functions as value function for minimax-like algorithms

### Input:

The current chessboard state.

- A 4-dimensional tensor that encodes different aspects of the board state.
  - Binary 3D tensors for each type of Piece, for both ally pieces and enemy pieces.
  - (Probably normalized) 3D tensors filled with the no progress count and the state repetition count.

### Output:

The value of the given chessboard state

- Must be greater than or equal to zero
- Decision between:
  - bounding the value between 0 and 1.
  - open upper bound but applying log function to prioritize differentiating between very low and very high values.
  - linear open upper bound -> could lead to instability and inaccuracy when using floats.

## Option 2: NN that values a given legal move on a board state

### Input:

- A 4-dimensional tensor that encodes different aspects of the board state.
  - Binary 3D tensors for each type of Piece, for both ally pieces and enemy pieces.
  - Binary 3D tensors (very sparse) to encode the move. It will only be two single True values, one for the move origin and one for the move destination at the appropriate tensor for the given piece type that is moved. This helps better encode pawn promotions.
  - (Probably normalized) 3D tensors filled with the no progress count and the state repetition count.

### Output:

The value of the given chessboard state

- Must be greater than or equal to zero
- Decision between:
  - bounding the value between 0 and 1.
  - open upper bound but applying log function to prioritize differentiating between very low and very high values.
  - linear open upper bound -> could lead to instability and inaccuracy when using floats.