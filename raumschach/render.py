import numpy as np
import math

from board import ChessBoard, INITIAL_5_5_BOARD_SETUP
from figures import Colour

# print("In module products __package__, __name__ ==", __package__, __name__)



def render_ascii(chess_board):
    cb = chess_board
    s = np.array(cb.cube)
    for pos in [(i, j, k) for i in range(cb.size) for j in range(cb.size) for k in range(cb.size)]:
        if s[pos] in cb.figure_id_map:
            figure, colour = cb.figure_id_map[s[pos]]
            s[pos] = ord(figure.name[1]) if colour == Colour.WHITE else ord(figure.name[2])

    
    
    mask = np.full((3+cb.size, 3+len(" ".join(cb.size*"."))), ord(' '), np.int32)
    mask[-3: -(3+cb.size): -1, 0] = np.arange(ord('1'), ord('1')+cb.size, 1, dtype=np.byte)
    mask[2+cb.size, 2: 2+len(" ".join(cb.size*".")): 2] = np.arange(ord('a'), ord('a')+cb.size, 1, dtype=np.byte)
    mask[0, 1] = ord('╔')
    mask[0, 2+len(" ".join(cb.size*"."))] = ord('╗')
    mask[1+cb.size, 1] = ord('╚')
    mask[1+cb.size, 2+len(" ".join(cb.size*"."))] = ord('╝')
    mask[0, 2: 2+len(" ".join(cb.size*"."))] = ord('═')
    mask[0, 2+math.floor(len(" ".join(cb.size*"."))/2)] = ord('╩')
    mask[1+cb.size, 2: 2+len(" ".join(cb.size*"."))] = ord('═')
    mask[1: 1+cb.size, 1] = ord('║')
    mask[1: 1+cb.size, 2+len(" ".join(cb.size*"."))] = ord('║')

    h_dist = mask.shape[1] + 3
    v_dist = 3

    canvas = np.full((1+mask.shape[0]+(cb.size-1)*v_dist, mask.shape[1]+(cb.size-1)*h_dist), ord(' '), np.int32)

    for i in range(cb.size):
        plane_mask = np.array(mask)
        plane_mask[1: cb.size+1, 2: len(" ".join(cb.size*"."))+2: 2] = np.flipud(s[(cb.size-1)-i])
        canvas[1+(i*v_dist): 1+mask.shape[0]+(i*v_dist), 0+(i*h_dist): mask.shape[1]+(i*h_dist)] = plane_mask
        canvas[i*v_dist, i*h_dist+2+math.floor(len(" ".join(cb.size*"."))/2)] = ord('A')+(cb.size-1-i)

    print(np.array2string(canvas, max_line_width=math.ceil(1.5*canvas.shape[1]), separator='', formatter={'int':lambda x: "·" if x == 0 else chr(x)}, threshold=math.ceil(1.5*canvas.shape[0]*canvas.shape[1])).replace(" [", 5*' ').replace("[[", 5*' ').replace("]", ""))

board = ChessBoard(5, INITIAL_5_5_BOARD_SETUP)

render_ascii(board)