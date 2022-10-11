import numpy as np

from board import ChessBoard, INITIAL_BOARD_SETUP

# print("In module products __package__, __name__ ==", __package__, __name__)



def render_ascii(board_state):
    s = board_state
    canvas = np.full((20, 72), ord(' '), np.int32)
    mask = np.full((8, 12), ord(' '), np.int32)
    mask[-3: -8: -1, 0] = np.arange(ord('1')+0, ord('1')+5, 1, dtype=np.byte)
    mask[7, 2: 11: 2] = np.arange(ord('a'), ord('e')+1, 1, dtype=np.byte)
    mask[0, 1] = ord('╔')
    mask[0, 11] = ord('╗')
    mask[6, 1] = ord('╚')
    mask[6, 11] = ord('╝')
    mask[0, 2: 11] = ord('═')
    mask[6, 2: 11] = ord('═')
    mask[1: 6, 1] = ord('║')
    mask[1: 6, 11] = ord('║')

    maskE = np.array(mask)
    maskE[1: 6, 2: 11: 2] = np.flipud(s[4])
    canvas[0: 8, 0: 12] = maskE

    maskD = np.array(mask)
    maskD[1: 6, 2: 11: 2] = np.flipud(s[3])
    canvas[3: 11, 15: 27] = maskD

    maskC = np.array(mask)
    maskC[1: 6, 2: 11: 2] = np.flipud(s[2])
    canvas[6: 14, 30: 42] = maskC

    maskB = np.array(mask)
    maskB[1: 6, 2: 11: 2] = np.flipud(s[1])
    canvas[9: 17, 45: 57] = maskB

    maskA = np.array(mask)
    maskA[1: 6, 2: 11: 2] = np.flipud(s[0])
    canvas[12: 20, 60: 72] = maskA

    print(np.array2string(canvas, max_line_width=24*75, separator='', formatter={'int':lambda x: "·" if x == 0 else chr(x)}, threshold=24*75).replace(" [", 5*' ').replace("[[", 5*' ').replace("]", ""))

board = ChessBoard(INITIAL_BOARD_SETUP)

render_ascii(board.cube)