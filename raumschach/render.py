import numpy as np
import math

from raumschach.figures import FIGURE_ID_MAP

BOARD_GAP_H = 2
BOARD_GAP_V = 1

OFFSET_H = 3
OFFSET_V = 3


def render_board_ascii(board_a):
    render_figure_moves_ascii(board_a, None)
    return
    size = board_a.shape[0]
    s = np.array(board_a)
    for pos in [(i, j, k) for i in range(size) for j in range(size) for k in range(size)]:
        if s[pos] in FIGURE_ID_MAP:
            figure, colour = FIGURE_ID_MAP[s[pos]]
            s[pos] = ord(figure.name[1+colour])

    hgap = BOARD_GAP_H
    vgap = BOARD_GAP_V

    mask = np.full((5+size+(size-1)*vgap, 5+size+(size-1)*hgap), ord(' '), np.int32)
    mask[-4: 1: -(vgap+1), 0] = np.arange(ord('1'), ord('1')+size, 1, dtype=np.byte)
    mask[-1, 3: -2: (hgap+1)] = np.arange(ord('a'), ord('a')+size, 1, dtype=np.byte)
    mask[0, 1] = ord('╔')
    mask[0, -1] = ord('╗')
    mask[-2, 1] = ord('╚')
    mask[-2, -1] = ord('╝')
    mask[0, 2: -1] = ord('═')
    mask[0, 3+math.floor((size+(size-1)*hgap)/2)] = ord('╩')
    mask[-2, 2: -1] = ord('═')
    mask[1: -2, 1] = ord('║')
    mask[1: -2, -1] = ord('║')

    h_dist = mask.shape[1] + OFFSET_H
    v_dist = OFFSET_V

    canvas = np.full((1+mask.shape[0]+(size-1)*v_dist, mask.shape[1]+(size-1)*h_dist), ord(' '), np.int32)

    for i in range(size):
        plane_mask = np.array(mask)
        plane_mask[2: 2+size+(size-1)*vgap: (vgap+1), 3: 3+size+(size-1)*hgap: (hgap+1)] = np.flipud(s[(size-1)-i])

        canvas[1+(i*v_dist): 1+mask.shape[0]+(i*v_dist), 0+(i*h_dist): mask.shape[1]+(i*h_dist)] = plane_mask
        canvas[i*v_dist, i*h_dist+3+math.floor((size+(size-1)*hgap)/2)] = ord('A')+(size-1-i)

    print('\n' + np.array2string(canvas, max_line_width=math.ceil(1.5*canvas.shape[1]), separator='', formatter={'int':lambda x: "·" if x == 0 else chr(x)}, threshold=math.ceil(1.5*canvas.shape[0]*canvas.shape[1])).replace(" [", 5*' ').replace("[[", 5*' ').replace("]", "") + '\n')



# TODO Change this to fit the new move style
def render_figure_moves_ascii(board_a, moves):
    size = board_a.shape[0]
    s = np.array(board_a)

    if moves:
        passives, captures = moves
        passive_pieces, passive_coords = list(zip(*passives))
        capture_pieces, capture_coords = list(zip(*captures))
    else:
        passive_pieces, passive_coords, capture_pieces, capture_coords = [], [], [], []

    for pos in [(i, j, k) for i in range(size) for j in range(size) for k in range(size)]:
        if s[pos] in FIGURE_ID_MAP:
            figure, colour = FIGURE_ID_MAP[s[pos]]
            s[pos] = ord(figure.name[1+colour])
            if pos in capture_coords:
                s[pos] = -s[pos]
    for passive_pos in passive_coords:
        s[passive_pos] = ord('X')
                

    hgap = BOARD_GAP_H
    vgap = BOARD_GAP_V

    mask = np.full((5+size+(size-1)*vgap, 5+size+(size-1)*hgap), ord(' '), np.int32)
    mask[-4: 1: -(vgap+1), 0] = np.arange(ord('1'), ord('1')+size, 1, dtype=np.byte)
    mask[-1, 3: -2: (hgap+1)] = np.arange(ord('a'), ord('a')+size, 1, dtype=np.byte)
    mask[0, 1] = ord('╔')
    mask[0, -1] = ord('╗')
    mask[-2, 1] = ord('╚')
    mask[-2, -1] = ord('╝')
    mask[0, 2: -1] = ord('═')
    mask[0, 3+math.floor((size+(size-1)*hgap)/2)] = ord('╩')
    mask[-2, 2: -1] = ord('═')
    mask[1: -2, 1] = ord('║')
    mask[1: -2, -1] = ord('║')

    h_dist = mask.shape[1] + OFFSET_H
    v_dist = OFFSET_V

    canvas = np.full((1+mask.shape[0]+(size-1)*v_dist, mask.shape[1]+(size-1)*h_dist), ord(' '), np.int32)

    for i in range(size):
        plane_mask = np.array(mask)
        plane_mask[2: 2+size+(size-1)*vgap: (vgap+1), 3: 3+size+(size-1)*hgap: (hgap+1)] = np.flipud(s[(size-1)-i])

        canvas[1+(i*v_dist): 1+mask.shape[0]+(i*v_dist), 0+(i*h_dist): mask.shape[1]+(i*h_dist)] = plane_mask
        canvas[i*v_dist, i*h_dist+3+math.floor((size+(size-1)*hgap)/2)] = ord('A')+(size-1-i)

    capture_pos = np.asarray((canvas < 0).nonzero()).T
    for pos in [ (p[0], p[1]) for p in capture_pos ]:
        row, col = pos
        canvas[pos] = -canvas[pos]
        if canvas[row+1, col] == ord(' '):
            if canvas[row+2, col] < 0:
                canvas[row+1, col] = ord('┼')
            else:
                canvas[row+1, col] = ord('┴')
        if canvas[row-1, col] == ord(' '):
            if canvas[row-2, col] < 0:
                canvas[row-1, col] = ord('┼')
            else:
                canvas[row-1, col] = ord('┬')
        if canvas[row, col+1] == ord(' '):
            if canvas[row, col+2] < 0:
                canvas[row, col+1] = ord('┼')
            else:
                canvas[row, col+1] = ord('┤')
        if canvas[row, col-1] == ord(' '):
            if canvas[row, col-2] < 0:
                canvas[row, col-1] = ord('┼')
            else:
                canvas[row, col-1] = ord('├')


    print('\n' + np.array2string(canvas, max_line_width=math.ceil(1.5*canvas.shape[1]), separator='', formatter={'int':lambda x: "·" if x == 0 else chr(x)}, threshold=math.ceil(1.5*canvas.shape[0]*canvas.shape[1])).replace(" [", 5*' ').replace("[[", 5*' ').replace("]", "") + '\n')
