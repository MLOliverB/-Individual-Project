from collections import deque, namedtuple
import random

from raumschach.raumschach_game.engine.board_state import SimpleBoardState


# Transition = namedtuple('Transition',
#                         ('board_state', 'move', 'next_board_state', 'reward'))

State_Stats = namedtuple('State_Stats',
                        ('board_state', 'win_count', 'draw_count', 'loss_count')
                        )

# class ReplayMemory(object):

#     size: int
#     threshold_capacity: int

#     # def __init__(self, capacity) -> None:
#     #     self.memory = deque([], maxlen=capacity)

#     def __init__(self, threshold_capacity) -> None:
#         self.memory = deque([])
#         self.size = 0
#         self.threshold_capacity = threshold_capacity

#     def push(self, *args) -> None:
#         self.size += 1
#         self.memory.append(State_Stats(*args))

#     def sample(self, batch_size) -> list:
#         self.size -= batch_size
#         if self.size < 0:
#             self.size = 0
#         return random.sample(self.memory, batch_size)

class StateMemory(object):

    state_map: dict

    def __init__(self):
        self.state_map = dict()

    def clear(self):
        self.state_map = dict()

    def push(self, b: 'SimpleBoardState', win_count=0, draw_count=0, loss_count=0):
        map_tuple = (b.board_a.data.tobytes(), b.colour, b.state_repetition_count, b.no_progress_count)

        if map_tuple in self.state_map:
            prev_stats = self.state_map[map_tuple]
        else:
            prev_stats = State_Stats(b, 0, 0, 0)
        
        self.state_map[map_tuple] = State_Stats(
            prev_stats.board_state,
            win_count=prev_stats.win_count  + win_count,
            draw_count=prev_stats.draw_count + draw_count,
            loss_count=prev_stats.loss_count + loss_count
        )

    def iter(self):
        return self.state_map.items()