from dataclasses import dataclass
from enum import Enum

import numpy as np
from matplotlib import pyplot as plt

INPUT_DATA_PATH = "input_data/06.txt"

FREE_CELL = 0
OCCUPIED_CELL = -1
VISITED_CELL = 1


char_to_int = {".": FREE_CELL, "#": OCCUPIED_CELL}


class Orientation(Enum):
    up = 0
    right = 1
    down = 2
    left = 3

    def get_delta(self):
        return {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}[self.value]

    def get_rotated(self):
        return Orientation((self.value + 1) % 4)


START_POSITION_CHAR = "^"
START_ORIENTATION = Orientation.up


@dataclass
class Position:
    row_col: np.ndarray
    orientation: Orientation

    def get_next_row_col(self):
        return self.row_col + np.array(self.orientation.get_delta())


class MapAgent:
    def __init__(self, input_data_path):
        self.map, self.start_pos = read_input(input_data_path)
        self.current_pos = self.start_pos
        self.cells_visited = 0
        self.mark_current_cell_as_visited()

    def mark_current_cell_as_visited(self):
        if not self.current_pos_is_in_grid():
            return

        row, col = self.current_pos.row_col

        current_value = self.map[row][col]
        if current_value == VISITED_CELL:
            # Already marked
            return

        assert current_value == FREE_CELL, (row, col)
        self.map[row][col] = VISITED_CELL
        self.cells_visited += 1

    def current_pos_is_in_grid(self):
        width, height = self.map.shape
        row, col = self.current_pos.row_col
        return 0 <= col < width and 0 <= row < height

    def cell_is_occupied(self, row_col):
        row, col = row_col
        return self.map[row][col] == OCCUPIED_CELL

    def step(self):
        if not self.current_pos_is_in_grid():
            print("Current position is not in grid.")
            return

        print("Current pos:", self.current_pos)

        next_row_col = self.current_pos.get_next_row_col()
        if self.cell_is_occupied(next_row_col):
            # rotate
            self.current_pos.orientation = self.current_pos.orientation.get_rotated()
        else:
            # move
            self.current_pos.row_col = next_row_col

        self.mark_current_cell_as_visited()
        return self.current_pos_is_in_grid()


def create_row(line: str):
    line: list = list(line.strip())
    start_col_id = None
    if START_POSITION_CHAR in line:
        start_col_id = line.index(START_POSITION_CHAR)
        line[start_col_id] = "."

    raw_elements = list(map(lambda x: char_to_int[x], line))
    return raw_elements, start_col_id


def read_input(path):
    grid = []
    start_pos = None
    with open(path, "r") as file:
        for row_id, line in enumerate(file.readlines()):
            row_contents, start_col_id = create_row(line)
            if start_col_id is not None:
                start_pos = Position(
                    row_col=np.array([row_id, start_col_id]),
                    orientation=Orientation.up,
                )

            grid.append(row_contents)

    return np.array(grid), start_pos


def part_1(agent: MapAgent):
    while True:
        # map_viz = np.copy(agent.map) * 100
        # pos = agent.current_pos.row_col
        # map_viz[pos[0]][pos[1]] = 200
        # plt.imshow(map_viz)
        # plt.show()
        agent.step()

        if not agent.current_pos_is_in_grid():
            break

    print(agent.cells_visited)


if __name__ == "__main__":
    map_agent = MapAgent(INPUT_DATA_PATH)
    part_1(map_agent)
