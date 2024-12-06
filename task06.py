from dataclasses import dataclass
from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

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

    def get_mark(self):
        return self.value + 1

    def clone(self):
        return Orientation(self.value)


START_POSITION_CHAR = "^"
START_ORIENTATION = Orientation.up


@dataclass
class Position:
    row_col: np.ndarray
    orientation: Orientation

    def get_next_row_col(self):
        return self.row_col + np.array(self.orientation.get_delta())

    def clone(self):
        return Position(np.copy(self.row_col), self.orientation.clone())


class MapAgent:
    OUTSIDE = -1
    NEW_CELL = 0
    CROSSING = 1
    LOOP = 2

    def __init__(self, map: np.ndarray, start_pos: Position):
        self.map = np.copy(map)
        self.start_pos = start_pos
        self.current_pos = start_pos.clone()
        self.cells_visited = 0
        self.mark_current_cell_as_visited()

    def mark_current_cell_as_visited(self):
        if not self.current_pos_is_in_grid():
            return MapAgent.OUTSIDE

        row, col = self.current_pos.row_col

        current_value = self.map[row][col]
        new_value = current_value + 1

        self.map[row][col] = new_value

        if new_value > 4:
            return MapAgent.LOOP

        if current_value > 0:
            return MapAgent.CROSSING

        if current_value == 0:
            self.cells_visited += 1
            return MapAgent.NEW_CELL

    def current_pos_is_in_grid(self):
        return self.row_col_is_in_grid(self.current_pos.row_col)

    def row_col_is_in_grid(self, row_col):
        width, height = self.map.shape
        row, col = row_col
        return 0 <= col < width and 0 <= row < height

    def cell_is_occupied(self, row_col):
        if not self.row_col_is_in_grid(row_col):
            return False
        row, col = row_col
        return self.map[row][col] == OCCUPIED_CELL

    def step(self):
        if not self.current_pos_is_in_grid():
            print("Current position is not in grid.")
            return

        # print("Current pos:", self.current_pos)

        next_row_col = self.current_pos.get_next_row_col()
        if self.cell_is_occupied(next_row_col):
            # rotate
            self.current_pos.orientation = self.current_pos.orientation.get_rotated()
        else:
            # move
            self.current_pos.row_col = next_row_col

        return self.mark_current_cell_as_visited()

    def run(self):
        steps = []
        while True:
            steps.append(self.current_pos.clone())
            # print(len(steps))

            new_cell_type = self.step()

            if new_cell_type in (MapAgent.OUTSIDE, MapAgent.LOOP):
                # stop when outside or when loop has been created
                return steps, new_cell_type


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


def part_1(path: str):
    map, start_pos = read_input(path)
    agent = MapAgent(map, start_pos)

    agent.run()

    print(agent.cells_visited)


def run_is_loop(agent: MapAgent):
    _, result = agent.run()
    return result == MapAgent.LOOP


def part_2(path):
    map, start_pos = read_input(path)
    agent = MapAgent(map, start_pos)

    steps, _ = agent.run()
    steps: list[Position]

    obstacles_that_create_loops = set()

    for possible_obstacle_pos in tqdm(steps[1:]):
        new_map = np.copy(map)
        row, col = possible_obstacle_pos.row_col
        new_map[row][col] = OCCUPIED_CELL
        agent = MapAgent(new_map, start_pos)

        if run_is_loop(agent):
            obstacles_that_create_loops.add((row, col))

    print(len(obstacles_that_create_loops))


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
