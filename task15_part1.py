from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

INPUT_DATA_PATH = "input_data/15.txt"

WALL_CELL = -1
WALL_CHAR = "#"

ROBOT_CHAR = "@"

OBJECT_CELL = 1
OBJECT_CHAR = "O"

FREE_CELL = 0
FREE_CHAR = "."

MAP_VALUES = {
    WALL_CHAR: WALL_CELL,
    ROBOT_CHAR: FREE_CELL,
    FREE_CHAR: FREE_CELL,
    OBJECT_CHAR: OBJECT_CELL,
}

MOVE_DELTAS = {"^": [-1, 0], ">": [0, 1], "v": [1, 0], "<": [0, -1]}


def create_row(line: str):
    values = [MAP_VALUES[ch] for ch in line]

    robot_col = line.find(ROBOT_CHAR)

    return values, robot_col if robot_col != -1 else None


@dataclass
class MoveMap:
    data: np.array
    robot_rowcol: np.array

    def plot(self):
        viz_map = np.copy(self.data)
        row, col = self.robot_rowcol
        viz_map[row][col] = 2

        plt.matshow(viz_map)
        plt.show()

    def get_value(self, row, col):
        return self.data[row][col]

    def set_value(self, row, col, value):
        self.data[row][col] = value

    def compute_score(self):
        object_locations = np.argwhere(self.data == OBJECT_CELL)
        object_locations[:, 0] *= 100

        return np.sum(object_locations)

    def move_cell(self, row_col: np.ndarray, move_delta: np.ndarray):
        """
        If given cell can move in the given direction, modify the map and return True.
        Otherwise return False.
        """
        i_am_robot = tuple(row_col) == tuple(self.robot_rowcol)

        if not i_am_robot:
            if self.get_value(*row_col) == FREE_CELL:
                # Nothing to modify, can move
                return True

            if self.get_value(*row_col) == WALL_CELL:
                # Nothing to modify, cannot move
                return False

        i_am_object = self.get_value(*row_col) == OBJECT_CELL

        # Object or robot wants to move
        assert i_am_object or i_am_robot

        # Push neighbor
        neighbor = row_col + move_delta
        neighbor_has_moved = self.move_cell(neighbor, move_delta)
        if not neighbor_has_moved:
            return False

        if i_am_object:
            # Move object to next cell
            self.set_value(*neighbor, OBJECT_CELL)

            # Mark current cell as free
            self.set_value(*row_col, FREE_CELL)

            return True

        elif i_am_robot:
            # Update robot location
            self.robot_rowcol = neighbor
            return True


def read_input(path):
    moves = []
    map_rows = []
    robot_rowcol = None
    with open(path, "r") as file:
        for row_id, line in enumerate(file.readlines()):
            if WALL_CHAR in line:
                row, robot_col = create_row(line.strip())
                map_rows.append(row)

                if robot_col is not None:
                    robot_rowcol = np.array((row_id, robot_col))
            else:
                moves.extend(line.strip())

    output_map = MoveMap(np.array(map_rows), robot_rowcol)

    return output_map, moves


def part_1(path):
    move_map, moves = read_input(path)

    for move in moves:
        # move_map.plot()

        move_map.move_cell(move_map.robot_rowcol, np.array(MOVE_DELTAS[move]))

    print(move_map.compute_score())


if __name__ == "__main__":
    part_1(INPUT_DATA_PATH)
