from dataclasses import dataclass
from typing import List, Set

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

INPUT_DATA_PATH = "input_data/15.txt"

WALL_CELL = -1
WALL_CHAR = "#"

ROBOT_CHAR = "@"

OBJECT_CHAR = "O"

FREE_CELL = 0
FREE_CHAR = "."

MAP_VALUES = {
    WALL_CHAR: [WALL_CELL] * 2,
    ROBOT_CHAR: [FREE_CELL] * 2,
    FREE_CHAR: [FREE_CELL] * 2,
    OBJECT_CHAR: [FREE_CELL] * 2,
}

MOVE_DELTAS = {"^": (-1, 0), ">": (0, 1), "v": (1, 0), "<": (0, -1)}


def create_row(line: str):

    row_vals = []
    robot_col = None
    object_cols = []
    for col, ch in enumerate(line):
        row_vals.extend(MAP_VALUES[ch])

        if ch == OBJECT_CHAR:
            object_cols.append(2 * col)

        if ch == ROBOT_CHAR:
            robot_col = 2 * col

    return row_vals, robot_col, object_cols


@dataclass
class MapObject:
    location: np.ndarray

    def __hash__(self):
        return hash(tuple(self.location))

    def intersects_cell(self, cell: np.ndarray):
        cell_tup = tuple(cell)
        return cell_tup == tuple(self.location) or cell_tup == tuple(
            self.location + np.array([0, 1])
        )

    def get_move_cells(self, move: str):
        deltas = {
            "^": [(-1, 0), (-1, 1)],
            ">": [(0, 2)],
            "v": [(1, 0), (1, 1)],
            "<": [(0, -1)],
        }[move]

        return [self.location + np.array(d) for d in deltas]


@dataclass
class MoveMap:
    data: np.array
    robot_rowcol: np.array
    objects: List[MapObject]
    moved_objects: Set[MapObject] = None
    verbose: bool = False

    def reset_moved_objects(self):
        self.moved_objects = set()

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def plot(self):
        row, col = self.robot_rowcol

        plt.matshow(self.data, cmap="hot")
        plt.scatter(col, row, marker="x", color="blue")

        xs_left, ys_left, xs_right, ys_right = [], [], [], []

        # plot objects
        for obj_loc in self.objects:
            row, col = obj_loc.location

            xs_left.append(col)
            ys_left.append(row)

            xs_right.append(col + 1)
            ys_right.append(row)

        plt.scatter(xs_left, ys_left, marker="<", color="chocolate")
        plt.scatter(xs_right, ys_right, marker=">", color="chocolate")
        plt.show()

    def get_value(self, row, col):
        return self.data[row][col]

    def set_value(self, row, col, value):
        self.data[row][col] = value

    def compute_score(self):
        object_locations = np.array([obj.location for obj in self.objects])
        object_locations[:, 0] *= 100

        return np.sum(object_locations)

    def get_object(self, cell: np.ndarray):
        if self.get_value(*cell) == WALL_CELL:
            return None

        # Can be improved by grouping objects by row
        for obj in self.objects:
            if obj.intersects_cell(cell):
                return obj

        return None

    def move_robot(self, move: str):
        can_move = self.can_move_cell(self.robot_rowcol, move)

        if can_move:
            self.reset_moved_objects()
            self.move_cell(self.robot_rowcol, move)

        if self.verbose:
            self.plot()

    def move_cell(self, row_col: np.ndarray, move: str):
        """
        This should only be called when the can_move_cell has returned True for the row_col cell.
        """
        i_am_robot = tuple(row_col) == tuple(self.robot_rowcol)

        if i_am_robot:
            neighbor = row_col + np.array(MOVE_DELTAS[move])
            self.move_cell(neighbor, move)

            self.robot_rowcol = neighbor

        matched_obj = self.get_object(row_col)

        if matched_obj is not None and matched_obj not in self.moved_objects:
            cells_to_move = matched_obj.get_move_cells(move)

            # move dependent cells
            for c in cells_to_move:
                self.move_cell(c, move)

            # move current object
            matched_obj.location += np.array(MOVE_DELTAS[move])

            self.moved_objects.add(matched_obj)

    def can_move_cell(self, row_col: np.ndarray, move: str):
        """
        Check if the given move can be applied to the given cell.
        """
        i_am_robot = tuple(row_col) == tuple(self.robot_rowcol)

        matched_obj = self.get_object(row_col)
        i_am_object = matched_obj is not None

        if not i_am_robot and not i_am_object:
            if self.get_value(*row_col) == FREE_CELL:
                # Nothing to modify, can move
                self.print("CAN MOVE [FREE]")
                return True

            if self.get_value(*row_col) == WALL_CELL:
                # Nothing to modify, cannot move
                self.print("CAN NOT MOVE [WALL]")
                return False

        if i_am_robot:
            # Check neighbor
            neighbor = row_col + np.array(MOVE_DELTAS[move])
            result = self.can_move_cell(neighbor, move)
            self.print(f"CAN {'' if result else 'NOT '}MOVE ROBOT")
            return result

        if i_am_object:
            cells_to_move = matched_obj.get_move_cells(move)
            can_move = True
            for cell in cells_to_move:
                can_move = can_move and self.can_move_cell(cell, move)

            self.print(f"CAN {'' if can_move else 'NOT '}MOVE OBJECT")
            return can_move


def read_input(path):
    moves = []
    map_rows = []
    robot_rowcol = None
    objects = []
    with open(path, "r") as file:
        for row_id, line in enumerate(file.readlines()):
            if WALL_CHAR in line:
                row, robot_col, object_cols = create_row(line.strip())

                map_rows.append(row)

                if robot_col is not None:
                    robot_rowcol = np.array((row_id, robot_col))

                objects.extend(
                    map(lambda col: MapObject(np.array((row_id, col))), object_cols)
                )
            else:
                moves.extend(line.strip())

    output_map = MoveMap(np.array(map_rows), robot_rowcol, objects)

    return output_map, moves


def part_2(path):
    move_map, moves = read_input(path)

    for move in tqdm(moves):
        move_map.move_robot(move)

    print(move_map.compute_score())


if __name__ == "__main__":
    part_2(INPUT_DATA_PATH)
