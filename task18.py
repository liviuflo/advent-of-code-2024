from bisect import insort_left
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt

# INPUT_DATA_PATH = "input_data/18_test.txt"
# MAP_SIZE = 7

INPUT_DATA_PATH = "input_data/18.txt"
MAP_SIZE = 71

FREE_CELL = 0
OBSTACLE_CELL = 1


@dataclass
class ByteStep:
    row_col: np.ndarray
    distance_from_start: float

    @property
    def distance_to_end_h(self):
        end_cell = np.array([MAP_SIZE, MAP_SIZE]) - 1
        return np.linalg.norm(self.row_col - end_cell, 1)

    @property
    def value(self):
        return self.distance_from_start + self.distance_to_end_h

    def get_neighbors(self):
        deltas = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        for d in deltas:
            yield self.row_col + np.array(d)


@dataclass
class ByteMap:
    data_per_step: np.ndarray

    @staticmethod
    def create_from_values(values: List[Tuple[int]]):
        data = np.full((len(values) + 1, MAP_SIZE, MAP_SIZE), FREE_CELL)
        for step, (col, row) in enumerate(values):
            # copy from previous step
            data[step + 1] = data[step]

            # add new obstacle
            data[step + 1][row][col] = OBSTACLE_CELL

        return ByteMap(data)

    def is_valid_row_col(self, row, col):
        return 0 <= row < MAP_SIZE and 0 <= col < MAP_SIZE

    def plot(self, step: int):
        plt.matshow(self.data_per_step[step])
        plt.show()

    def find_shortest_path(self, step: int, visualize=False):
        destination = tuple((MAP_SIZE - 1, MAP_SIZE - 1))
        wall_map = self.data_per_step[step]
        visit_map = np.zeros_like(wall_map)

        start_step = ByteStep(np.array([0, 0]), 0)

        next_steps = [start_step]
        while next_steps:
            current = next_steps.pop(0)

            if tuple(current.row_col) == destination:

                if visualize:
                    viz_map = np.copy(visit_map)
                    viz_map[wall_map == OBSTACLE_CELL] = -1
                    plt.matshow(viz_map)
                    plt.show()

                return current.distance_from_start

            for n_row_col in current.get_neighbors():
                if not self.is_valid_row_col(*n_row_col):
                    continue

                if wall_map[*n_row_col] == OBSTACLE_CELL:
                    continue

                if visit_map[*n_row_col] != 0:
                    continue

                next_step = ByteStep(n_row_col, current.distance_from_start + 1)
                insort_left(next_steps, next_step, key=lambda step: step.value)

                visit_map[*n_row_col] = current.distance_from_start + 1


def read_input(path):
    with open(path, "r") as file:
        return [tuple(map(int, line.strip().split(","))) for line in file.readlines()]


def part_1(path):
    values = read_input(path)

    cell_map = ByteMap.create_from_values(values)

    print(cell_map.find_shortest_path(1024, visualize=True))


def binary_search(left: int, right: int, checker):
    """Find smallest x between left and right for which checker(x) is False."""
    res_left = checker(left)
    res_right = checker(right)
    if res_left == res_right:
        return None

    if res_left and not res_right and right == left + 1:
        return right

    if left < right:
        mid = (left + right) // 2
        solution_left = binary_search(left, mid, checker)
        solution_right = binary_search(mid, right, checker)

        if solution_left is not None:
            return solution_left

        if solution_right is not None:
            return solution_right


def part_2(path):
    values = read_input(path)

    cell_map = ByteMap.create_from_values(values)

    def path_exists(byte_num: int):
        return cell_map.find_shortest_path(byte_num) is not None

    # Perform binary search
    res = binary_search(0, len(values), path_exists)
    print(values[res - 1])


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
