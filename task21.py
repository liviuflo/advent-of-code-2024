from dataclasses import dataclass
from typing import List

import numpy as np

INPUT_DATA_PATH = "input_data/21.txt"

EMPTY_CHAR = " "

COST_STORAGE = dict()


DELTAS_AND_MOVES = {(0, 1): ">", (-1, 0): "^", (0, -1): "<", (1, 0): "v"}

MAX_LEVEL = 26  # for part 1, set this to 3

KEYPAD_PATHS = None
DIRECTIONAL_PATHS = None


@dataclass
class PadPath:
    row_col: np.ndarray
    path: str


@dataclass
class PadPaths:
    row_col: np.ndarray
    paths: List[str]


@dataclass
class InputPad:
    layout: np.ndarray

    def coords_are_valid(self, x, y):
        max_x, max_y = self.layout.shape
        return 0 <= x < max_x and 0 <= y < max_y

    def get_neighbors(self, row, col):
        iterator = DELTAS_AND_MOVES.items()

        for d, move in iterator:
            nb_coords = np.array([row, col]) + np.array(d)

            if self.coords_are_valid(*nb_coords):
                yield nb_coords, move

    def get_value(self, x, y):
        return self.layout[x][y]

    def compute_all_paths_from(self, start_pos: np.ndarray):
        path_map = dict()
        path_map[tuple(start_pos)] = set([""])

        open_cells = [tuple(start_pos)]
        while open_cells:
            current = open_cells.pop(0)
            current_paths = path_map[current]
            new_len = min(map(len, current_paths)) + 1

            for nb, move in self.get_neighbors(*current):
                if self.layout[*nb] == EMPTY_CHAR:
                    continue
                added = False
                neighbor = tuple(nb)

                existing_paths = path_map.get(neighbor, ["-" * 9])
                current_len = min(map(len, existing_paths))
                path_map.setdefault(neighbor, set())

                for path in current_paths:
                    if new_len <= current_len:
                        path_map[neighbor].add(path + move)
                        added = True

                if added:
                    open_cells.append(neighbor)

        return path_map

    def compute_all_paths(self):
        start_positions = np.argwhere(self.layout != EMPTY_CHAR)

        all_paths_dict = dict()
        # {(current char, target char): ["moves1", "moves2", ...]}

        for start_pos in start_positions:
            paths_dict = self.compute_all_paths_from(start_pos)
            for target, paths in paths_dict.items():
                source_char = self.layout[*start_pos]
                target_char = self.layout[*target]
                all_paths_dict[(source_char, target_char)] = paths

        return all_paths_dict


def compute_keypad_paths():
    layout = np.array(
        [
            ["7", "8", "9"],
            ["4", "5", "6"],
            ["1", "2", "3"],
            [EMPTY_CHAR, "0", "A"],
        ]
    )
    return InputPad(layout).compute_all_paths()


def compute_dir_paths():
    layout = np.array(
        [
            [EMPTY_CHAR, "^", "A"],
            ["<", "v", ">"],
        ]
    )
    return InputPad(layout).compute_all_paths()


def read_input(path):
    with open(path, "r") as file:
        return [line.strip() for line in file.readlines()]


def get_paths(level, start_char, target_char):
    if level == 0:
        return KEYPAD_PATHS[(start_char, target_char)]

    return DIRECTIONAL_PATHS[(start_char, target_char)]


def path_cost(level: int, path: str):
    global COST_STORAGE
    if level == MAX_LEVEL:
        return len(path)

    current = "A"
    total_cost = 0
    for char in path:
        cost_key = (level, current, char)
        if cost_key in COST_STORAGE:
            c = COST_STORAGE[cost_key]
        else:
            c = cost(*cost_key)
            COST_STORAGE[cost_key] = c

        total_cost += c
        current = char

    return total_cost


def cost(level: int, start_char: str, target_char: str):
    paths = get_paths(level, start_char, target_char)
    return min([path_cost(level + 1, path + "A") for path in paths])


def part_12(path):
    targets = read_input(path)

    res = 0
    for target in targets:
        c = path_cost(level=0, path=target)
        res += int(target[:3]) * c

    print(res)


def main():
    global KEYPAD_PATHS
    KEYPAD_PATHS = compute_keypad_paths()

    global DIRECTIONAL_PATHS
    DIRECTIONAL_PATHS = compute_dir_paths()

    part_12(INPUT_DATA_PATH)


if __name__ == "__main__":
    main()
