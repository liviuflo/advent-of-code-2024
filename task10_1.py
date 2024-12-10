from dataclasses import dataclass

import numpy as np

INPUT_DATA_PATH = "input_data/10.txt"


@dataclass
class Map:
    data: np.ndarray

    def coords_are_valid(self, x, y):
        max_x, max_y = self.data.shape
        return 0 <= x < max_x and 0 <= y < max_y

    def get_valid_neighbors(self, xy):
        deltas = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
        for d in deltas:
            nb_coords = xy + d

            if self.coords_are_valid(*nb_coords):
                yield nb_coords

    def get_value(self, x, y):
        return self.data[x][y]

    def navigate_from(self, coords):
        to_visit = [coords]

        reached_ends = set()

        while to_visit:
            current = to_visit.pop(0)
            current_val = self.get_value(*current)
            if current_val == 9:
                reached_ends.add(tuple(current))
                continue

            for nb in self.get_valid_neighbors(current):
                if self.get_value(*nb) == current_val + 1:
                    to_visit.append(nb)

        return len(reached_ends)


def read_map(path):
    with open(path, "r") as file:
        lines = [list(map(int, line.strip())) for line in file.readlines()]
        return np.array(lines)


def part_1(path):
    map_obj = Map(read_map(path))

    zero_coords = np.argwhere(map_obj.data == 0)

    score = sum([map_obj.navigate_from(coords) for coords in zero_coords])
    print(score)


if __name__ == "__main__":
    part_1(INPUT_DATA_PATH)
