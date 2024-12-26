from collections import Counter
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

INPUT_DATA_PATH = "input_data/20.txt"
# INPUT_DATA_PATH = "input_data/20_test.txt"

FREE_CHAR = "."
FREE_CELL = 0

WALL_CHAR = "#"
WALL_CELL = -10

START_CHAR = "S"
END_CHAR = "E"

CHAR_TO_CELL = {
    START_CHAR: FREE_CELL,
    END_CHAR: FREE_CELL,
    FREE_CHAR: FREE_CELL,
    WALL_CHAR: WALL_CELL,
}


@dataclass
class TrackCheat:
    start_row_col: np.ndarray
    end_row_col: np.ndarray
    saving: int

    def __hash__(self):
        return hash(tuple([*self.start_row_col, *self.end_row_col]))


@dataclass
class TrackMap:
    data: np.ndarray
    start_row_col: np.ndarray
    end_row_col: np.ndarray

    def coords_are_valid(self, x, y):
        max_x, max_y = self.data.shape
        return 0 <= x < max_x and 0 <= y < max_y

    def get_value(self, x, y):
        return self.data[x][y]

    def get_neighbors(self, row, col):
        deltas = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
        for d in deltas:
            nb_coords = np.array([row, col]) + d

            if self.coords_are_valid(*nb_coords):
                yield nb_coords

    def traverse(self):
        step_map = np.copy(self.data)
        UNVISITED = -1
        step_map[step_map == FREE_CELL] = UNVISITED
        open_cells = [self.end_row_col]
        step_map[*self.end_row_col] = 0

        while open_cells:
            current = open_cells.pop(0)
            current_step = step_map[*current]

            for neighbor in self.get_neighbors(*current):
                if step_map[*neighbor] != UNVISITED:
                    # Already visited
                    continue

                if self.get_value(*neighbor) != FREE_CELL:
                    continue

                # set step value
                step_map[*neighbor] = current_step + 1

                # add to open cells
                open_cells.append(neighbor)

        return step_map

    def compute_cheats(self, step_map: np.ndarray):
        track_cells = np.argwhere(step_map != WALL_CELL)

        cheats = set()

        for cell in track_cells:
            current_value = step_map[*cell]
            # print(f"Cheating from cell {cell}, val {current_value}")

            # move to a wall neighbor
            for wall_nb in self.get_neighbors(*cell):
                if self.get_value(*wall_nb) != WALL_CELL:
                    continue

                # move to a non-wall cell with a smaller value
                for path_nb in self.get_neighbors(*wall_nb):
                    if self.get_value(*path_nb) == WALL_CELL:
                        continue

                    saving = current_value - step_map[*path_nb] - 2
                    if saving <= 0:
                        continue

                    # print(f"Reached {path_nb}, val {step_map[*path_nb]}")

                    cheats.add(TrackCheat(cell, path_nb, saving))

        # for cheat in cheats:
        #     print(cheat)

        # plt.matshow(step_map)
        # plt.show()

        cheat_savings = Counter([cheat.saving for cheat in cheats])
        print(cheat_savings)
        result = sum(
            [count for saving, count in cheat_savings.items() if saving >= 100]
        )
        return result


def create_row(line: str):
    start_col = line.find("S")
    if start_col == -1:
        start_col = None

    end_col = line.find("E")
    if end_col == -1:
        end_col = None

    return list(map(lambda ch: CHAR_TO_CELL[ch], line)), start_col, end_col


def read_input(path):
    rows, start_pos, end_pos = [], None, None
    with open(path, "r") as file:
        for row_id, line in enumerate(file.readlines()):
            row_data, start_col, end_col = create_row(line.strip())

            rows.append(row_data)
            if start_col is not None:
                start_pos = (row_id, start_col)

            if end_col is not None:
                end_pos = (row_id, end_col)

    return TrackMap(np.array(rows), np.array(start_pos), np.array(end_pos))


def part_1(path):
    track_map = read_input(path)

    steps = track_map.traverse()

    val = track_map.compute_cheats(steps)
    print(val)

    # 877054 too high
    # 1341 too high


if __name__ == "__main__":
    part_1(INPUT_DATA_PATH)
