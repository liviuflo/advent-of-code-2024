from bisect import insort_left
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

INPUT_DATA_PATH = "input_data/16.txt"

FREE_CELL = 0
WALL_CELL = -100

CHAR_TO_MAP_VALUE = {"#": WALL_CELL, ".": FREE_CELL, "E": FREE_CELL, "S": FREE_CELL}


def create_row(line: str):
    start_col = line.find("S")
    if start_col == -1:
        start_col = None

    end_col = line.find("E")
    if end_col == -1:
        end_col = None

    return list(map(lambda ch: CHAR_TO_MAP_VALUE[ch], line)), start_col, end_col


def read_input(path):
    rows = []
    start_pos = None
    end_pos = None
    with open(path, "r") as file:
        for row_id, line in enumerate(file.readlines()):
            row_data, start_col, end_col = create_row(line.strip())

            if start_col is not None:
                start_pos = (row_id, start_col)

            if end_col is not None:
                end_pos = (row_id, end_col)

            rows.append(row_data)

    return np.array(rows), start_pos, end_pos


ORIENTATION_TO_DELTA = [
    (0, 1),  # East
    (-1, 0),  # North
    (0, -1),  # West
    (1, 0),  # South
]

DELTA_ORIENTATION_TO_TURNS = [0, 1, 2, 1]

ORIENTATION_STR = [">", "^", "<", "v"]


@dataclass
class MapPose:
    row_col: np.ndarray
    orientation: int
    score: int

    def __repr__(self):
        return (
            f"{tuple(self.row_col)} {ORIENTATION_STR[self.orientation]} #{self.score}"
        )

    def get_neighbors(self):
        for delta_orientation in range(4):
            new_orientation = (self.orientation + delta_orientation) % 4
            extra_score = DELTA_ORIENTATION_TO_TURNS[delta_orientation] * 1000

            new_location = self.row_col + np.array(
                ORIENTATION_TO_DELTA[new_orientation]
            )
            yield MapPose(new_location, new_orientation, self.score + extra_score + 1)

    def as_tuple(self):
        return (*self.row_col, ORIENTATION_STR[self.orientation], self.score)


@dataclass
class LabyrinthMap:
    data: np.ndarray
    end_cell: tuple

    def is_wall(self, row: int, col: int):
        return self.data[row][col] == WALL_CELL

    def count_cells_from(self, start_pose: MapPose):
        DEFAULT_SCORE = np.inf

        sorted_poses = [start_pose]
        best_score_map = np.full((*self.data.shape, 4), DEFAULT_SCORE)

        ancestor_tree = {}

        def expand(node: MapPose):

            for neighbor in node.get_neighbors():
                if self.is_wall(*neighbor.row_col):
                    continue

                row, col = neighbor.row_col
                ori = neighbor.orientation
                current_score = best_score_map[row][col][ori]

                nb_as_tuple = neighbor.as_tuple()
                node_as_tuple = node.as_tuple()
                current_ancestors = ancestor_tree.get(nb_as_tuple, [])

                if neighbor.score < current_score:
                    best_score_map[row][col][ori] = neighbor.score
                    ancestor_tree[nb_as_tuple] = [node_as_tuple]

                elif (
                    neighbor.score == current_score
                    and node_as_tuple not in current_ancestors
                ):
                    ancestor_tree[nb_as_tuple].append(node_as_tuple)
                else:
                    continue

                insort_left(sorted_poses, neighbor, key=lambda x: x.score)

        end_nodes = []
        while sorted_poses:
            current = sorted_poses.pop(0)
            expand(current)

            if tuple(current.row_col) == self.end_cell:
                end_nodes.append(current.as_tuple())

        viz_map = np.min(best_score_map, axis=2)
        viz_map[viz_map == DEFAULT_SCORE] = -10
        plt.matshow(viz_map, cmap="seismic")
        plt.title("Best cell scores")
        plt.colorbar()
        plt.show()

        min_end_score = min([x[3] for x in end_nodes])
        ancestors = [x for x in end_nodes if x[3] == min_end_score]
        trail_cells_map = np.zeros_like(self.data)
        while ancestors:
            current = ancestors.pop(0)
            trail_cells_map[*current[:2]] = 1  # mark cell as visited

            if current != start_pose.as_tuple():  # start pose has no ancestor
                ancestors.extend(ancestor_tree[current])

        return np.sum(trail_cells_map)

    def traverse_from(self, pose: MapPose):
        sorted_poses = [pose]
        visited_map = np.zeros_like(self.data)

        while sorted_poses:
            current = sorted_poses.pop(0)
            visited_map[*current.row_col] = 1

            if tuple(current.row_col) == self.end_cell:
                return current.score

            for neighbor in current.get_neighbors():
                if self.is_wall(*neighbor.row_col):
                    continue

                if visited_map[*neighbor.row_col] != 0:
                    # best score is already set
                    continue

                insort_left(sorted_poses, neighbor, key=lambda x: x.score)


def part_1(path: str):
    map_data, start_pos, end_pos = read_input(path)

    labyrinth = LabyrinthMap(map_data, end_pos)
    score = labyrinth.traverse_from(
        MapPose(np.array(start_pos), orientation=0, score=0)
    )
    print(score)

    return


def part_2(path: str):
    map_data, start_pos, end_pos = read_input(path)

    labyrinth = LabyrinthMap(map_data, end_pos)
    cell_count = labyrinth.count_cells_from(
        MapPose(np.array(start_pos), orientation=0, score=0)
    )
    print(cell_count)


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
