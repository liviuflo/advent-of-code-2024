from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

INPUT_DATA_PATH = "input_data/12.txt"

VISITED = 2
TO_VISIT = 1
NOT_VISITED = 0
BOUNDARY = 3


@dataclass
class Garden:
    data: np.ndarray

    def is_in_garden(self, xy: np.ndarray):
        return 0 <= xy[0] < self.data.shape[0] and 0 <= xy[1] < self.data.shape[1]

    def get_neighbors(self, xy: np.ndarray):
        deltas = [[0, 1], [-1, 0], [0, -1], [1, 0]]

        for d in deltas:
            yield xy + d

    def get_value(self, xy: np.ndarray):
        return self.data[xy[0]][xy[1]]

    def compute_from(self, xy: np.ndarray):
        visit_map = np.zeros_like(self.data, dtype=np.uint16)

        patch_type = self.get_value(xy)

        area = 0
        perimeter = 0

        outside_points = set()

        points_to_visit = [xy]
        while points_to_visit:

            current = points_to_visit.pop(0)
            cx, cy = current

            visit_map[cx][cy] = VISITED
            area += 1

            for n in self.get_neighbors(current):
                nx, ny = n
                if not self.is_in_garden(n):
                    # Boundary cell
                    if tuple(n) in outside_points:
                        # Already marked
                        continue

                    # New boundary outside map
                    outside_points.add(tuple(n))
                    perimeter += 1
                    continue

                if visit_map[nx][ny] == VISITED:
                    continue

                if visit_map[nx][ny] == BOUNDARY:
                    perimeter += 1
                    continue

                if self.get_value(n) == patch_type:
                    if visit_map[nx][ny] == TO_VISIT:
                        # already known
                        continue

                    # new point
                    points_to_visit.append(n)
                    visit_map[nx][ny] = TO_VISIT
                else:
                    # new boundary inside map
                    visit_map[nx][ny] = BOUNDARY
                    perimeter += 1

        # print(area, perimeter)
        # plt.imshow(visit_map)
        # plt.show()
        return area, perimeter, visit_map == VISITED


def read_data(path):
    with open(path, "r") as file:
        return np.array([list(line.strip()) for line in file.readlines()])


def part_1(path):
    data = read_data(path)

    garden = Garden(data)

    full_visit_map = np.zeros_like(data, dtype=bool)

    cost = 0
    patches = 0
    for x, y in np.argwhere(data):
        if full_visit_map[x][y]:
            # this patch has been visited
            continue

        patches += 1

        area, perimeter, visited = garden.compute_from(np.array([x, y]))
        cost += area * perimeter

        full_visit_map = np.logical_or(full_visit_map, visited)

    print(cost)
    print(f"Visited {patches} patches")


if __name__ == "__main__":
    part_1(INPUT_DATA_PATH)
