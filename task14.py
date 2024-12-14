from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

INPUT_DATA_PATH = "input_data/14.txt"
# INPUT_DATA_PATH = "input_data/14_test.txt" # grid_size = (11, 7)


@dataclass
class Robot:
    start_xy: np.ndarray
    velocity: np.ndarray

    @staticmethod
    def create_from_line(line: str):
        x = int(line.split("p=")[-1].split(",")[0])
        y = int(line.split(",")[1].split(" ")[0])

        vel_x = int(line.split("v=")[-1].split(",")[0])
        vel_y = int(line.split(",")[-1])

        return Robot(np.array([x, y]), np.array([vel_x, vel_y]))

    def move_seconds(self, seconds: int, grid_size: tuple = None):
        new_loc = self.start_xy + seconds * self.velocity
        return np.mod(new_loc, np.array(grid_size))


def generate_ranges(val: int):
    assert val % 2 == 1
    half = val // 2

    return [[0, half], [half + 1, val]]


def compute_score(robot_locations: np.ndarray):
    x, y = robot_locations.shape

    score = 1
    for x_start, x_end in generate_ranges(x):
        for y_start, y_end in generate_ranges(y):
            quadrant = robot_locations[x_start:x_end, y_start:y_end]
            score *= int(np.sum(quadrant))

    return score


def read_robots(path):
    with open(path, "r") as file:
        return [Robot.create_from_line(line.strip()) for line in file.readlines()]


def part_1(path):
    robots = read_robots(path)
    grid_size = (101, 103)

    robot_locations = get_map_at_sec(robots, 100, grid_size)

    plt.imshow(robot_locations.T)
    plt.show()

    score = compute_score(robot_locations)
    print(score)


def get_map_at_sec(robots, sec, grid_size):
    robot_map = np.zeros(grid_size)
    for r in robots:
        new_loc = r.move_seconds(sec, grid_size)
        robot_map[*new_loc] += 1

    return robot_map


def compute_spread(robot_map: np.ndarray):
    locations = np.argwhere(robot_map)
    stdev = np.std(locations, axis=0)

    return np.linalg.norm(stdev)


def part_2(path):
    robots = read_robots(path)

    max_seconds = 10000

    grid_size = (101, 103)

    min_std = 100
    min_std_map = None
    min_std_sec = 0
    for sec in tqdm(range(max_seconds)):
        robot_locations = get_map_at_sec(robots, sec, grid_size)
        std = compute_spread(robot_locations)
        if std < min_std:
            min_std = std
            min_std_sec = sec
            min_std_map = robot_locations.T

    print(min_std_sec)
    plt.matshow(min_std_map)
    plt.show()


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
