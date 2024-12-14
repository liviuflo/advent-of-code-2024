from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

INPUT_DATA_PATH = "input_data/14.txt"


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

    seconds = 100
    grid_size = (101, 103)
    # grid_size = (11, 7)

    robot_locations = np.zeros(grid_size)

    for robot in robots:
        new_loc = robot.move_seconds(seconds, grid_size)
        robot_locations[*new_loc] += 1

    plt.imshow(robot_locations.T)
    plt.show()

    score = compute_score(robot_locations)
    print(score)


if __name__ == "__main__":
    part_1(INPUT_DATA_PATH)
