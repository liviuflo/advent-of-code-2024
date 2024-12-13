from dataclasses import dataclass
from typing import List

import numpy as np

INPUT_DATA_PATH = "input_data/13.txt"

# Change to 0 for part 1
EXTRA_COST = 10000000000000


@dataclass
class Machine:
    a_xy: np.ndarray
    b_xy: np.ndarray
    prize_xy: np.ndarray

    def solve(self):
        mat = np.hstack((self.a_xy.reshape((2, 1)), self.b_xy.reshape((2, 1))))
        t = self.prize_xy.reshape((2, 1))

        sol = np.linalg.inv(mat.T @ mat) @ mat.T @ t

        if np.linalg.norm(sol - np.round(sol)) < 0.001:
            return sol

        return None


def extract_button(line: str):
    x = line.split("X+")[-1].split(",")[0]
    y = line.split("Y+")[-1]
    return np.array([int(x), int(y)])


def extract_prize(line: str):
    x = line.split("X=")[-1].split(",")[0]
    y = line.split("Y=")[-1]
    return np.array([int(x), int(y)]) + EXTRA_COST


def read_input(path):
    machines: List[Machine] = []
    a, b, prize = None, None, None
    with open(path, "r") as file:
        for line in file.readlines():
            if "Button A" in line:
                a = extract_button(line.strip())
            elif "Button B" in line:
                b = extract_button(line.strip())
            elif "Prize" in line:
                prize = extract_prize(line.strip())
                machines.append(Machine(a, b, prize))

    return machines


def solve(path):
    machines = read_input(path)

    cost = 0
    for machine in machines:
        solution = machine.solve()
        if solution is None:
            continue

        buttons = solution.squeeze()
        cost += 3 * buttons[0] + buttons[1]

    print(int(cost))


if __name__ == "__main__":
    solve(INPUT_DATA_PATH)
