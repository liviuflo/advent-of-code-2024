import numpy as np
from scipy.signal import correlate2d

INPUT_DATA_PATH = "input_data/04.txt"

CHAR_TO_INT = {letter: 2**i for i, letter in enumerate("XMAS")}
CHAR_TO_INT["."] = 0


def read_input(path: str):
    with open(path, "r") as file:
        return np.array(
            [list(map(lambda c: CHAR_TO_INT[c], l.strip())) for l in file.readlines()]
        )


def apply_filter(matrix: np.ndarray, filter: np.ndarray, target_value: int):
    result = correlate2d(matrix, filter, mode="valid")
    return np.sum(result == target_value)


def part_1(path):
    matrix = read_input(path)

    XMAS = [1, 4, 16, 64]
    XMAS_REV = list(reversed(XMAS))

    filters = [
        np.array(XMAS).reshape((1, 4)),
        np.array(XMAS_REV).reshape((1, 4)),
        np.array(XMAS).reshape((4, 1)),
        np.array(XMAS_REV).reshape((4, 1)),
        np.diag(XMAS),
        np.diag(XMAS_REV),
        np.diag(XMAS)[:, ::-1],
        np.diag(XMAS_REV)[:, ::-1],
    ]

    result = sum([apply_filter(matrix, f, target_value=585) for f in filters])
    print(result)


def part_2(path):
    matrix = read_input(path)

    FILTER = np.array(
        [
            [1, 0, 4],
            [0, 16, 0],
            [64, 0, 256],
        ]
    )

    filters = [
        FILTER,  # M top
        FILTER.T,  # M left
        FILTER[::-1, :],  # M bottom
        FILTER.T[:, ::-1],  # M right
    ]

    result = sum([apply_filter(matrix, f, target_value=2634) for f in filters])
    print(result)


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
