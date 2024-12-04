import numpy as np
from scipy.signal import correlate2d

INPUT_DATA_PATH = "input_data/04.txt"
# INPUT_DATA_PATH = "input_data/04_test.txt"
# INPUT_DATA_PATH = "input_data/04_testme.txt"

char_to_int = {letter: 2**i for i, letter in enumerate("XMAS")}
char_to_int["."] = 0

XMAS = [1, 2, 4, 8]
XMAS_REV = list(reversed(XMAS))


def line_to_array(line):
    return list(map(lambda c: char_to_int[c], line.strip()))


def read_input(path):
    with open(path, "r") as file:
        lines = file.readlines()

    return np.array([line_to_array(l) for l in lines])


def apply_filter(matrix: np.ndarray, filter: np.ndarray):
    result = correlate2d(matrix, filter, mode="valid")
    # print((result == 85).astype(np.int32))
    return np.sum(result == 85)


def pad_matrix(matrix: np.ndarray, pad_size: int):
    top_bottom = np.zeros((pad_size, matrix.shape[1]))
    m = np.vstack((top_bottom, matrix, top_bottom))

    sides = np.zeros((m.shape[0], pad_size))
    m = np.hstack((sides, m, sides))

    return m


def part_1(path):
    matrix = np.array(read_input(path))
    # matrix = pad_matrix(matrix, 4)

    filters = [
        np.array(XMAS).reshape((1, 4)),
        np.array(XMAS_REV).reshape((1, 4)),
        np.array(XMAS).reshape((4, 1)),
        np.array(XMAS_REV).reshape((4, 1)),
        np.diag(XMAS),
        np.diag(XMAS_REV),
        np.copy(np.diag(XMAS)[:, ::-1]),
        np.copy(np.diag(XMAS_REV)[:, ::-1]),
    ]

    result = sum([apply_filter(matrix, f) for f in filters])
    print(result)


if __name__ == "__main__":
    part_1(INPUT_DATA_PATH)
