import numpy as np

INPUT_DATA_PATH = "input_data/25.txt"

CHAR_TO_VAL = {"#": 1, ".": 0}


def create_entity(lines):
    int_lines = [list(map(lambda c: CHAR_TO_VAL[c], line)) for line in lines]
    return np.sum(int_lines, axis=0) - 1


def lines_are_lock(lines):
    return lines[0] == "#####"


def read_input(path):
    current_lines = []
    locks = []
    keys = []

    def create_lock_or_key(lines):
        entity = create_entity(lines)
        is_lock = lines_are_lock(lines)

        target_list = locks if is_lock else keys
        target_list.append(entity)

    with open(path, "r") as file:
        for line in file.readlines():
            line = line.strip()
            if not line.strip():
                create_lock_or_key(current_lines)
                current_lines = []
            else:
                current_lines.append(line)

    if current_lines:
        create_lock_or_key(current_lines)

    return locks, keys


def lock_fits_key(lock, key):
    return not np.any(lock + key > 5)


def part_1(path):
    locks, keys = read_input(path)

    result = 0
    for lock in locks:
        for key in keys:
            if lock_fits_key(lock, key):
                result += 1

    print(result)


if __name__ == "__main__":
    part_1(INPUT_DATA_PATH)
