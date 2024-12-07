import numpy as np
from tqdm import tqdm

INPUT_DATA_PATH = "input_data/07.txt"

OPERATORS = {2: {}, 3: {}}
BASE_REP = {2: {}, 3: {}}


def get_base_rep(x, base):
    val = BASE_REP[base].get(x, None)
    if val is None:
        val = np.base_repr(x, base)
        BASE_REP[base][x] = val

    return val


def prepend_zero(l: list[str]):
    return list(map(lambda x: "0" + x, l))


def precompute_operators(base: int, max_length: int):
    OPERATORS[base][0] = []

    def compute_option(x: int):
        val = get_base_rep(x, base)
        return "0" * (length - len(val)) + val

    def compute_operators(base: int, length: int):
        prev_operators = prepend_zero(OPERATORS[base][length - 1])
        new_operators = [
            compute_option(x) for x in range(base ** (length - 1) - 1, base**length)
        ]
        return prev_operators + new_operators

    for length in range(1, max_length + 1):
        OPERATORS[base][length] = compute_operators(base, length)


def find_solution(result, operands, options):
    for operator_list in options:
        value = operands[0]
        assert len(operator_list) == len(operands) - 1
        for operator, operand in zip(operator_list, operands[1:]):
            if operator == "0":  # add
                value += operand
            elif operator == "1":  # multiply
                value *= operand
            elif operator == "2":  # concatenate
                value = int(str(value) + str(operand))

            if value > result:
                break

        if value == result:
            return True

    return False


def extract_data(line: str):
    result = int(line.split(":")[0])
    operands = list(map(int, line.strip().split(": ")[1].split(" ")))
    return result, operands


def read_input(path: str):
    with open(path, "r") as file:
        return [extract_data(line) for line in file.readlines()]


def part_1(path):
    data = read_input(path)
    precompute_operators(2, max(len(op) for r, op in data))

    s = 0
    for result, operands in data:
        options_2 = OPERATORS[2][len(operands) - 1]
        if find_solution(result, operands, options_2):
            s += result

    print(s)


def part_2(path):
    data = read_input(path)
    max_len = max(len(op) for r, op in data)
    precompute_operators(2, max_len)
    precompute_operators(3, max_len)

    s = 0
    for result, operands in tqdm(data):
        op_len = len(operands) - 1

        if find_solution(result, operands, OPERATORS[2][op_len]):
            s += result
        else:
            if find_solution(result, operands, OPERATORS[3][op_len]):
                s += result

    print(s)


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
