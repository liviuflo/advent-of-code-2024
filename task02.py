INPUT_DATA_PATH = "input_data/02.txt"


def read_lines(path):
    with open(path, "r") as file:
        return [list(map(int, l.strip().split(" "))) for l in file.readlines()]


def is_safe_line(line):
    if len(line) < 2:
        return False

    prev_dif = None
    for i in range(1, len(line)):
        dif = line[i] - line[i - 1]

        # Check dif absolute value
        if abs(dif) < 1 or abs(dif) > 3:
            return False

        if prev_dif is None:
            prev_dif = dif
            continue

        # Check difs have same sign
        if dif * prev_dif < 0:
            return False

    return True


def is_safe_line_2(line):
    if len(line) < 2:
        return False

    if is_safe_line(line):
        return True

    for i in range(len(line)):
        line_without_i = line[:i] + line[i + 1 :]
        if is_safe_line(line_without_i):
            return True

    return False


def part_1(path):
    lines = read_lines(path)

    safe_lines = list(filter(is_safe_line, lines))
    print("safe lines part 1:", len(safe_lines))


def part_2(path):
    lines = read_lines(path)

    safe_lines = list(filter(is_safe_line_2, lines))
    print("safe lines part 2:", len(safe_lines))


if __name__ == "__main__":
    part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
