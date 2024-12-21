INPUT_DATA_PATH = "input_data/19.txt"


def read_patterns(line: str):
    return line.split(", ")


def read_input(path):
    patterns = []
    targets = []
    with open(path, "r") as file:
        for line in file.readlines():
            l = line.strip()
            if "," in l:
                patterns = read_patterns(l)
            elif len(l):
                targets.append(l)

    return patterns, targets


def can_build_target(target, patterns):
    working_solutions = set([""])
    while True:
        new_solutions = set()
        for sol in working_solutions:
            for pattern in patterns + [""]:
                # Try adding this pattern
                potential_sol = sol + pattern

                # Check if it matches the target so far
                if target[: len(potential_sol)] != potential_sol:
                    # Discard pattern
                    continue

                # Check if it's an actual solution
                if len(potential_sol) == len(target) and potential_sol == target:
                    return True

                # Good pattern
                new_solutions.add(potential_sol)

        working_solutions = new_solutions.difference(working_solutions)
        if not working_solutions:
            # Nothing left to try
            return False


def part_1(path):
    patterns, targets = read_input(path)

    valid_targets = sum(map(lambda t: can_build_target(t, patterns), targets))
    print(valid_targets)


if __name__ == "__main__":
    part_1(INPUT_DATA_PATH)
