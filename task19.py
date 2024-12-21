from tqdm import tqdm

INPUT_DATA_PATH = "input_data/19.txt"
# INPUT_DATA_PATH = "input_data/19_test.txt"


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


def count_combinations(target, patterns):
    """Count the number of ways a given target can be obtained using the set of patterns."""

    # Organise patterns by first letter
    patterns_by_first_letter = {}
    for p in patterns:
        first_letter = p[0]
        patterns_by_first_letter.setdefault(first_letter, list())
        patterns_by_first_letter[first_letter].append(p)

    temp_solutions = {"": 1}
    solution_count = 0
    while True:
        new_solutions = dict()
        for sol, sol_variants in temp_solutions.items():
            next_letter = target[len(sol)]
            for pattern in patterns_by_first_letter.get(next_letter, []):
                # Try adding this pattern
                potential_sol = sol + pattern

                # Check if it matches the target so far
                if target[: len(potential_sol)] != potential_sol:
                    # Discard pattern
                    continue

                new_solutions.setdefault(potential_sol, 0)
                new_solutions[potential_sol] += sol_variants

        if target in new_solutions:
            solution_count += new_solutions.pop(target)

        temp_solutions = new_solutions
        if not temp_solutions:
            break

    return solution_count


def part_2(path):
    patterns, targets = read_input(path)

    combinations = sum(map(lambda t: count_combinations(t, patterns), tqdm(targets)))
    print(combinations)


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
