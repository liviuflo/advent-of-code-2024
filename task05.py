from functools import cmp_to_key

INPUT_DATA_PATH = "input_data/05.txt"


def extract_dep(line):
    return list(map(int, line.strip().split("|")))


def extract_seq(line):
    return list(map(int, line.strip().split(",")))


def get_middle_value(seq):
    return seq[len(seq) // 2]


def fix_sequence(seq, deps):
    def dep_compare(item1, item2):
        if item1 in deps and item2 in deps[item1]:
            # item2 should appear before item1
            return 1

        if item2 in deps and item1 in deps[item2]:
            # item 1 should appear before item2
            return -1

        return 0

    return sorted(seq, key=cmp_to_key(dep_compare))


def is_valid_seq(seq, deps):
    seq_len = len(seq)
    for i in range(seq_len):
        x = seq[i]
        expected_deps = deps[x]
        for j in range(i + 1, seq_len):
            y = seq[j]
            if y in expected_deps:
                # print(f"{seq} is not valid because: {y}|{x}")
                return False

    return True


def read_inputs(path):
    deps = {}
    seqs = []
    with open(path, "r") as file:
        for line in file.readlines():
            if "|" in line:
                x, y = extract_dep(line)

                # x should appear before y
                # => if y appears before x, the sequence is wrong
                # for each y, we store the x's it shouldn't appear before

                deps.setdefault(y, set())
                deps[y].add(x)

            elif "," in line:
                seqs.append(extract_seq(line))

    return deps, seqs


def part_1(path):
    deps, seqs = read_inputs(path)

    valid_seq_middle_element_sum = 0
    for seq in seqs:
        if is_valid_seq(seq, deps):
            valid_seq_middle_element_sum += get_middle_value(seq)

    print(valid_seq_middle_element_sum)


def part_2(path):
    deps, seqs = read_inputs(path)

    fixed_seq_middle_element_sum = 0
    for seq in seqs:
        if is_valid_seq(seq, deps):
            continue

        fixed_seq = fix_sequence(seq, deps)
        assert is_valid_seq(fixed_seq, deps)

        fixed_seq_middle_element_sum += get_middle_value(fixed_seq)

    print(fixed_seq_middle_element_sum)


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
