from collections import Counter

INPUT_DATA_PATH = "input_data/01.txt"


def read_lists(path):
    l1, l2 = [], []
    with open(path, "r") as file:
        for l in file.readlines():
            v1, v2 = map(int, l.strip().split("   "))
            l1.append(v1)
            l2.append(v2)

    return l1, l2


def part_1(path):
    l1, l2 = read_lists(path)
    l1 = sorted(l1)
    l2 = sorted(l2)

    dif = [abs(v1 - v2) for v1, v2 in zip(l1, l2)]
    print(sum(dif))


def part_2(path):
    l1, l2 = read_lists(path)

    counter = Counter(l2)

    score = 0
    for x in l1:
        score += x * counter[x]

    print(score)


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
