import re

INPUT_DATA_PATH = "input_data/03.txt"


def read_raw_txt(path):
    with open(path, "r") as file:
        return file.read()


def compute_mul(text: str):
    first_number = text.split(",")[0].split("(")[-1]
    second_number = text.split(",")[1].split(")")[0]

    return int(first_number) * int(second_number)


def part_1(path):
    search_string = "mul\(\d+,\d+\)"

    raw_text = read_raw_txt(path)
    matches = re.findall(search_string, raw_text)

    values = map(compute_mul, matches)
    print(sum(values))


if __name__ == "__main__":
    part_1(INPUT_DATA_PATH)
