import re

INPUT_DATA_PATH = "input_data/03.txt"
MUL_REGEX = "mul\(\d+,\d+\)"
DONT_STR = "don't()"
DO_STR = "do()"


def read_raw_txt(path):
    with open(path, "r") as file:
        return file.read()


def compute_mul_single(text: str):
    first_number = text.split(",")[0].split("(")[-1]
    second_number = text.split(",")[1].split(")")[0]

    return int(first_number) * int(second_number)


def compute_all_muls(text: str):
    matches = re.findall(MUL_REGEX, text)
    return sum(map(compute_mul_single, matches))


def part_1(path):
    raw_text = read_raw_txt(path)
    return compute_all_muls(raw_text)


def split_do_dont(text):
    parts = []

    dont_index = text.find(DONT_STR)
    if dont_index == -1:
        return [text]

    while True:
        # Select text between do and dont
        parts.append(text[:dont_index])

        # Find the next do after dont
        do_index = text.find(DO_STR, dont_index)
        text = text[do_index:]

        # Find the next dont after do
        dont_index = text.find(DONT_STR)
        if dont_index == -1:
            return parts + [text]


def part_2(path):
    text = read_raw_txt(path)
    parts = split_do_dont(text)
    return sum(map(compute_all_muls, parts))


if __name__ == "__main__":
    # print(part_1(INPUT_DATA_PATH))
    print(part_2(INPUT_DATA_PATH))
