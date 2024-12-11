from dataclasses import dataclass
from typing import List

INPUT_DATA_PATH = "input_data/11.txt"


def read_input(path):
    with open(path, "r") as file:
        first_line = file.readline()
        return list(map(int, first_line.strip().split(" ")))


@dataclass
class BlinkElement:
    value: int
    count: int = 1

    def __repr__(self):
        return f"(v={self.value}, c={self.count})"

    def blink(self):
        if self.value == 0:
            return [BlinkElement(1, self.count)]

        string = str(self.value)
        string_len = len(string)
        if string_len % 2 == 0:
            half = string_len // 2
            return [
                BlinkElement(int(string[:half]), self.count),
                BlinkElement(int(string[half:]), self.count),
            ]

        return [BlinkElement(self.value * 2024, self.count)]


@dataclass
class BlinkSequence:
    elements: List[BlinkElement]

    @staticmethod
    def create_from_list(values: List[int]) -> "BlinkSequence":
        seq = BlinkSequence(BlinkElement(v) for v in values)
        seq.__squeeze()
        return seq

    def __squeeze(self):
        count_per_value = {}

        for el in self.elements:
            count_per_value.setdefault(el.value, 0)
            count_per_value[el.value] += el.count

        self.elements = [
            BlinkElement(val, count) for val, count in count_per_value.items()
        ]

    def blink(self):
        new_els = []
        for el in self.elements:
            new_els.extend(el.blink())

        self.elements = new_els
        self.__squeeze()

    @property
    def length(self):
        return sum([el.count for el in self.elements])

    @property
    def count_elements(self):
        return len(self.elements)


def part_12(path: str, wanted_blinks: int):
    data = read_input(path)

    sequence = BlinkSequence.create_from_list(data)

    for i in range(wanted_blinks):
        sequence.blink()

        print("Blink", i + 1)
        print(f"Unique values: {sequence.count_elements}; Length: {sequence.length}")


if __name__ == "__main__":
    part_12(INPUT_DATA_PATH, 75)
