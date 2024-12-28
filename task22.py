from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

INPUT_DATA_PATH = "input_data/22_test.txt"
INPUT_DATA_PATH = "input_data/22.txt"

MOD_VAL = 2**24

SECRET_NUMBERS_STORAGE = dict()


def generate_next(sn: int):
    v = ((sn << 6) ^ sn) % MOD_VAL
    v = ((v >> 5) ^ v) % MOD_VAL
    v = ((v << 11) ^ v) % MOD_VAL

    return v


@dataclass
class SecretNumber:
    start_value: int

    def get_value(self, steps: int):
        sn = self.start_value
        for _ in range(steps):
            sn = generate_next(sn)

        return sn

    def get_sequence(self, steps: int):
        sn = self.start_value
        yield sn

        for _ in range(steps):
            sn = generate_next(sn)
            yield sn


def read_input(path):
    with open(path, "r") as file:
        return [int(line.strip()) for line in file.readlines()]


def part_1(path):
    start_values = read_input(path)

    print(sum([SecretNumber(v).get_value(2000) for v in start_values]))


def part_2(path):
    start_values = read_input(path)

    STEPS = 2000

    seq_prices = dict()
    for v in tqdm(start_values, desc="Computing sequences..."):
        seq = np.array(list(SecretNumber(v).get_sequence(STEPS)))
        prices = seq % 10
        difs = prices[1:] - prices[:-1]

        seen_seq = set()
        for idx in range(4, len(prices)):
            seq_ = tuple(difs[idx - 4 : idx])
            if seq_ not in seen_seq:
                seen_seq.add(seq_)
                seq_prices[seq_] = seq_prices.get(seq_, 0) + prices[idx]

    print(max(seq_prices.values()))


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
