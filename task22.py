from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

INPUT_DATA_PATH = "input_data/22_test.txt"
INPUT_DATA_PATH = "input_data/22.txt"

MOD_VAL = 2**24

SECRET_NUMBERS_STORAGE = dict()


@dataclass
class SecretNumber:
    start_value: int

    def __op1(self, secret_num: int):
        v = secret_num << 6  # multiply by 64
        v = v ^ secret_num
        v = v % MOD_VAL
        return v

    def __op2(self, secret_num: int):
        v = secret_num >> 5  # divide by 32
        v = v ^ secret_num
        v = v % MOD_VAL
        return v

    def __op3(self, secret_num: int):
        v = secret_num << 11
        v = v ^ secret_num
        v = v % MOD_VAL
        return v

    def __op123(self, secret_num: int):

        sn = secret_num

        v = sn << 6  # multiply by 64
        v = (v ^ sn) % MOD_VAL

        sn = v
        v = sn >> 5  # divide by 32
        v = (v ^ sn) % MOD_VAL

        sn = v
        v = sn << 11  # multiply by 2048
        v = (v ^ sn) % MOD_VAL

        return v

    def get_value(self, step: int):
        sn = self.start_value
        for _ in range(step):
            for op in [self.__op1, self.__op2, self.__op3]:
                sn = op(sn)

        return sn

    def get_sequence(self, steps: int):
        global SECRET_NUMBERS_STORAGE

        sn = self.start_value
        yield sn

        for _ in range(steps):
            # if sn in SECRET_NUMBERS_STORAGE:
            #     sn = SECRET_NUMBERS_STORAGE[sn]
            # else:
            #     res = self.__op123(sn)
            #     SECRET_NUMBERS_STORAGE[sn] = res
            #     sn = res
            sn = self.__op123(sn)
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
