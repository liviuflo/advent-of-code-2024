from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# INPUT_DATA_PATH = "input_data/22_test.txt"
INPUT_DATA_PATH = "input_data/22.txt"

MOD_VAL = 2**24


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
        sn = self.start_value
        yield sn

        for _ in range(steps):
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
    # seqs = []
    # difs_seqs = []

    seq_dicts: List[Dict] = []
    for v in tqdm(start_values, desc="Computing sequences..."):
        seq = np.array(list(SecretNumber(v).get_sequence(STEPS)))
        prices = seq % 10
        difs = prices[1:] - prices[:-1]

        # print(prices)
        # print(difs)

        # seqs.append(prices)
        # difs_seqs.append(difs)

        seq_to_price = dict()
        for price_idx in range(len(prices) - 1, 3, -1):
            seq_ = difs[price_idx - 4 : price_idx]
            seq_to_price[tuple(seq_)] = prices[price_idx]
            # print(prices[price_idx], seq_)

        seq_dicts.append(seq_to_price)

    all_seqs = set()
    for d in seq_dicts:
        all_seqs.update(d.keys())

    best_result = 0
    for seq in tqdm(all_seqs, desc="Finding best sequence"):
        result = sum([d.get(seq, 0) for d in seq_dicts])
        if result > best_result:
            best_result = result

    print(best_result)


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
