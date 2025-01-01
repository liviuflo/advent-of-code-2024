from dataclasses import dataclass
from typing import Callable, Dict, List, Set

import numpy as np
from matplotlib import pyplot as plt

INPUT_DATA_PATH = "input_data/24_test.txt"
INPUT_DATA_PATH = "input_data/24.txt"

GATE_TO_FUNC = {
    "OR": lambda x, y: x | y,
    "XOR": lambda x, y: x ^ y,
    "AND": lambda x, y: x & y,
}


@dataclass
class Wire:
    code: str
    gate_type: str = None
    operands: List["Wire"] = None
    value: int = None

    def call(self):
        if self.gate_type == "input":
            return self.value

        if self.value is not None:
            return self.value

        self.value = GATE_TO_FUNC[self.gate_type](
            self.operands[0].call(), self.operands[1].call()
        )
        return self.value

    def reset(self):
        self.value = None


def read_input(path):
    wire_storage: Dict[str, Callable] = {}

    def get_wire(code):
        wire = wire_storage.get(code, None)
        if not wire:
            # Create new wire
            wire = Wire(code=code)
            wire_storage[code] = wire

        return wire

    with open(path, "r") as file:
        for line in file.readlines():
            line = line.strip()

            if ":" in line:
                # create wire with given value
                code, value = line.split(":")
                assert code not in wire_storage

                wire_storage[code] = Wire(
                    code=code, gate_type="input", value=int(value)
                )

            elif "->" in line:
                # create connections
                input1, gate_type, input2, _, output = line.split(" ")

                input1_wire = get_wire(input1)
                input2_wire = get_wire(input2)

                output_wire = get_wire(output)

                output_wire.gate_type = gate_type
                output_wire.operands = [input1_wire, input2_wire]

    return wire_storage


def part_1(path):
    wire_storage = read_input(path)

    z_wires = [wire for code, wire in wire_storage.items() if code[0] == "z"]
    z_wires_sorted = sorted(z_wires, key=lambda wire: wire.code, reverse=True)

    z_values = map(lambda wire: str(wire.call()), z_wires_sorted)
    result = int("".join(z_values), base=2)
    print(result)


if __name__ == "__main__":
    part_1(INPUT_DATA_PATH)
