from dataclasses import dataclass
from typing import Dict, Set

INPUT_DATA_PATH = "input_data/24_test.txt"
# INPUT_DATA_PATH = "input_data/24.txt"


@dataclass
class Wire:
    code: str
    value: int = None
    gate_type: str = None
    is_input_for: Set["Wire"] = None

    def add_output(self, output: "Wire"):
        self.is_input_for.add(output)

    def __hash__(self):
        return hash(self.code)

    def __repr__(self):
        output_str = (
            ",".join([x.code for x in self.is_input_for]) if self.is_input_for else "{}"
        )
        type_str = f"type={self.gate_type}, " if self.gate_type else ""
        return f"Wire({self.code}, v={self.value}, {type_str}is_input_for={output_str}"


def read_input(path):
    wire_storage: Dict[str, Wire] = {}

    def get_wire(code: str):
        if code in wire_storage:
            return wire_storage[code]

        new_wire = Wire(code=code, is_input_for=set())
        wire_storage[code] = new_wire

        return new_wire

    with open(path, "r") as file:
        for line in file.readlines():
            line = line.strip()

            if ":" in line:
                # create wire with given value
                code, value = line.split(":")
                assert code not in wire_storage

                wire_storage[code] = Wire(
                    code=code, value=int(value), is_input_for=set()
                )

            elif "->" in line:
                # create connections
                input1, gate_type, input2, _, output = line.split(" ")

                # print(input1, gate_type, input2, output)
                input1_wire = get_wire(input1)
                input2_wire = get_wire(input2)

                output_wire = get_wire(output)

                output_wire.gate_type = gate_type

                input1_wire.add_output(output_wire)
                input2_wire.add_output(output_wire)

    return wire_storage


def part_1(path):
    wire_storage = read_input(path)

    print(wire_storage)


if __name__ == "__main__":
    part_1(INPUT_DATA_PATH)
