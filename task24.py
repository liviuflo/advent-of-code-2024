from dataclasses import dataclass
from typing import Callable, Dict, List, Set

# INPUT_DATA_PATH = "input_data/24_test.txt"
INPUT_DATA_PATH = "input_data/24.txt"
INPUT_DATA_PATH = "input_data/24-swap1.txt"
INPUT_DATA_PATH = "input_data/24-swap2.txt"
INPUT_DATA_PATH = "input_data/24-swap3.txt"

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
    feeds_into: Set["Wire"] = None

    def __repr__(self):
        return f"Wire({self.code})"

    def __hash__(self):
        return hash(self.code)

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

    def add_output(self, output: "Wire"):
        if self.feeds_into is None:
            self.feeds_into = set()

        self.feeds_into.add(output)


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

                input1, input2 = sorted([input1, input2])

                input1_wire = get_wire(input1)
                input2_wire = get_wire(input2)

                output_wire = get_wire(output)

                output_wire.gate_type = gate_type
                output_wire.operands = [input1_wire, input2_wire]

                input1_wire.add_output(output_wire)
                input2_wire.add_output(output_wire)

    return wire_storage


def compute(wire_storage):

    z_wires = [wire for code, wire in wire_storage.items() if code[0] == "z"]
    z_wires_sorted = sorted(z_wires, key=lambda wire: wire.code, reverse=True)

    z_values = map(lambda wire: str(wire.call()), z_wires_sorted)
    result = int("".join(z_values), base=2)
    return result


def part_1(path):
    wire_storage = read_input(path)
    print(compute(wire_storage))


def get_nth_bit(x, n):
    return (x >> n) % 2


def get_output(wire_storage: Dict[str, Wire], x, y):

    for code, wire in wire_storage.items():
        wire.reset()

        if code[0] == "x":
            wire.value = get_nth_bit(x, int(code[1:]))
        elif code[0] == "y":
            wire.value = get_nth_bit(y, int(code[1:]))

    return compute(wire_storage)


def get_xyz(i):
    id_val = f"{i:02}"
    return [s + id_val for s in "xyz"]


def check_structure(wire_storage: Dict[str, Wire]):

    def get_common(w1: Wire, w2: Wire, type: str):

        intersection = w1.feeds_into.intersection(w2.feeds_into)
        return [w for w in intersection if w.gate_type == type]

    carry = None
    i = 0

    while i <= 44:
        print(i)
        x, y, z = get_xyz(i)
        x_wire = wire_storage[x]
        y_wire = wire_storage[y]

        common_outputs = list(x_wire.feeds_into.intersection(y_wire.feeds_into))

        common_outputs = sorted(common_outputs, key=lambda x: x.gate_type)

        assert tuple(map(lambda x: x.gate_type, common_outputs)) == (
            "AND",
            "XOR",
        )

        and_gate = common_outputs[0]
        xor_gate = common_outputs[1]

        print("CARRY:", carry)

        print("AND:", and_gate, "XOR:", xor_gate)

        # compute output
        if i == 0:
            output = xor_gate
            carry = and_gate
        elif i >= 1:
            output = get_common(carry, xor_gate, "XOR")[0]
            if output.code != z:
                print("SWAP", output.code, z)
            temp_carry = get_common(carry, xor_gate, "AND")[0]
            print("Temp carry", temp_carry)
            carry = get_common(temp_carry, and_gate, "OR")[0]

        i += 1


def part_2(path):
    wire_storage: Dict[str, Wire] = read_input(path)

    check_structure(wire_storage)


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
