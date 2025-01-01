from dataclasses import dataclass
from typing import Dict, List, Set

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
    value: int = None
    gate_type: str = None
    is_input_for: Set["Wire"] = None
    inputs: List["Wire"] = None

    def add_output(self, output: "Wire"):
        self.is_input_for.add(output)

    def add_input(self, input: "Wire", input_id: int):
        if self.inputs is None:
            self.inputs = [None, None]

        self.inputs[input_id] = input

    def ready_to_fire(self):
        if self.inputs is None:
            return False

        for input_wire in self.inputs:
            if input_wire is None:
                return False

            if input_wire.value == None:
                return False

        return True

    def try_to_compute_value(self):
        if not self.ready_to_fire():
            return False

        if self.gate_type is None:
            return False

        self.value = GATE_TO_FUNC[self.gate_type](
            *map(lambda wire: wire.value, self.inputs)
        )
        return True

    def __hash__(self):
        return hash(self.code)

    def __repr__(self):
        output_str = (
            ",".join([x.code for x in self.is_input_for]) if self.is_input_for else "{}"
        )
        input_str = ",".join([x.code for x in self.inputs]) if self.inputs else "{}"
        type_str = f"type={self.gate_type}, " if self.gate_type else ""
        return f"Wire({self.code}, v={self.value}, {type_str}is_input_for={output_str}, inputs={input_str})"


def read_input(path):
    wire_storage: Dict[str, Wire] = {}
    live_wires: Set[Wire] = set()

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

                new_wire = Wire(code=code, value=int(value), is_input_for=set())
                wire_storage[code] = new_wire

                live_wires.add(new_wire)

            elif "->" in line:
                # create connections
                input1, gate_type, input2, _, output = line.split(" ")

                # print(input1, gate_type, input2, output)
                input1_wire = get_wire(input1)
                input2_wire = get_wire(input2)

                output_wire = get_wire(output)

                output_wire.gate_type = gate_type

                for input_id, input_wire in enumerate([input1_wire, input2_wire]):
                    input_wire.add_output(output_wire)
                    output_wire.add_input(input_wire, input_id)

    return wire_storage, live_wires


def part_1(path):
    wire_storage, live_wires = read_input(path)

    z_outputs = set()

    while live_wires:

        new_live_wires = set()
        for wire in live_wires:
            for output_wire in wire.is_input_for:

                if output_wire in new_live_wires:
                    continue

                if output_wire.try_to_compute_value():
                    new_live_wires.add(output_wire)

                    if output_wire.code[0] == "z":
                        z_outputs.add(output_wire)

        live_wires = new_live_wires

    z_values = map(
        lambda wire: str(wire.value),
        sorted(z_outputs, key=lambda wire: wire.code, reverse=True),
    )
    result = int("".join(z_values), base=2)
    print(result)


if __name__ == "__main__":
    part_1(INPUT_DATA_PATH)
