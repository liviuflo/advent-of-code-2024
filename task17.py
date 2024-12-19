import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

INPUT_DATA_PATH = "input_data/17.txt"


@dataclass
class ProgramStep:
    opcode: int
    operand: int


@dataclass
class Register:
    a: int
    b: int
    c: int

    def get_values(self):
        return [self.a, self.b, self.c]


@dataclass
class Program:
    reg: Register
    seq: List[int]

    def run_custom(self, target=None):
        """Run all the operations from the official input in a single step."""
        outputs = []
        while True:
            a = self.reg.a
            d = (a % 8) ^ 2

            new_a = a >> 3
            new_b = (d ^ 3) ^ (a >> d)
            new_c = a >> d

            outputs.append(new_b % 8)
            if target is not None:
                # Compare output so far with target
                if outputs[-1] != target[len(outputs) - 1]:
                    return None

            if new_a == 0:
                return outputs

            self.reg = Register(new_a, new_b, new_c)

    def run(self, target=None):
        """Execute each operation in the program sequence."""
        outputs = []
        pointer = 0
        while pointer < len(self.seq):
            output, pointer = self.run_step(pointer)

            if output is not None:
                outputs.append(output)

                if target is not None:
                    # Compare output so far with target
                    if tuple(outputs) != tuple(target[: len(outputs)]):
                        return None

        return ",".join(map(str, outputs))

    def __combo_operand(self, operand: int):
        """Retrieve the value of a combo operand given its literal value."""
        return [0, 1, 2, 3, *self.reg.get_values()][operand]

    def run_step(self, pointer_val: int) -> Tuple[Optional[int], int]:
        """Run a single step at the given pointer value."""
        next_pointer_val = pointer_val + 2
        step = ProgramStep(*self.seq[pointer_val:next_pointer_val])
        if step.opcode == 0:
            # A = A / 2**combo operand
            self.reg.a = self.reg.a >> self.__combo_operand(step.operand)
            return None, next_pointer_val

        if step.opcode == 1:
            # B = B XOR literal
            self.reg.b = self.reg.b ^ step.operand
            return None, next_pointer_val

        if step.opcode == 2:
            # B = combo operand mod 8
            self.reg.b = self.__combo_operand(step.operand) % 8
            return None, next_pointer_val

        if step.opcode == 3:
            # Jump
            if self.reg.a != 0:
                next_pointer_val = step.operand

            return None, next_pointer_val

        if step.opcode == 4:
            # B = B XOR C
            self.reg.b = self.reg.b ^ self.reg.c
            return None, next_pointer_val

        if step.opcode == 5:
            # Output: combo operand mod 8
            return self.__combo_operand(step.operand) % 8, next_pointer_val

        if step.opcode == 6:
            # B = A / 2**combo operand
            self.reg.b = self.reg.a >> self.__combo_operand(step.operand)
            return None, next_pointer_val

        if step.opcode == 7:
            # C = A / 2**combo operand
            self.reg.c = self.reg.a >> self.__combo_operand(step.operand)
            return None, next_pointer_val


def extract_register(line: str):
    return int(line.split(": ")[-1])


def extract_program_sequence(line: str):
    values = line.split(": ")[-1].split(",")
    return list(map(int, values))


def extract_steps(seq: List[int]) -> List[ProgramStep]:
    return [ProgramStep(seq[i], seq[i + 1]) for i in range(0, len(seq), 2)]


def read_input(path):
    registers = []
    prog_seq = []
    with open(path, "r") as file:
        for line in file.readlines():
            if "Register" in line:
                registers.append(extract_register(line.strip()))
            elif "Program" in line:
                prog_seq = extract_program_sequence(line.strip())

    return registers, prog_seq


def part_1(path):
    register_values, program_sequence = read_input(path)

    register = Register(*register_values)

    program = Program(register, program_sequence)
    print(program.run_custom())


def part_2(path):
    _, program_sequence = read_input(path)

    target = list(program_sequence)
    wanted_output = ",".join(map(str, target))

    def run_with_a(value: int):
        register_values = [value, 0, 0]
        register = Register(*register_values)
        program = Program(register, program_sequence)
        return program.run_custom()

    def split_bits(a: int):
        if a == 0:
            return ["000"]
        bit_groups = []
        while a:
            bit_groups.append(f"{int(bin(a % 8)[2:]):03}")
            a = a >> 3

        return bit_groups[::-1]

    """ Observation
    The program behaviour is closely related to the powers of 8.
    The first 3*N bits of A determine the value of the last N outputs,
    so we can build the solution incrementally.

    Traversing the output in reversed order, we find the 3 bits
    that would output K correct values (the end of the wanted output).
    All the solutions at the previous step represent a base for 
    potential guesses for the next (actually prior) value in the output.
    """

    previous_solutions = [[]]
    for current_loc in range(len(target)):
        wanted_output = target[-current_loc - 1 :]

        next_solutions = set()
        for solution in previous_solutions:
            for option in range(8):
                option_bits = split_bits(option)

                potential_solution = list(solution) + option_bits

                # join solution with new option
                a_bits = "".join(potential_solution)
                a = int(a_bits, 2)

                # generate output
                output = run_with_a(a)

                if output == wanted_output:
                    next_solutions.add(tuple(potential_solution))

        previous_solutions = list(next_solutions)

    # convert solutions to int
    solutions = map(lambda sol: int("".join(sol), 2), previous_solutions)

    print(min(solutions))


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
