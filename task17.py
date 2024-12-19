import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from matplotlib import pyplot as plt
from tqdm import tqdm

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

    def run2(self, target=None):
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
                # return ",".join(map(str, outputs))
                return outputs

            self.reg = Register(new_a, new_b, new_c)

    def run(self, target=None):
        # print("RUNNING")
        outputs = []
        pointer = 0
        while pointer < len(self.seq):
            # print("Step:", *self.seq[pointer : pointer + 2])
            output, pointer = self.run_step(pointer)
            # print("Reg:", self.reg.get_values())

            if output is not None:
                outputs.append(output)

                if target is not None:
                    # Compare output so far with target
                    if tuple(outputs) != tuple(target[: len(outputs)]):
                        return None

            # input()

        return ",".join(map(str, outputs))

    def __combo_operand(self, operand: int):
        return [0, 1, 2, 3, *self.reg.get_values()][operand]

    def run_step(self, pointer_val: int) -> Tuple[Optional[int], int]:
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
    # program_steps = extract_steps(program_sequence)
    # print(program_steps)
    register = Register(*register_values)

    program = Program(register, program_sequence)
    print(program.run2())


def part_2(path):

    register_values, program_sequence = read_input(path)

    target = list(program_sequence)
    wanted_output = ",".join(map(str, target))

    all_outputs = []

    def run_with_a(value: int):
        register_values = [value, 0, 0]
        register = Register(*register_values)
        program = Program(register, program_sequence)
        output = program.run2()
        all_outputs.append(output)
        return

        output = program.run2(target)

        if output is None:
            return

        print("A:", value)
        print("output:", output)
        input()

        if len(output) == len(wanted_output):
            print("FOUND")

    # start = int(1e10)
    # end = int(1e11)

    # for having 0 as last value
    # start = 8**15
    # end = 8**15 + math.ceil((8**16 - 8**15) / 7)

    # for a in tqdm(range(start, end)):
    #     run_with_a(a)

    start = 1
    end = 8**3

    for a in tqdm(range(start, end)):
        run_with_a(a)

    idx = 1
    elements = [str(x[-idx]) if len(x) >= idx else -1 for x in all_outputs]

    xs = range(len(elements))
    plt.plot(xs, elements)
    plt.scatter(list(map(lambda x: 8**x, range(3))), [0] * 3)
    plt.show()


def part_22(path):
    register_values, program_sequence = read_input(path)

    target = list(program_sequence)
    wanted_output = ",".join(map(str, target))

    def run_with_a(value: int):
        register_values = [value, 0, 0]
        register = Register(*register_values)
        program = Program(register, program_sequence)
        return program.run2()

    def find_a_that_outputs_x(x, existing_solution, step):
        print("EXISTING SOL:", existing_solution)
        for a in range(8 ** (step + 1)):
            new_a = (existing_solution << (step * 3)) + a
            output = run_with_a(new_a)
            print(f"Trying with: {new_a} ({bin(new_a)[2:]}) => {output}")

            if output == x:
                yield new_a

    solutions = [0]
    for level in range(1, len(program_sequence) + 1):
        # print("So far:", a, bin(a) if a is not None else None)
        wanted_output = program_sequence[-level:]
        print("Wanted output:", wanted_output)
        print("Previous solutions:", solutions)

        new_solutions = []

        for a in solutions:
            for new_a in find_a_that_outputs_x(wanted_output, a, level - 1):
                new_solutions.append(new_a)

        solutions = new_solutions


def part_23(path):
    register_values, program_sequence = read_input(path)

    target = list(program_sequence)
    wanted_output = ",".join(map(str, target))

    def run_with_a(value: int):
        register_values = [value, 0, 0]
        register = Register(*register_values)
        program = Program(register, program_sequence)
        return program.run2()

    def split_bits(a: int):
        if a == 0:
            return ["000"]
        bit_groups = []
        while a:
            bit_groups.append(f"{int(bin(a % 8)[2:]):03}")
            a = a >> 3

        return bit_groups[::-1]

    def find_first_a_that_outputs(x):
        n = len(x)
        for a in range(8**3):
            output = run_with_a(a)
            if output[-n:] == x:
                return a

    # for output in target:
    #     find_a_that_outputs(output)
    #     break

    # all A values whose output ends with 0 begin with bits 001
    # all A values whose output ends with 3, 0 begin with bits 001, 000
    # all A values whose output ends with 3, 3, 0 begin with bits 001, 000, 110 or 001, 000, 011

    # def check_last_n_outputs(n):
    #     first_n_3_bits = set()
    #     checked = 0
    #     for a in tqdm(range(0, 8**7)):
    #         output = run_with_a(a)
    #         if output[-n:] != target[-n:]:
    #             continue

    #         checked += 1
    #         first_n_3_bits.add(tuple(split_bits(a)[:n]))

    #     print(checked)
    #     print(first_n_3_bits)

    # for n in range(1, 7):
    #     print("N:", n, target[-n:])
    #     check_last_n_outputs(n)

    # potential combinations for the first 6*3 bits:
    # options_6 = {
    #     ("001", "000", "110", "101", "110", "000"),
    #     ("001", "000", "011", "101", "101", "111"),
    #     ("001", "000", "011", "101", "101", "001"),
    #     ("001", "000", "110", "101", "110", "111"),
    # }

    # solutions_so_far = list(options_6)
    solutions_so_far = [[]]
    for current_loc in range(len(target)):
        wanted_output = target[-current_loc - 1 :]

        next_solutions = set()
        for solution in solutions_so_far:
            for option in range(8):
                option_bits = split_bits(option)
                # print("New option", option_bits)

                potential_solution = list(solution) + option_bits

                # join solution with new option
                a_bits = "".join(potential_solution)
                a = int(a_bits, 2)

                # print(a_bits, a)

                output = run_with_a(a)
                # print(output)
                # print(wanted_output)

                if output == wanted_output:
                    next_solutions.add(tuple(potential_solution))

        # print(next_solutions)
        solutions_so_far = list(next_solutions)

    for sol in solutions_so_far:
        a_bits = "".join(sol)
        a = int(a_bits, 2)
        output = run_with_a(a)
        print(a, output)

    return

    solutions_so_far = []
    for n in range(1, len(target) + 1):
        last_n_digits = target[-n:]
        print(f"Last {n} digits: {last_n_digits}")

        # If we find an A value whose output ends with last_n_digits, we need to use the first N*3 bits of A.
        first_a = find_first_a_that_outputs(last_n_digits)
        print(f"First A that outputs {last_n_digits}: {split_bits(first_a)}")
        break

    # first 3 bits = last digit
    # last 3 bits = first digit


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    # part_22(INPUT_DATA_PATH)
    part_23(INPUT_DATA_PATH)
