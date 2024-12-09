from dataclasses import dataclass

INPUT_DATA_PATH = "input_data/09.txt"
# INPUT_DATA_PATH = "input_data/09_test.txt"


@dataclass
class MemoryBlock:
    id: int
    value: int = None

    def is_free(self):
        return self.value is None

    def clear(self):
        self.value = None

    def copy_value(self, other_block: "MemoryBlock"):
        self.value = other_block.value

    def checksum(self):
        if self.is_free():
            return None

        return self.id * self.value

    def __repr__(self):
        val_str = "    " if self.is_free() else self.value
        return f"<id:{self.id} [{val_str}]>"


class Memory:
    def __init__(self, raw_data: list[int]):
        self.blocks: list[MemoryBlock] = create_memory(raw_data)
        print(f"Created memory space with {len(self.blocks)} blocks.")

    def first_free_block(self, start_index: int):
        i = start_index
        while not self.blocks[i].is_free():
            i += 1

        return self.blocks[i]

    def last_occupied_block(self, start_index: int):
        i = start_index
        while self.blocks[i].is_free():
            i -= 1

        return self.blocks[i]

    def reorganise(self):
        free_block = self.first_free_block(0)
        occupied_block = self.last_occupied_block(len(self.blocks) - 1)

        swaps = 0

        while occupied_block.id > free_block.id:
            # move the contents of occupied into free
            free_block.copy_value(occupied_block)
            occupied_block.clear()

            swaps += 1

            # find next occupied id
            occupied_block = self.last_occupied_block(occupied_block.id - 1)

            # find next free id
            free_block = self.first_free_block(free_block.id + 1)

        print(f"performed {swaps} swaps")

    def checksum(self):
        return sum([block.checksum() for block in self.blocks if not block.is_free()])


def read_input(path):
    with open(path, "r") as file:
        return list(map(int, file.readline().strip()))


def create_memory(raw_data):
    blocks: list[MemoryBlock] = []

    current_value = 0
    free = False

    for x in raw_data:
        val = None if free else current_value
        for _ in range(x):
            block = MemoryBlock(id=len(blocks), value=val)
            blocks.append(block)

        if not free:
            current_value += 1

        # alternate between free and not free
        free = not free

    return blocks


def part_1(path):
    raw_data = read_input(path)

    mem = Memory(raw_data)
    mem.reorganise()

    print(mem.checksum())


if __name__ == "__main__":
    part_1(INPUT_DATA_PATH)
