from dataclasses import dataclass

from task09_1 import read_input


@dataclass
class ContiguousMemoryBlock:
    start_id: int
    size: int
    value: int


def create_memory_files(raw_data):
    files = []
    gaps = []
    current_id = 0
    current_value = -1
    free = True
    for block_size in raw_data:
        # alternate between free and not free
        free = not free

        if not free:
            current_value += 1

        if block_size == 0:
            # skip empty blocks
            continue

        block = ContiguousMemoryBlock(
            start_id=current_id, size=block_size, value=None if free else current_value
        )
        current_id += block_size

        if free:
            gaps.append(block)
        else:
            files.append(block)

    return files, gaps


def organise_memory(files, gaps):
    first_gap_id = 0

    for file in reversed(files):
        print(file)


def part_2(path):
    raw_data = read_input(path)

    files, gaps = create_memory_files(raw_data)

    print("files:", len(files))
    print("gaps:", len(gaps))

    for f in files:
        print(f)


if __name__ == "__main__":
    part_2("input_data/09_test.txt")
