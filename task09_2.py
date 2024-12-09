from bisect import insort
from dataclasses import dataclass
from typing import Dict

from task09_1 import read_input


@dataclass
class ContiguousMemoryBlock:
    start_id: int
    size: int
    value: int

    def checksum(self):
        idxs = range(self.start_id, self.start_id + self.size)
        return sum([self.value * idx for idx in idxs])


def map_memory(raw_data):
    """
    Create a list of files and a dictionary that stores gaps grouped by size,
    in ascending order of their location.
    """
    files = []
    gaps = {x: [] for x in range(0, 10)}
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
            start_id=current_id,
            size=block_size,
            value=None if free else current_value,
        )
        current_id += block_size

        if free:
            gaps[block_size].append(block)
        else:
            files.append(block)

    return files, gaps


def organise_memory(
    files: list[ContiguousMemoryBlock], gaps: Dict[int, list[ContiguousMemoryBlock]]
):

    def fill_gap(file: ContiguousMemoryBlock):
        """Find the earliest gap that fits the given file and place the file there."""

        candidate_gaps = [
            gaps[size][0]
            for size in range(file.size, 10)
            if gaps[size] and gaps[size][0].start_id < file.start_id
        ]

        if not candidate_gaps:
            # Nothing to do, file is not moved
            return

        earliest_gap = sorted(candidate_gaps, key=lambda gap: gap.start_id)[0]

        gaps[earliest_gap.size].pop(0)

        target_gap = earliest_gap

        # move file
        file.start_id = target_gap.start_id

        # shrink gap
        target_gap.start_id += file.size
        target_gap.size -= file.size

        # re-assing gap in the dict based on its new size and location
        if target_gap.size > 0:
            insort(gaps[target_gap.size], target_gap, key=lambda gap: gap.start_id)

    for file in reversed(files):
        fill_gap(file)

    return files, gaps


def part_2(path):
    raw_data = read_input(path)

    files, gaps = map_memory(raw_data)
    files, gaps = organise_memory(files, gaps)

    checksum = sum(map(lambda file: file.checksum(), files))
    print(checksum)


if __name__ == "__main__":
    # part_2("input_data/09_test.txt")
    part_2("input_data/09.txt")
