import numpy as np

# INPUT_DATA_PATH = "input_data/08_test.txt"
INPUT_DATA_PATH = "input_data/08.txt"

EMPTY_CELL = "."


def read_input(path):
    with open(path, "r") as file:
        return np.array([list(line.strip()) for line in file.readlines()])


def extract_nodes(map_data: np.ndarray):
    nodes = {}
    for row_id in range(map_data.shape[0]):
        for col_id, val in enumerate(map_data[row_id]):
            if val == EMPTY_CELL:
                continue
            nodes.setdefault(val, [])
            nodes[val].append((row_id, col_id))

    return nodes


def location_is_in_map(loc: np.ndarray, map_shape: tuple):
    return 0 <= loc[0] < map_shape[0] and 0 <= loc[1] < map_shape[1]


def compute_antinodes_1(node_locations: list, map_shape: tuple):
    length = len(node_locations)
    antinodes = set()
    for i in range(length):
        for j in range(i + 1, length):
            n1 = np.array(node_locations[i])
            n2 = np.array(node_locations[j])

            delta = n2 - n1
            for antinode in (n1 - delta, n2 + delta):
                if location_is_in_map(antinode, map_shape):
                    antinodes.add(tuple(antinode))

    return antinodes


def compute_antinodes_2(node_locations: list, map_shape: tuple):
    length = len(node_locations)
    antinodes = set()
    for i in range(length):
        for j in range(i + 1, length):
            n1 = np.array(node_locations[i])
            n2 = np.array(node_locations[j])

            delta = n2 - n1

            new_loc = n1
            while location_is_in_map(new_loc, map_shape):
                antinodes.add(tuple(new_loc))
                new_loc -= delta

            new_loc = n2
            while location_is_in_map(new_loc, map_shape):
                antinodes.add(tuple(new_loc))
                new_loc += delta

    return antinodes


def compute_antinode_locations(nodes: dict, map_shape: tuple, compute_method):
    return {
        node_id: compute_method(node_locs, map_shape)
        for node_id, node_locs in nodes.items()
    }


def part_1(path):
    map_contents = read_input(path)
    nodes = extract_nodes(map_contents)

    antinodes = compute_antinode_locations(
        nodes, map_contents.shape, compute_antinodes_1
    )

    all_antinodes = set()
    for antinodes_list in antinodes.values():
        all_antinodes.update(antinodes_list)

    print(len(all_antinodes))


def part_2(path):
    map_contents = read_input(path)
    nodes = extract_nodes(map_contents)

    antinodes = compute_antinode_locations(
        nodes, map_contents.shape, compute_antinodes_2
    )

    all_antinodes = set()
    for antinodes_list in antinodes.values():
        all_antinodes.update(antinodes_list)

    print(len(all_antinodes))


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
