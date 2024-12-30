from functools import reduce
from typing import Dict, Set

from tqdm import tqdm

INPUT_DATA_PATH = "input_data/23_test.txt"
INPUT_DATA_PATH = "input_data/23.txt"


def read_input(path):
    edges: Dict[str, Set] = dict()

    with open(path, "r") as file:
        for line in file.readlines():
            n1, n2 = line.strip().split("-")

            edges.setdefault(n1, set())
            edges[n1].add(n2)

            edges.setdefault(n2, set())
            edges[n2].add(n1)

    return edges


def part_1(path):
    edges = read_input(path)

    solutions = set()
    for node, edge_set in edges.items():
        if not node.startswith("t"):
            continue

        for node2 in edge_set:
            # find node3 that is connected to node and node2

            for node3 in edge_set:
                if node3 == node2:
                    continue

                # check if node3 is connected to node2
                if node3 in edges[node2]:
                    sol = tuple(sorted([node, node2, node3]))
                    solutions.add(sol)

    print(len(solutions))


def part_2(path):
    edges = read_input(path)

    solutions = dict()
    for node in tqdm(edges.keys()):
        potential_sols = set([(node,)])
        while potential_sols:
            new_sols = set()
            for potential_sol in potential_sols:
                ln = len(potential_sol)
                solutions.setdefault(ln, set())
                solutions[ln].add(potential_sol)
                # try adding a node to the solution

                # a good node is connected to all nodes in potential_sol
                good_nodes = reduce(
                    lambda x, y: x.intersection(y),
                    [edges[n] for n in potential_sol],
                )

                for good_node in good_nodes:
                    new_sol = tuple(sorted(list(potential_sol) + [good_node]))

                    if new_sol not in solutions.get(len(new_sol), []):
                        # already checked, no need to repeat
                        new_sols.add(new_sol)

            potential_sols = new_sols

    longest = list(solutions[max(solutions)])[0]
    print(",".join(sorted(longest)))


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
