from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

INPUT_DATA_PATH = "input_data/06.txt"

FREE_CELL = 0
OCCUPIED_CELL = -1
VISITED_CELL = 1


char_to_int = {".": FREE_CELL, "#": OCCUPIED_CELL}


class Orientation(Enum):
    up = 0
    right = 1
    down = 2
    left = 3

    def get_delta(self):
        return [[-1, 0], [0, 1], [1, 0], [0, -1]][self.value]

    def get_incoming_delta(self):
        return -np.array(self.get_delta())

    def get_rotated(self, x=1):
        return Orientation((self.value + x) % 4)

    def get_mark(self):
        return self.value + 1

    def clone(self):
        return Orientation(self.value)


START_POSITION_CHAR = "^"
START_ORIENTATION = Orientation.up


@dataclass
class Position:
    row_col: np.ndarray
    orientation: Orientation

    def get_next_row_col(self):
        return self.row_col + np.array(self.orientation.get_delta())

    def clone(self):
        return Position(np.copy(self.row_col), self.orientation.clone())

    def to_tuple(self):
        return tuple([*self.row_col, self.orientation.value])


class MapAgent:
    OUTSIDE = -1
    NEW_CELL = 0
    CROSSING = 1
    LOOP = 2

    def __init__(self, map: np.ndarray, start_pos: Position):
        self.map = np.copy(map)
        self.start_pos = start_pos
        self.current_pos = start_pos.clone()
        self.cells_visited = 0
        self.mark_current_cell_as_visited()

    def mark_current_cell_as_visited(self):
        if not self.current_pos_is_in_grid():
            return MapAgent.OUTSIDE

        row, col = self.current_pos.row_col

        current_value = self.map[row][col]
        new_value = current_value + 1

        self.map[row][col] = new_value

        if new_value > 4:
            return MapAgent.LOOP

        if current_value > 0:
            return MapAgent.CROSSING

        if current_value == 0:
            self.cells_visited += 1
            return MapAgent.NEW_CELL

    def current_pos_is_in_grid(self):
        return self.row_col_is_in_grid(self.current_pos.row_col)

    def row_col_is_in_grid(self, row_col):
        width, height = self.map.shape
        row, col = row_col
        return 0 <= col < width and 0 <= row < height

    def cell_is_occupied(self, row_col):
        if not self.row_col_is_in_grid(row_col):
            return False
        row, col = row_col
        return self.map[row][col] == OCCUPIED_CELL

    def step(self):
        if not self.current_pos_is_in_grid():
            print("Current position is not in grid.")
            return

        # print("Current pos:", self.current_pos)

        next_row_col = self.current_pos.get_next_row_col()
        if self.cell_is_occupied(next_row_col):
            # rotate
            self.current_pos.orientation = self.current_pos.orientation.get_rotated()
        else:
            # move
            self.current_pos.row_col = next_row_col

        return self.mark_current_cell_as_visited()

    def get_turning_point(self):
        if not self.current_pos_is_in_grid():
            print("Current position is not in grid.")
            return

        while True:
            next_row_col = self.current_pos.get_next_row_col()
            if not self.row_col_is_in_grid(next_row_col):
                # stop, exit reached
                return self.current_pos, True

            if self.cell_is_occupied(next_row_col):
                # stop, exit not reached
                return self.current_pos, False

            # move
            self.current_pos.row_col = next_row_col

    def run(self):
        steps = []
        while True:
            steps.append(self.current_pos.clone())
            # print(len(steps))

            new_cell_type = self.step()

            if new_cell_type in (MapAgent.OUTSIDE, MapAgent.LOOP):
                # stop when outside or when loop has been created
                return steps, new_cell_type


def create_row(line: str):
    line: list = list(line.strip())
    start_col_id = None
    if START_POSITION_CHAR in line:
        start_col_id = line.index(START_POSITION_CHAR)
        line[start_col_id] = "."

    raw_elements = list(map(lambda x: char_to_int[x], line))
    return raw_elements, start_col_id


def read_input(path):
    grid = []
    start_pos = None
    with open(path, "r") as file:
        for row_id, line in enumerate(file.readlines()):
            row_contents, start_col_id = create_row(line)
            if start_col_id is not None:
                start_pos = Position(
                    row_col=np.array([row_id, start_col_id]),
                    orientation=Orientation.up,
                )

            grid.append(row_contents)

    return np.array(grid), start_pos


def part_1(path: str):
    map, start_pos = read_input(path)
    agent = MapAgent(map, start_pos)

    agent.run()

    print(agent.cells_visited)


@dataclass
class TurningPoint:
    row_col: np.ndarray
    incoming_orientation: Orientation
    next_stop: "TurningPoint" = None
    exit: bool = False
    prev_tp: "TurningPoint" = None

    def __hash__(self):
        return hash(self.to_tuple())

    def to_tuple(self):
        return tuple([*self.row_col, self.incoming_orientation.value])

    def __eq__(self, other):
        return hash(self) == hash(other)

    def to_position(self):
        return Position(self.row_col, self.incoming_orientation.get_rotated())

    def set_next_stop(self, next_tp: "TurningPoint"):
        self.next_stop = next_tp
        next_tp.prev_tp = self

    def __repr__(self):
        return f"TP({list(self.row_col)}, in_orientation={self.incoming_orientation})"


class Map:
    def __init__(self, data: np.ndarray):
        self.data = data

    def __get_row_col(self, args):
        if len(args) == 1:
            return args[0]

        if len(args) == 2:
            return args

    def row_col_is_in_grid(self, *args):
        row, col = self.__get_row_col(args)

        width, height = self.data.shape
        return 0 <= col < width and 0 <= row < height

    def get_value(self, *args):
        row, col = self.__get_row_col(args)

        return self.data[row][col]

    def set_value(self, value, *args):
        row, col = self.__get_row_col(args)

        self.data[row][col] = value

    # def get_neighbors(self, row_col):
    #     output = []
    #     deltas = np.array([
    #         [-1, 0],
    #         [0, 1],
    #         [1, 0],
    #         [0, -1]
    #     ])
    #     for d in deltas:
    #         new_row_col = row_col + d
    #         if self.row_col_is_in_grid(new_row_col):


def create_turning_points(map: np.ndarray):
    tps: Dict[tuple, TurningPoint] = {}

    m = Map(map)

    orientations = [Orientation(i) for i in range(4)]
    for row in range(map.shape[0]):
        for col in range(map.shape[1]):
            if m.get_value(row, col) != OCCUPIED_CELL:
                continue

            for o in orientations:
                tp_row_col = np.array([row, col]) + o.get_incoming_delta()
                if (
                    m.row_col_is_in_grid(tp_row_col)
                    and m.get_value(tp_row_col) == FREE_CELL
                ):
                    new_tp = TurningPoint(tp_row_col, o, exit=False)
                    tps[new_tp.to_tuple()] = new_tp

    exit_tps: Dict[tuple, TurningPoint] = {}

    for tp_tuple, tp in tps.items():
        agent = MapAgent(map, tp.to_position())
        next_turn_position, is_exit = agent.get_turning_point()

        new_tp = TurningPoint(
            next_turn_position.row_col, next_turn_position.orientation, exit=is_exit
        )

        if new_tp.to_tuple() in tps:
            assert not is_exit

            # Found an existing tp, create the link
            tp.set_next_stop(tps[new_tp.to_tuple()])

        if new_tp not in tps:
            assert is_exit

            exit_tps[new_tp.to_tuple()] = new_tp
            tp.set_next_stop(new_tp)

    # viz_map = np.copy(map)
    # viz_map = viz_map * 100

    # for tp in tps.values():
    #     row, col = tp.row_col
    #     viz_map[row][col] = 50 + 15 * tp.incoming_orientation.value

    # plt.imshow(viz_map)
    # plt.colorbar()
    # plt.show()

    return tps, exit_tps


def obstacle_adds_loop(
    map,
    obstacle_pos: np.ndarray,
    tps: Dict[tuple, TurningPoint],
    exit_tps: Dict[tuple, TurningPoint],
    start_pos: Position,
):
    new_map = Map(np.copy(map))

    new_tps: Dict[tuple, TurningPoint] = {}
    new_exit_tps: Dict[tuple, TurningPoint] = {}

    new_links_next: Dict[tuple, TurningPoint] = {}
    new_links_prev: Dict[tuple, TurningPoint] = {}

    def add_temp_link(prev: TurningPoint, next: TurningPoint):
        new_links_next[prev.to_tuple()] = next
        new_links_prev[next.to_tuple()] = prev

    def get_prev(tp: TurningPoint):
        new_prev = new_links_prev.get(tp.to_tuple(), None)
        if new_prev is not None:
            return new_prev

        return tp.prev_tp

    def get_next(tp: TurningPoint):
        new_next = new_links_next.get(tp.to_tuple(), None)
        if new_next is not None:
            return new_next

        return tp.next_stop

    def get_tp(tup: tuple):
        for possible_dict in (tps, exit_tps, new_tps, new_exit_tps):
            tp = possible_dict.get(tup, None)
            if tp is not None:
                return tp

        return None

    new_map.set_value(OCCUPIED_CELL, obstacle_pos)

    orientations = [Orientation(i) for i in range(4)]
    for o in orientations:
        tp_row_col = obstacle_pos + o.get_incoming_delta()
        if (
            new_map.row_col_is_in_grid(tp_row_col)
            and new_map.get_value(tp_row_col) == FREE_CELL
        ):

            new_tp = TurningPoint(tp_row_col, o)
            new_tps[new_tp.to_tuple()] = new_tp

    # for new_tp in new_tps.values():
    #     # find next stop
    #     agent = MapAgent(new_map.data, new_tp.to_position())
    #     next_turn_pos, is_exit = agent.get_turning_point()
    #     next_turn_pos_tuple = next_turn_pos.to_tuple()

    #     if not is_exit:
    #         assert next_turn_pos_tuple in tps

    #         # next stop already exists, just link it
    #         add_temp_link(new_tp, tps[next_turn_pos_tuple])
    #     else:
    #         # next stop is exit

    #         if next_turn_pos_tuple in exit_tps:
    #             # exit has already been discovered
    #             add_temp_link(new_tp, exit_tps[next_turn_pos_tuple])
    #         elif next_turn_pos_tuple in new_exit_tps:
    #             add_temp_link(new_tp, new_exit_tps[next_turn_pos_tuple])
    #         else:
    #             # new exit

    #             new_exit_tp = TurningPoint(
    #                 next_turn_pos.row_col,
    #                 next_turn_pos.orientation,
    #                 exit=is_exit,
    #             )
    #             new_exit_tps[new_exit_tp.to_tuple()] = new_exit_tp

    #             add_temp_link(new_tp, new_exit_tp)

    #     # find tps for which current should be next tp:
    #     # place agent at current obstacle, with incoming orientation
    #     agent = MapAgent(
    #         new_map.data,
    #         Position(obstacle_pos, new_tp.incoming_orientation),
    #     )
    #     next_turn_pos, is_exit = agent.get_turning_point()

    #     # extract prev of next
    #     next_turn_pos_tuple = next_turn_pos.to_tuple()
    #     next_tp = get_tp(next_turn_pos_tuple)

    #     if next_tp is not None:
    #         prev_of_next = get_prev(next_tp)

    #         if prev_of_next is not None:

    #             current_dist = np.linalg.norm(next_tp.row_col - prev_of_next.row_col)
    #             new_dist = np.linalg.norm(next_tp.row_col - new_tp.row_col)
    #             if current_dist > new_dist:
    #                 add_temp_link(prev_of_next, new_tp)

    def edge_crosses_obstacle(current_tp: TurningPoint, next_tp: TurningPoint):
        def get_min_max(a, b):
            return (a, b) if a < b else (b, a)

        r0, c0 = current_tp.row_col
        r1, c1 = next_tp.row_col
        r_obstacle, c_obstacle = obstacle_pos

        if r0 == r1 and r_obstacle == r0:
            # check col
            c_min, c_max = get_min_max(c0, c1)
            return c_min <= c_obstacle <= c_max

        if c0 == c1 and c_obstacle == c0:
            # check row
            r_min, r_max = get_min_max(r0, r1)
            return r_min <= r_obstacle <= r_max

    def fix_edge(current_tp: TurningPoint, next_tp: TurningPoint):

        def find_next_tp(tp: TurningPoint):
            agent = MapAgent(new_map.data, tp.to_position())
            turn_pos, _ = agent.get_turning_point()

            return turn_pos, get_tp(turn_pos.to_tuple())

        # create new TP where the edge intersects the obstacle

        unit_vector = current_tp.row_col - next_tp.row_col
        unit_vector = unit_vector // int(np.linalg.norm(unit_vector, ord=1))

        intersection_tp = TurningPoint(
            row_col=obstacle_pos + unit_vector,
            incoming_orientation=next_tp.incoming_orientation,
            prev_tp=current_tp,
        )

        tps_to_set_next = [intersection_tp]

        while tps_to_set_next:
            # try for the latest in the list
            turn_pos, next_tp = find_next_tp(tps_to_set_next[-1])
            if next_tp is not None:
                tps_to_set_next[-1].next_stop = next_tp
                tps_to_set_next = tps_to_set_next[:-1]
            else:
                # create new tp at turn_pos
                extra_tp = TurningPoint(
                    row_col=turn_pos.row_col,
                    incoming_orientation=turn_pos.orientation,
                )
                new_tps[extra_tp.to_tuple()] = extra_tp
                tps_to_set_next.append(extra_tp)

        return intersection_tp

    turn_pos_blind, is_exit = MapAgent(map, start_pos).get_turning_point()
    current_tp = TurningPoint(
        row_col=start_pos.row_col,
        incoming_orientation=start_pos.orientation.get_rotated(-1),
        next_stop=get_tp(turn_pos_blind.to_tuple()),
    )

    # turn_pos, is_exit = MapAgent(new_map.data, start_pos).get_turning_point()
    # current_tp = get_tp(turn_pos.to_tuple())
    # if is_exit:
    #     return False

    visited_tps = [current_tp.to_tuple()]

    print("START TP:", current_tp)

    while True:
        next_tp = get_next(current_tp)
        # print("----")
        # print("OBSTACLE:", obstacle_pos)

        # print("current:", current_tp)
        # print("next:", next_tp)

        if next_tp is None:
            raise RuntimeError

        if edge_crosses_obstacle(current_tp, next_tp):
            next_tp = fix_edge(current_tp, next_tp)
            print("fixed next:", next_tp)

        if next_tp.to_tuple() in visited_tps:
            return visited_tps, True
        if next_tp.exit:
            return visited_tps, False

        visited_tps.append(next_tp.to_tuple())

        current_tp = next_tp


def part_2(path):
    map, start_pos = read_input(path)

    tps, exit_tps = create_turning_points(map)

    agent = MapAgent(map, start_pos)

    steps, _ = agent.run()
    steps: list[Position]

    locs = list(set(tuple(step.row_col) for step in steps[1:]))
    # locs = [(113, 28)]

    loops_v2 = 0

    for loc in tqdm(locs):
        steps_v2, v2_answer = obstacle_adds_loop(
            map, np.array(loc), tps, exit_tps, start_pos
        )
        if v2_answer:
            loops_v2 += 1

    print(loops_v2)


if __name__ == "__main__":
    # part_1(INPUT_DATA_PATH)
    part_2(INPUT_DATA_PATH)
