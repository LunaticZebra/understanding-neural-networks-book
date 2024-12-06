from random import gauss
from typing import Callable
from kohonen_layer_topologies import Topology


def create_kohonen(input_layer_size: int, kohonen_layer_size: int,
                   distance_function: Callable[[[list[float], list[float]]], float],
                   shape: tuple[int, int],topology_type: Topology = Topology.RECTANGLE, neighbourhood: int = 2):

    # input layer serves as a fan-out layer, which distributes the inputs to all of kohonen-layer neurodes
    input_weights = [[gauss(0, 1) for _ in range(kohonen_layer_size)] for _ in range(input_layer_size)]
    # []
    # kohonen layer is connected but those are not weighted connections
    kohonen_layer = [[]]
    if topology_type == Topology.RECTANGLE:





def get_neighbours(neurode: tuple[int, int], shape: tuple[int, int], topology_type: Topology, neighbourhood: int) -> list[int]:
    rows, columns = shape
    neighbours = []
    if topology_type == Topology.LINEAR:
        for i in range(neighbourhood):
            neighbours.append(get_left_right_neighbours(neurode, columns, i))


    elif topology_type == Topology.RECTANGLE:

        for i in range(neighbourhood):
            neighbours.append(get_left_right_neighbours(neurode, columns, i))
            neighbours.append(get_top_bottom_neighbours(neurode, rows, i))

    elif topology_type == Topology.HEXAGONAL:


def get_left_right_neighbours(neurode: tuple[int, int], columns: int, shift: int) -> list[tuple[int,int]]:
    neighbours = []

    left_neighbour = neurode[0] - shift
    right_neighbour = neurode[0] + shift

    if left_neighbour >= 0:
        neighbours.append((left_neighbour, neurode[1]))
    if right_neighbour < columns:
        neighbours.append((right_neighbour, neurode[1]))

    return neighbours

def get_top_bottom_neighbours(neurode: tuple[int,int], rows: int, shift: int) -> list[tuple[int,int]]:

    neighbours = []

    upper_neighbor = neurode[1] - shift
    bottom_neighbor = neurode[1] + shift
    if upper_neighbor >= 0:
        neighbours.append((neurode[0], upper_neighbor))
    if bottom_neighbor < rows:
        neighbours.append((neurode[0], bottom_neighbor))

    return neighbours
def calculate_weight_change(learning_rate: float, input: list[float], weight: list[float]) -> list[float]:
    weight_change = []
    if len(input) != len(weight):
        raise Exception("Input and weights have different lengths!")

    for i in range(len(input)):
        weight_change.append(learning_rate * (input[i] - weight[i]))

    return weight_change


if __name__ == "__main__":
    create_kohonen(10, 5)
