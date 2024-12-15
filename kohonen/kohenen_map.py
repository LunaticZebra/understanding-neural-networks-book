from random import gauss
from typing import Callable
from kohonen_layer_topologies import Topology
from hexagonal import *


def get_top_bottom_neighbours(neurode: tuple[int, int], rows: int, shift: int) -> list[tuple[int, int]]:
    neighbours = []

    upper_neighbor = neurode[1] - shift
    bottom_neighbor = neurode[1] + shift
    if upper_neighbor >= 0:
        neighbours.append((neurode[0], upper_neighbor))
    if bottom_neighbor < rows:
        neighbours.append((neurode[0], bottom_neighbor))

    return neighbours


def get_left_right_neighbours(neurode: tuple[int, int], columns: int, shift: int) -> list[tuple[int, int]]:
    neighbours = []

    left_neighbour = neurode[0] - shift
    right_neighbour = neurode[0] + shift
    if left_neighbour >= 0:
        neighbours.append((left_neighbour, neurode[1]))
    if right_neighbour < columns:
        neighbours.append((right_neighbour, neurode[1]))

    return neighbours


def learning_law(learning_rate: float, input_pattern: list[float], weight_vector: list[float]):
    weight_change_vector = []
    for i in range(len(input_pattern)):
        weight_change_vector.append(learning_rate*(input_pattern[i] - weight_vector[i]))

    return weight_change_vector


class KohonenNetwork:
    def __init__(self, input_layer_size: int, kohonen_layer_size: int, shape: tuple[int, int], topology_type: Topology):
        self.input_layer_weights = [[gauss(0, 1) for _ in range(input_layer_size)] for _ in range(kohonen_layer_size)]
        self.kohonen_layer_size = kohonen_layer_size
        self.input_layer_size = input_layer_size
        self.shape = shape
        self.topology_type = topology_type

    def process_input(self, input_pattern: list[float]) -> int:

        max_weighted_input = 0
        kohonen_neurode = 0

        for i in range(len(self.input_layer_weights)):
            weighted_input = 0
            neurode = self.input_layer_weights[i]
            for j in range(len(neurode)):
                weight = neurode[j]
                weighted_input += weight * input_pattern[j]

            if weighted_input > max_weighted_input:
                kohonen_neurode = i
                max_weighted_input = weighted_input

        return kohonen_neurode

    def adjust_weights(self, winning_neurode: int, neighbourhood_size: int, input_pattern: list[float],
                       learning_rate: float, distance_function: Callable[[tuple[int, int], tuple[int,int]], float]) -> None:

        neurodes_to_adjust = []

        width, height = self.shape

        neurode = (winning_neurode % width, round(winning_neurode / width))

        neurodes_to_adjust.extend(self.get_neighbours(neurode, neighbourhood_size))

        for n in neurodes_to_adjust:
            weight_change_multiplier = distance_function(neurode, n)
            n_index = n[0] * height + n[1]
            delta_vector = learning_law(learning_rate, input_pattern, self.input_layer_weights[n_index])
            for i in range(len(self.input_layer_weights[n_index])):
                self.input_layer_weights[n_index][i] += delta_vector[i] * weight_change_multiplier

    def get_neighbours(self, neurode: tuple[int, int], neighbourhood: int):
        columns, rows = self.shape
        neighbours = []
        if self.topology_type == Topology.LINEAR:
            for i in range(neighbourhood):
                neighbours.extend(get_left_right_neighbours(neurode, columns, i+1))

        elif self.topology_type == Topology.RECTANGLE:

            for i in range(neighbourhood):
                neighbours.extend(get_left_right_neighbours(neurode, columns, i+1))
                neighbours.extend(get_top_bottom_neighbours(neurode, rows, i+1))

        elif self.topology_type == Topology.HEXAGONAL:
            grid = HexGrid(columns, rows)
            neighbours.extend(list(grid.get_neighbourhood(neurode[0], neurode[1], neighbourhood)))

        return neighbours


    def train_network(self, input_patterns: list[list[float]], learning_rate: float, neighbourhood: int):

        for pattern in input_patterns:
            winning_neurode = self.process_input(pattern)
            self.adjust_weights(winning_neurode, neighbourhood, pattern, learning_rate, euclidean_distance)
            
    def predict_labels(self, input_patterns: list[list[float]]) -> list[int]:
        labels = []

        for pattern in input_patterns:
            labels.append(self.process_input(pattern))

        return labels
def euclidean_distance(n1: tuple[int, int], n2: tuple[int, int]) -> float:
    x1, y1 = n1
    x2, y2 = n2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5


if __name__ == "__main__":
    network = KohonenNetwork(10, 18, (6, 3), Topology.HEXAGONAL)