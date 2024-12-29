import math
import random
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
        weight_change_vector.append(learning_rate * (input_pattern[i] - weight_vector[i]))

    return weight_change_vector


def euclidean_distance(x: list[float], y: list[float]) -> float:
    distance = 0
    for i in range(len(x)):
        distance += (x[i] - y[i]) ** 2

    return distance ** 0.5


def manhattan_distance(x: list[float], y: list[float]) -> float:
    distance = 0

    for i in range(len(x)):
        distance += abs(x[i] - y[i])

    return distance


class KohonenNetwork:
    def __init__(self, input_layer_size: int, kohonen_layer_size: int, shape: tuple[int, int], topology_type: Topology,
                 distance_function: Callable[[list[float], list[float]], float] = euclidean_distance):

        self.width = shape[0]
        self.height = shape[1]

        self.input_layer_weights = [
            [
                [random.uniform(0, 1) for _ in range(input_layer_size)]
                for _ in range(self.width)
            ]
            for _ in range(self.height)
        ]

        self.distance_function = distance_function
        self.kohonen_layer_size = kohonen_layer_size
        self.input_layer_size = input_layer_size
        self.shape = shape
        self.topology_type = topology_type

    def process_input(self, input_pattern: list[float]) -> tuple[int, int]:

        max_weighted_input = -float('inf')
        kohonen_neurode = (0, 0)

        for y in range(self.height):
            for x in range(self.width):
                weighted_input = 0
                neurode = self.input_layer_weights[y][x]
                for i in range(len(neurode)):
                    weight = neurode[i]
                    weighted_input += weight * input_pattern[i]

                if weighted_input > max_weighted_input:
                    kohonen_neurode = (x, y)
                    max_weighted_input = weighted_input

        return kohonen_neurode


    def adjust_weights(self, winning_neurode: tuple[int, int], neighbourhood_size: int, input_pattern: list[float],
                       learning_rate: float) -> None:

        neurodes_to_adjust = []


        neurodes_to_adjust.extend(self.get_neighbours(winning_neurode, neighbourhood_size))

        for n in neurodes_to_adjust:

            distance = self.distance_function(self.input_layer_weights[winning_neurode[1]][winning_neurode[0]],
                                         self.input_layer_weights[n[1]][n[0]])
            weight_change_multiplier = math.e ** -((distance ** 2) / (neighbourhood_size ** 2))
            delta_vector = learning_law(learning_rate, input_pattern, self.input_layer_weights[n[1]][n[0]])
            for i in range(len(self.input_layer_weights[n[1]][n[0]])):
                self.input_layer_weights[n[1]][n[0]][i] += delta_vector[i] * weight_change_multiplier


        delta_vector = learning_law(learning_rate, input_pattern,
                                    self.input_layer_weights[winning_neurode[1]][winning_neurode[0]])

        for i in range(len(self.input_layer_weights[winning_neurode[1]][winning_neurode[0]])):
            self.input_layer_weights[winning_neurode[1]][winning_neurode[0]][i] += delta_vector[i]


    def get_neighbours(self, neurode: tuple[int, int], neighbourhood: int):
        columns, rows = self.shape
        neighbours = []
        if self.topology_type == Topology.LINEAR:
            for i in range(neighbourhood):
                neighbours.extend(get_left_right_neighbours(neurode, columns, i + 1))

        elif self.topology_type == Topology.RECTANGLE:

            for i in range(neighbourhood):
                neighbours.extend(get_left_right_neighbours(neurode, columns, i + 1))
                neighbours.extend(get_top_bottom_neighbours(neurode, rows, i + 1))

        elif self.topology_type == Topology.HEXAGONAL:
            grid = HexGrid(columns, rows)
            neighbours.extend(list(grid.get_neighbourhood(neurode[0], neurode[1], neighbourhood)))

        return neighbours

    def train_network(self, input_patterns: list[list[float]], learning_rate: float, neighbourhood: int,
                      num_of_epochs: int,
                      lr_decay: float, neighbourhood_decay: bool = False):

        for epoch in range(num_of_epochs):
            if neighbourhood_decay:
                neighbourhood = math.ceil(neighbourhood * (1 - (epoch / num_of_epochs)))


            for pattern in input_patterns:
                winning_neurode = self.process_input(pattern)
                self.adjust_weights(winning_neurode, neighbourhood, pattern, learning_rate)

            learning_rate *= lr_decay

    def predict_labels(self, input_patterns: list[list[float]]) -> list[int]:
        labels = []

        for pattern in input_patterns:
            winning_neurode = self.process_input(pattern)
            labels.append(winning_neurode[1] * self.width + winning_neurode[0])

        return labels
