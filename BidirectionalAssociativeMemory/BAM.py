import copy
from numpy import array
import numpy as np

def _bipolorize_pattern(pattern: array):

    bipolarized_pattern = copy.deepcopy(pattern)

    for i in range(pattern.shape[0]):
        for j in range(pattern.shape[1]):
            if pattern[i][j] == 0:
                bipolarized_pattern[i][j] = -1

    return bipolarized_pattern

class BAM:
    def __init__(self, A_pattern_shape: tuple[int,int], B_pattern_shape: tuple[int,int]):
        self.A_width = A_pattern_shape[0]
        self.A_height = A_pattern_shape[1]
        self.B_width = B_pattern_shape[0]
        self.B_height = B_pattern_shape[1]
        self.weight_matrix = np.zeros((self.A_width * self.A_height, self.B_width * self.B_height)) #

    def store_pattern(self, input_pattern: list[list[int]], output_pattern: list[list[int]]):
        pattern = _bipolorize_pattern(input_pattern).flatten()
        outer_product = np.outer(pattern, np.array(output_pattern).flatten())
        self.weight_matrix += outer_product

    def recall_pattern(self, pattern: list[list[int]], is_pattern_an_A_pattern: bool):
        prev_A_vector = np.zeros((self.A_width * self.A_height))
        prev_B_vector = np.zeros((self.B_width * self.B_height))

        V = _bipolorize_pattern(pattern).flatten()

        weight_matrix = self.weight_matrix

        done = True

        while done:

            if is_pattern_an_A_pattern:
                weight_matrix = weight_matrix.T

            result = np.dot(weight_matrix, V)

            for 
