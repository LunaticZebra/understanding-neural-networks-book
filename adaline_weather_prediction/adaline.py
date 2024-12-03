import math
from itertools import filterfalse



def check_data(input_patterns: list[list[float]], labels: list[int]):
    prev_len = len(input_patterns[0])
    for pattern in input_patterns:
        if len(pattern) != prev_len:
            print("The input patterns should all be of the same length")
            return False

    for label in labels:
        if label != 1 and label != -1:
            print("The labels should be either 1 or -1")
            return False

    return True


def compute_received_input(input_pattern: list[float], weights: list[float]) -> float:
    dot_product = 0
    for val1, val2 in zip(input_pattern, weights):
        dot_product += val1 * val2

    return dot_product

def calculate_output(input_value: float) -> int:
    if input_value <= 0:
        return -1
    else:
        return 1


def activation_function(net_input: float) -> float:
    net_input = max(-700,min(700, net_input))
    return 2 * 1/(1+ math.exp(-net_input)) - 1


def calculate_square_magnitude(pattern: list[float]) -> float:
    square_magnitude = 0
    for val in pattern:
        square_magnitude += val * val

    return square_magnitude

def train_adaline(input_patterns: list[list[float]],labels: list[int],weights: list[float], learning_rate: float, error_margin: float, max_epochs: int):

    if not check_data(input_patterns,labels):
        return None

    for i in range(len(input_patterns)):
        label = labels[i]
        pattern = input_patterns[i]
        net_input = compute_received_input(pattern, weights)
        activation_value = activation_function(net_input)
        error_val = label - activation_value
        error_margin *= 2 # because the absolute error is in range between 0 and 2
        square_magnitude = calculate_square_magnitude(pattern)

        if abs(error_val) >= error_margin:

            # adjust weight for current pattern
            adjust_weights(error_val, error_margin, pattern, learning_rate, weights, square_magnitude, label, max_epochs)

            # adjust weight for previous patterns
            for j in range(0, i):
                label = labels[j]
                pattern = input_patterns[j]
                net_input = compute_received_input(pattern,weights)

                activation_value = activation_function(net_input)
                error_val = label - activation_value

                adjust_weights(error_val, error_margin, pattern, learning_rate, weights, square_magnitude, label, max_epochs)

            for j in range(len(weights)):
                weights[j] -= 0.001 * weights[j]

    return weights

def adjust_weights(error_val: float, error_margin: float, pattern: list[float], learning_rate: float, weights: list[float],square_magnitude: float, label: int, max_epochs: int):

    epochs = 1

    while abs(error_val) >= error_margin:

        delta_rule_vector = []
        for element in pattern:
            weight_change = learning_rate * error_val * element / square_magnitude
            delta_rule_vector.append(weight_change)

        for j in range(len(delta_rule_vector)):
            weights[j] = weights[j] + delta_rule_vector[j]

        net_input = compute_received_input(pattern, weights)
        activation_value = activation_function(net_input)
        error_val = label - activation_value

        if epochs == max_epochs:
            break

        epochs += 1

def adaline_predict(pattern: list[float], weights: list[float]) -> int:
    output_value = calculate_output(compute_received_input(pattern, weights))
    return output_value