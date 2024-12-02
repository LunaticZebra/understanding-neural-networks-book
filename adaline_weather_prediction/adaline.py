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

def calculate_square_magnitude(pattern: list[float]) -> float:
    square_magnitude = 0
    for val in pattern:
        square_magnitude += val * val

    return square_magnitude

def train_adaline(input_patterns: list[list[float]],labels: list[int],weights: list[float], learning_rate: float):

    if not check_data(input_patterns,labels):
        return None

    for i in range(len(input_patterns)):
        label = labels[i]
        pattern = input_patterns[i]
        net_input = compute_received_input(pattern, weights)
        output_value = calculate_output(net_input)
        error_val = label - output_value

        if error_val != 0:

            while error_val != 0:
                square_magnitude = calculate_square_magnitude(pattern)
                delta_rule_vector = []
                for element in pattern:

                    weight_change = learning_rate * error_val * element / square_magnitude
                    delta_rule_vector.append(weight_change)

                for j in range(len(delta_rule_vector)):
                    weights[j] = weights[j] + delta_rule_vector[j]

                output_value = calculate_output(compute_received_input(pattern,weights))
                error_val = label - output_value

            for j in range(0, i):
                label = labels[j]
                pattern = input_patterns[j]
                net_input = compute_received_input(pattern,weights)
                output_value = calculate_output(net_input)
                error_val = label - output_value

                while error_val != 0:
                    square_magnitude = calculate_square_magnitude(pattern)
                    delta_rule_vector = []
                    for element in pattern:
                        weight_change = learning_rate * error_val * element / square_magnitude
                        delta_rule_vector.append(weight_change)

                    for z in range(len(delta_rule_vector)):
                        weights[z] = weights[z] + delta_rule_vector[z]

                    output_value = calculate_output(compute_received_input(pattern,weights))
                    error_val = label - output_value



    return weights

def adaline_predict(pattern: list[float], weights: list[float]) -> int:
    output_value = calculate_output(compute_received_input(pattern, weights))
    return output_value