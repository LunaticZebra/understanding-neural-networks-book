import math
import random

import adaline
import weather_data
from adaline import check_data,train_adaline,adaline_predict


def standardize_data(data: list[float]) -> list[float]:

    mean = sum(data) / len(data)
    deviations = [(x-mean)**2 for x in data]
    deviations_sum = sum(deviations)
    std = math.sqrt(deviations_sum/len(data))
    standardized_data = []
    for value in data:
        standardized_data.append((value - mean)/std)

    return standardized_data


def normalize_data(data: list[float]) -> list[float]:
    min_value = min(data)
    max_value = max(data)
    normalized_data = []

    for value in data:
        normalized_data.append((value - min_value)/(max_value - min_value))

    return normalized_data


def extract_pattern_and_label(data):
    data_to_return = {}

    data_to_return["temperature"] = standardize_data(data["temperature"])
    data_to_return["mean_pressure"] = standardize_data(data["mean_pressure"])

    patterns = [[pressure, temp] for pressure, temp in zip(data_to_return["mean_pressure"], data_to_return["temperature"])]

    labels = [1 if val == True else -1 for val in data["was_raining"]]
    return patterns, labels

def save_to_csv(learning_rates, acc, err, weights_len):
    with open("stats.csv", "w") as f:
        f.write("Learning rates, accuracy, error, weights_len\n")
        for i in range(len(learning_rates)):
            f.write(f"{learning_rates[i]},{acc[i]},{err[i]},{weights_len[i]}\n"

    )


def test_adaline(test_patterns,test_labels,weights):
    correct_sum = 0
    counter = 0
    total_loss = 0

    for test_pattern,test_label in zip(test_patterns,test_labels):

        predicted_val = adaline_predict(test_pattern,weights)
        net_input = adaline.compute_received_input(test_pattern,weights)
        error = test_label - adaline.activation_function(net_input)
        total_loss += error ** 2

        if predicted_val == test_label:
            correct_sum += 1

        counter += 1

    return correct_sum,total_loss,counter


def test_adaline_parameters(start_val: float, end_val: float, step: float, patterns, labels, test_patterns, test_labels):

    weights = [0.0,0.0]

    learning_rate = start_val

    learning_rates = []
    acc = []
    err = []
    weights_len = []

    while learning_rate < end_val:
        learning_rate += step
        learning_rates.append(round(learning_rate, 3))

        train_adaline(patterns,labels,weights, learning_rate, 0.03, 20)


        correct_sum, total_loss, counter = test_adaline(test_patterns,test_labels,weights)

        err.append(total_loss / counter)
        acc.append(correct_sum / counter)
        weights_len.append(math.sqrt(weights[0]**2 + weights[1]**2))
        weights = [0.0, 0.0]

    save_to_csv(learning_rates, acc, err, weights_len)



if __name__ == "__main__":

    data = weather_data.retrieve_weather_data(52.196835, 21.08856, "2023-01-01", "2023-06-30")
    test_data = weather_data.retrieve_weather_data(52.196835, 21.08856, "2024-01-31", "2024-06-30")

    patterns, labels = extract_pattern_and_label(data)

    test_patterns, test_labels = extract_pattern_and_label(test_data)

    weights = train_adaline(patterns, labels, [0.0,0.0], 0.04, 0.03, 20)

    correct_preds, loss, counter = test_adaline(test_patterns, test_labels, weights)

    print(f"Average accuracy: {correct_preds / counter}, mean squared loss: {loss / counter}")

