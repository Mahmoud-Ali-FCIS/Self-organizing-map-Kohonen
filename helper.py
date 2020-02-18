from random import random
import numpy as np


def read_data(path):
    """

    :param path:
    :return data:
    """
    with open(path) as file:
        data = [[float(x) for x in line.split()] for line in file]
    return np.array(data)


def create_inti_weight_matrix(num_input, num_output):
    """

    :param num_input:
    :param num_output:
    :return weight_matrix:
    """
    weight_matrix = np.empty((num_output, num_input))
    for o in range(num_output):
        for n in range(num_input):
            weight_matrix[o][n] = random()
    return weight_matrix


def euclidean_distance(input_value, weights):
    """

    :param input_value:
    :param weights:
    :return distance:
    """
    distance = []
    for weight in weights:
        distance.append(sum((input_value - weight)**2))

    return distance


def update_weight(input_value, weights, index_min_dist, learning_rate):
    """

    :param input_value:
    :param weights:
    :param index_min_dist:
    :param learning_rate:
    :return new_weight:
    """
    new_weight = weights[index_min_dist] + learning_rate * (input_value - weights[index_min_dist])
    return np.array(new_weight)
