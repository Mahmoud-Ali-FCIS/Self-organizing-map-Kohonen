from random import random
import numpy as np


def read_data(path):
    """
    This function to read the data from file using the path based on any size fo this data
    and save it in list then convert to numpy array and return this array.

    :param path: [string] the full path of the file which want read the data from it
    :return data: [numpy array [flout]] Contains the data which used it to train or test the model
    """
    with open(path) as file:
        data = [[float(x) for x in line.split()] for line in file if line.strip()]
    return np.array(data)


def create_inti_weight_matrix(num_input, num_output):
    """
    This function to create the initial weight matrix randomly based on the number of input and number of output.

    :param num_input: [integer number] the number node in input layer
    :param num_output: [integer number] the number node in output layer
    :return weight_matrix: [numpy array [flout]] Contains the random initial weight to train the model
    """
    weight_matrix = np.empty((num_output, num_input))
    for o in range(num_output):
        for n in range(num_input):
            weight_matrix[o][n] = random()
    return weight_matrix


def euclidean_distance(input_value, weights):
    """
    This function to calculate the Euclidean Distances for the feature input with all the weights.

    :param input_value: [numpy array [flout]] Contains the data which want calculate the distance fot it
    :param weights: [numpy array [flout]] Contains weight to calculate the distance
    :return distance: [list of flout] Contains the distance for each class in the data
    """
    distance = []
    for weight in weights:
        distance.append(sum((input_value - weight)**2))

    return distance


def update_weight(input_value, weights, index_min_dist, learning_rate):
    """
    This function to update the weight based on the index of min distance for the feature.

    :param input_value: [numpy array [flout]] Contains the data which used to update the wight
    :param weights: [numpy array [flout]] Contains weight which want to update
    :param index_min_dist: [integer number] the index of the weight which want to update
    :param learning_rate: [flout number] used to update the wight
    :return new_weight: [numpy array [flout]] Contains the updated weight
    """
    new_weight = weights[index_min_dist] + learning_rate * (input_value - weights[index_min_dist])
    return np.array(new_weight)
