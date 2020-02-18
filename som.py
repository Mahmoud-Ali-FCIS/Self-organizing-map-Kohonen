from helper import *


def train(train_data, weight, learning_rate, epochs):
    """

    :param train_data:
    :param weight:
    :param learning_rate:
    :param epochs:
    :return weight:
    """
    for n in range(epochs):
        for tr_data in train_data:
            distance = euclidean_distance(tr_data, weight)
            index_min_dist = np.argmin(np.array(distance))
            weight[index_min_dist] = update_weight(tr_data, weight, index_min_dist, learning_rate)

    return weight


def test(test_data, final_weight):
    """

    :param test_data:
    :param final_weight:
    :return classes:
    """
    classes = []
    for tr_data in test_data:
        distance = euclidean_distance(tr_data, final_weight)
        index_min_dist = np.argmin(np.array(distance))
        classes.append(index_min_dist)
    return classes
