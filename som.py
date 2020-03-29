from helper import *


def train(train_data, weight, learning_rate, epochs):
    """
    This function for train the model for number of epochs using the input data and the weight.

    :param train_data: [numpy array [flout]] Contains the data which used it to train the model
    :param weight: [numpy array [flout]] Contains the random initial weight to train the model
    :param learning_rate: [flout number] used to update the wight
    :param epochs: [int number] the number of iteration for train the model
    :return weight: [numpy array [flout]] Contains the train weight after finish the the training
    """
    for n in range(epochs):
        for tr_data in train_data:
            distance = euclidean_distance(tr_data, weight)
            index_min_dist = np.argmin(np.array(distance))
            weight[index_min_dist] = update_weight(tr_data, weight, index_min_dist, learning_rate)

    return weight


def test(test_data, final_weight, name_c1, name_c2):
    """
    This function for test the model for test data using get the min distance for features based on final weight.

    :param test_data: [numpy array [flout]] Contains the data which need test it
    :param final_weight:[numpy array [flout]] Contains the train weight after finish the the training
    :param name_c1:The name of class 1
    :param name_c2:The name of class 2
    :return classes: [list of output name of classes] Contains the result of prediction after test the data
    """
    classes = []
    for tr_data in test_data:
        distance = euclidean_distance(tr_data, final_weight)
        index_min_dist = np.argmin(np.array(distance))
        classes.append(name_c1 if index_min_dist == 0 else name_c2)
    return classes
