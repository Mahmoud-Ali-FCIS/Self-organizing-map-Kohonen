from som import *
import argparse

if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-ptr", "--path_train_data", type=str, default='train.txt',
                    help="path of train data .txt format")
    ap.add_argument("-pts", "--path_test_data", type=str, default='test.txt',
                    help="path of test data .txt format")
    ap.add_argument("-np", "--num_input", type=int, default=4,
                    help="number of input node int")
    ap.add_argument("-no", "--num_output", type=int, default=2,
                    help="number of output node int")
    ap.add_argument("-e", "--epochs", type=int, default=200,
                    help="number of iteration")
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.5,
                    help="learning rate flout")
    ap.add_argument("-s", "--spread", type=int, default=0,
                    help="spread int")
    args = vars(ap.parse_args())

    # Train phase
    train_data = read_data(args["path_train_data"])
    weight = create_inti_weight_matrix(args["num_input"], args["num_output"])
    final_weight = train(np.array(train_data), weight, args["learning_rate"], args["epochs"])

    # Test phase
    test_data = read_data(args["path_test_data"])
    classes_result = test(test_data, final_weight)
    print(classes_result)
