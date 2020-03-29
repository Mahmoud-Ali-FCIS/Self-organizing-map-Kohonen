from som import *
import argparse

if __name__ == '__main__':

    # Construct the argument parse and parse the arguments
    # We can get all the parameters of this program using terminal
    ap = argparse.ArgumentParser()
    ap.add_argument("-ptr1", "--path_train_data_class_1", type=str,
                    default='patient.txt',
                    help="path of train data class 1 .txt format")
    ap.add_argument("-ptr2", "--path_train_data_class_2", type=str,
                    default='control.txt',
                    help="path of train data class 2 .txt format")
    ap.add_argument("-pts", "--path_test_data", type=str,
                    default='ali.txt',
                    help="path of test data .txt format")
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
    # In this step start train the model using input data
    name_c1 = args["path_train_data_class_1"].split('.')[0]
    name_c2 = args["path_train_data_class_2"].split('.')[0]
    class1 = read_data('Data/' + args["path_train_data_class_1"])  # 10X650
    class2 = read_data('Data/' + args["path_train_data_class_2"])  # 10X650
    train_data = np.concatenate((class1, class2), axis=0)
    weight = create_inti_weight_matrix(train_data.shape[1], args["num_output"])
    final_weight = train(np.array(train_data), weight, args["learning_rate"], args["epochs"])

    # Test phase
    # In this step start test the model using test data
    test_data = read_data('Data/' + args["path_test_data"])  # 4X650
    classes_result = test(test_data, final_weight, name_c1, name_c2)
    print(classes_result)
