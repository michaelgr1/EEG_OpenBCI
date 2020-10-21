import numpy as np
import matplotlib.pyplot as plt
from classification import LogisticRegressionClassifier, KNearestNeighborsClassifier
import utils

if __name__ == "__main__":
    # x = np.array([[2, 1], [3, 1], [4, 1], [5, 1], [3, 2], [4, 2], [5, 2], [4, 3],
    #               [1, 4], [1, 5], [1, 6], [2, 5], [2, 6], [2, 7], [3, 6], [4, 7]])
    # y = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1]])

    # x = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [1.5, 1.5], [-1.5, 1.5], [-1.5, -1.5], [1.5, -1.5]])

    # x = np.concatenate((x, np.multiply(x, x)), axis=1)

    # y = np.array([[1], [1], [1], [1], [0], [0], [0], [0]])

    # classifier = LogisticRegressionClassifier(x, y)
    # cost_arr = classifier.train(iterations=400, learning_rate=0.1)
    # plt.plot(cost_arr)

    x_arr = []
    y_arr = []

    for i in range(100):
        x_arr.append([np.random.randint(-20, 20), np.random.randint(-40, 40)])
        if x_arr[-1][-1] > 0:
            y_arr.append([1])
        else:
            y_arr.append([0])

    x = np.array(x_arr, dtype=float)
    y = np.array(y_arr, dtype=float)

    x, features_min_max = utils.normalize_feature_matrix(x, -1, 1)

    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.scatter(x[i, 0], x[i, 1], color="r")
        else:
            plt.scatter(x[i, 0], x[i, 1], color="b")

    classifier = KNearestNeighborsClassifier(x, y)

    print("Accuracy = {}".format(classifier.test_set_accuracy(k=5)))

    x_p = np.array([[-1], [-1]], dtype=float)
    utils.normalize_sample(x_p, features_min_max, -1, 1)

    plt.scatter(x_p[0], x_p[1], color="k")

    label = classifier.classify(x_p, k=5)

    print("label = {}".format(label))

    k_values, accuracy = classifier.test_set_learning_curve()

    plt.plot(k_values, accuracy)

    k_values, accuracy = classifier.cross_validation_learning_curve()

    plt.plot(k_values, np.array(accuracy) + 5)

    # t = np.linspace(-10, 10, 100)

    # Plot decision boundary

    # plt.plot(t, - (classifier.weights[0] + classifier.weights[1] * t) / classifier.weights[2])

    # print("Accuracy = {}".format(classifier.compute_accuracy()))
    # print(classifier.predict(np.array([[1, 1, 7], [1, 7, 1]])))
    # print(classifier.classify(np.array([[1, 1, 7], [1, 7, 1]])))

    # plt.xlabel("Number Of Iterations")
    # plt.ylabel("Cost Function")
    plt.show()
