import numpy as np
import classification

import matplotlib.pyplot as plt

if __name__ == "__main__":

    sample_size = 10

    a = np.random.rand(sample_size, 2)
    a = a + 2

    b = np.random.rand(sample_size, 2)
    b = b - 2

    c = np.random.rand(sample_size, 2)
    c[:, 0] = c[:, 0] - 2
    c[:, 1] = c[:, 1] + 2

    d = np.random.rand(sample_size, 2)
    d[:, 0] = d[:, 0] + 2
    d[:, 1] = d[:, 1] - 2

    data = np.concatenate((a, b, c, d), axis=0)

    labels = np.concatenate((np.zeros((sample_size, 1)), np.ones((sample_size, 1)),
                       2 * np.ones((sample_size, 1)), 3*np.ones((sample_size, 1))), axis=0)

    print(labels)

    # plt.scatter(a[:, 0], a[:, 1])
    # plt.scatter(b[:, 0], b[:, 1])
    # plt.scatter(c[:, 0], c[:, 1])
    # plt.scatter(d[:, 0], d[:, 1])

    # classifier = classification.LogisticRegressionClassifier(data, labels)

    # cost = classifier.train()

    # print(classifier.test_set_accuracy())

    # print(classifier.classify(np.array([2, 2])))
    # print(classifier.classify(np.array([-2, -2])))
    # print(classifier.classify(np.array([-2, 2])))
    # print(classifier.classify(np.array([2, -2])))
    #
    # knn = classification.KNearestNeighborsClassifier(data, labels)
    # print(f"KNN Accuracy = {knn.test_set_accuracy(2)}")
    # print(f"Logistic Regression Accuracy = {classifier.test_set_accuracy()}")
    #
    # print(knn.classify(np.array([[2], [2]])))
    # print(knn.classify(np.array([[-2], [-2]])))
    # print(knn.classify(np.array([[-2], [2]])))
    # print(knn.classify(np.array([[2], [-2]])))
    #
    # # print("Accuracy = {}".format(classifier.test_set_accuracy())

    # plt.plot(cost[0])
    # plt.plot(cost[1])
    # plt.plot(cost[2])
    # plt.plot(cost[3])

    perceptron = classification.PerceptronClassifier(data, labels)
    perceptron.train()

    print(perceptron.test_set_accuracy())

    print(perceptron.classify(np.array([
        [2, 2],
        [-2, -2],
        [-2, 2],
        [2, -2]
    ])))

    print()

    plt.show()
