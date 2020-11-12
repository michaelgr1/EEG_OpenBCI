import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import statistics
import utils
from data import DataSubSetType, DataSet
from data import is_vector


def one_versus_rest_labels(labels: np.ndarray, label: float) -> np.ndarray:
    """
        Alter labels, make it have only two classes, the given label and all the rest
        (1 when given labels is zero, 0 otherwise)
    """
    altered_labels = labels.copy()
    if label != 0:
        altered_labels[labels != label] = 0
    else:
        altered_labels[labels != label] = 1

    return altered_labels


def classification_accuracy(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> float:
    positives = np.ones_like(actual_labels)[predicted_labels == actual_labels]
    return np.sum(positives) / predicted_labels.shape[0]


def sigmoid(z: np.ndarray):
    return 1 / (1 + np.exp(-z))


def shifted_sigmoid(z: np.ndarray, shift: float):
    return 1 / (1 + np.exp(-(z - shift)))


class LogisticRegressionClassifier:
    NAME = "logistic_regression"

    def __init__(self, data_matrix: np.ndarray, labels: np.ndarray,
                 positive_label: float = 1, negative_label: float = 0):
        self.sub_classifiers = {}
        self.unique_labels = np.unique(labels)

        self.data_set = DataSet(data_matrix, labels, add_x0=True, shuffle=True)

        if self.unique_labels.shape[0] > 2:  # OVR
            self.multi_class = True

            for label in self.unique_labels:

                if label == 0:
                    negative_label = 1
                else:
                    negative_label = 0

                self.sub_classifiers[label] = LogisticRegressionClassifier(
                    data_matrix, one_versus_rest_labels(labels, label), label, negative_label
                )
        else:
            self.multi_class = False
            self.weights = np.ones((self.data_set.feature_count() + 1, 1))

            self.label_translator = LabelTranslator()

            if positive_label not in self.unique_labels or negative_label not in self.unique_labels:
                negative_label = self.unique_labels[0]
                positive_label = self.unique_labels[1]

            self.label_translator.append_pair(negative_label, 0)
            self.label_translator.append_pair(positive_label, 1)

    def predict(self, x: np.ndarray, label: float = -1, raw: bool = True):
        if self.multi_class:
            return self.sub_classifiers[label].predict(x, raw=raw)
        else:
            if raw:
                x = self.data_set.process_data(x)
            return sigmoid(np.dot(x, self.weights))

    def confidence(self, x: np.ndarray, label: float = -1, raw: bool = True):
        if self.multi_class:
            return self.sub_classifiers[label].predict(x)
        else:
            return self.predict(x, raw=raw)

    def train(self, iterations=1000, learning_rate=1, accuracy_threshold: float = 1.0):
        if self.multi_class:
            total_cost = []
            for classifier in self.sub_classifiers.values():
                cost = classifier.train(iterations, learning_rate, accuracy_threshold)
                total_cost.append(cost)
            return total_cost
        else:
            m = self.data_set.sample_count(DataSubSetType.TRAINING)
            cost = []

            # Gradient descent
            for i in range(iterations):
                cost.append(
                    self.compute_cost(self.data_set.get_training_set(), self.data_set.training_set_labels, raw=False)[0, 0])

                print("Training, iteration #{}".format(i))

                translated_labels = self.label_translator.translate_all(self.data_set.training_set_labels)

                self.weights = self.weights - learning_rate / m *\
                               np.dot(np.transpose(self.data_set.get_training_set()),
                                   (self.predict(self.data_set.get_training_set(), raw=False) - translated_labels))

                accuracy = classification_accuracy(
                    self.classify(self.data_set.get_training_set(), raw=False), self.data_set.training_set_labels)

                if accuracy >= accuracy_threshold:
                    print("Reached Accuracy threshold...")
                    break

            return cost

    def compute_cost(self, data: np.ndarray, labels: np.ndarray, raw: bool = True):
        if self.multi_class:
            raise NotImplemented()
        else:
            x = data
            labels = np.transpose(self.label_translator.translate_all(labels))
            predictions = self.predict(x, raw=raw)
            m = data.shape[0]
            return 1 / m * (np.dot(-labels, np.log(predictions)) - np.dot(1 - labels, np.log(1 - predictions)))

    def classify(self, unseen_data: np.ndarray, raw: bool = True):
        if self.multi_class:
            predictions = {}
            for label in self.unique_labels:
                predictions[label] = self.predict(unseen_data, label, raw)

            if unseen_data.ndim == 1:
                unseen_data = unseen_data.reshape((1, unseen_data.shape[0]))
            predictions_list = list(predictions.values())

            # A matrix with samples as rows and labels as columns
            predictions_matrix = np.zeros((unseen_data.shape[0], self.unique_labels.shape[0]))

            for i in range(len(predictions_list)):
                predictions_matrix[:, i] = predictions_list[i].flatten()

            max_predictions = np.max(predictions_matrix, axis=1, keepdims=True)

            labels = np.zeros((unseen_data.shape[0], 1))

            for row in range(max_predictions.shape[0]):
                for label in self.unique_labels:
                    if predictions[label][row] == max_predictions[row]:
                        labels[row] = label
                        break

            return labels
        else:
            prediction = self.predict(unseen_data, raw=raw)
            labels = np.empty_like(prediction)
            labels[prediction >= 0.5] = self.label_translator.original(1)
            labels[prediction < 0.5] = self.label_translator.original(0)
            return labels

    def test_set_accuracy(self) -> float:
        return classification_accuracy(self.classify(self.data_set.get_test_set(), raw=False), self.data_set.test_set_labels)

    def cross_validation_accuracy(self) -> float:
        return classification_accuracy(
            self.classify(self.data_set.get_cross_validation_set(), raw=False), self.data_set.cross_validation_labels)

    def training_set_accuracy(self) -> float:
        return classification_accuracy(
            self.classify(self.data_set.get_training_set(), raw=False), self.data_set.training_set_labels
        )


class KNearestNeighborsClassifier:
    NAME = "k_nearest_neighbors"

    def __init__(self, data_matrix: np.ndarray, labels: np.ndarray, k: int):
        self.unique_labels = np.unique(labels)

        self.default_k_value = k

        self.data_set = DataSet(data_matrix, labels, add_x0=False, shuffle=True)

    def distances_list_from(self, x: np.ndarray, raw: bool = True):
        """
            Computes a list of tuples with distances as a first value
            and class label as a second value. Each tuple represents the distance from one of
            the samples in the training set and its corresponding label.
        :param x: The sample from which the distance is calculated, should be a column vector.
        :param raw:
        :return: An ordered (by distance) list of the tuples mentioned above.
        """

        if raw:
            x = self.data_set.process_data(x)

        m = self.data_set.sample_count(DataSubSetType.TRAINING)

        # Contains tuples with distance and index
        distances = []

        # Loop over training set and compute the distance from the given x to each sample
        for row in range(m):
            training_set_sample = self.data_set.sample_at(row, DataSubSetType.TRAINING).reshape((-1, 1))
            distance_between = utils.distance_between(x, training_set_sample)
            distances.append((distance_between, row))

        distances.sort()

        return distances

    def classify(self, x: np.ndarray, k: int = -1, raw: bool = True):

        if not is_vector(x):
            return self.classify_data_matrix(x, k, raw)

        if raw:
            x = self.data_set.process_data(x)

        m = self.data_set.sample_count(DataSubSetType.TRAINING)

        if k == -1:
            k = self.default_k_value

        if k > m:
            k = m

        distances = self.distances_list_from(x, False)

        # A dictionary containing labels as keys and counters as values specifying how many of the k closest
        # training samples are of the given class.
        class_voting = {}

        # Loop over the k closest training set samples
        for i in range(0, k):
            # Obtain the row index in the training set
            row = distances[i][1]

            training_set_label = self.data_set.label_at(row, DataSubSetType.TRAINING)

            if training_set_label in class_voting.keys():
                class_voting[training_set_label] += 1
            else:
                class_voting[training_set_label] = 1

        max_voting = sorted(class_voting.values())[-1]

        for class_label in class_voting.keys():
            if class_voting[class_label] == max_voting:
                return class_label

        return -1

    def classify_data_matrix(self, data_matrix: np.ndarray, k: int = -1, raw: bool = True):
        """
        Classify a data matrix with rows as samples and columns as features.
        :param data_matrix:
        :param k:
        :param raw:
        :return: A column vector of the classified labels.
        """

        labels = np.zeros((0, 1))

        for row in range(data_matrix.shape[0]):
            x = data_matrix[row, :].reshape((-1, 1))  # Retrieve the current sample and reshape it to a column vector
            label = self.classify(x, k, raw)

            labels = np.append(labels, np.array([[label]]), axis=0)

        return labels

    def mean_distance_from_k_closest_samples(self, x: np.ndarray, k: int = -1, raw: bool = True):
        """
            Computes the mean distance between the given sample/samples and its k-closest
            neighbors in the training set.
        :param x:
        :param k:
        :return:
        """

        if k == -1:
            k = self.default_k_value

        if x.shape[1] > 1:
            mean_distances = np.zeros((x.shape[0], 1))

            for i in range(x.shape[0]):
                sample = x[i, :].reshape((-1, 1))
                mean_distances[i, 0] = self.mean_distance_from_k_closest_samples(sample)

            return mean_distances

        distances = self.distances_list_from(x, raw=raw)

        average_distance = utils.AccumulatingAverage()

        for i in range(0, k):  # Loop over the K closest ones.
            average_distance.add_value(distances[i][0])

        return average_distance.compute_average()

    def test_set_accuracy(self, k: int = -1) -> float:
        if k == -1:
            k = self.default_k_value
        return classification_accuracy(self.classify(self.data_set.get_test_set(), k, raw=False),
                                       self.data_set.test_set_labels)

    def cross_validation_accuracy(self, k: int = -1) -> float:
        if k == -1:
            k = self.default_k_value
        return classification_accuracy(self.classify(self.data_set.get_cross_validation_set(), k, raw=False),
                                       self.data_set.cross_validation_labels)

    def accuracy_learning_curve(self, data: np.ndarray, labels: np.ndarray, raw: bool = True) -> ([], []):

        k_values = []
        accuracy_values = []

        for k in range(1, data.shape[0]):
            k_values.append(k)
            accuracy_values.append(classification_accuracy(self.classify(data, k, raw=raw), labels))

        return k_values, accuracy_values

    def test_set_learning_curve(self) -> ([], []):
        return self.accuracy_learning_curve(self.data_set.get_test_set(), self.data_set.test_set_labels, raw=False)

    def cross_validation_learning_curve(self) -> ([], []):
        return self.accuracy_learning_curve(
            self.data_set.get_cross_validation_set(), self.data_set.cross_validation_labels, raw=False)

    def training_set_accuracy(self, k=-1):
        if k == -1:
            k = self.default_k_value
        return classification_accuracy(self.classify(self.data_set.get_training_set(), k, raw=False),
                                       self.data_set.training_set_labels)


class NeuralNetworkStructure:

    def __init__(self, input_neurons_count: int, output_neurons_count: int):
        self.input_neurons_count = input_neurons_count
        self.output_neurons_count = output_neurons_count

        self.hidden_layers_sizes = []

    def add_hidden_layer(self, neuron_count: int):
        self.hidden_layers_sizes.append(neuron_count)

    def hidden_layers_count(self):
        return len(self.hidden_layers_sizes)

    def weights_matrix_size(self, layer_index: int) -> (int, int):
        """"
            Computes the size of the weights matrix which multiplies the values of the given layer.
            The input layer has the index of 1
        """

        if layer_index < 1:
            return None

        if layer_index == 1:
            return self.hidden_layers_sizes[0], self.input_neurons_count + 1
        elif layer_index < len(self.hidden_layers_sizes):  # One of the hidden layers which is not the last one
            return self.hidden_layers_sizes[layer_index - 2 + 1], self.hidden_layers_sizes[layer_index - 2] + 1
        elif layer_index == len(self.hidden_layers_sizes) + 1:  # Last hidden layer
            return self.output_neurons_count, self.hidden_layers_sizes[layer_index - 2] + 1
        else:
            return None


class PerceptronClassifier:
    NAME = "perceptron"

    def __init__(self, data_matrix: np.ndarray, labels: np.ndarray, positive_label: float = +1, negative_label: float = -1):
        """
        Initializes a new perceptron classifier. The classifier can be either binary or not and use OVR strategy.
        The data set is a matrix with data as rows and features as columns.
        The labels is a column vector with the same number of rows as the data set matrix.
        The positive label and negative label arguments specify which label in the data set should be treated as the positive
        label, i.e., the dot product with it should be positive, and which label should be treated as negative.
        When used as a binary classifier those arguments are not necessary and will not be used.
        Their primary use is when a multi class perceptron initiates its sub classifier where a given label should be positive
        and all the rest should be negative.
        :param data_matrix: A matrix containing the data set. This data set will be divided into smaller ones for testing and training.
        :param labels: The labels for the given data set.
        :param positive_label: Explained above.
        :param negative_label: Explained above.
        """

        self.unique_labels = np.unique(labels)

        self.label_translator = LabelTranslator()

        self.multi_class = False

        self.sub_classifiers = {}

        self.data_set = DataSet(data_matrix, labels, add_x0=True, shuffle=True)

        # Feature count
        self.n = self.data_set.feature_count()

        if self.unique_labels.shape[0] > 2:
            self.multi_class = True
            print("Multi-Class perceptron...")

            for label in self.unique_labels:

                if label == 0:
                    negative_label = 1
                else:
                    negative_label = 0

                self.sub_classifiers[label] = PerceptronClassifier(
                    data_matrix, one_versus_rest_labels(labels, label), label, negative_label
                )

        else:
            if negative_label not in self.unique_labels or positive_label not in self.unique_labels:
                print("Positive or negative label doesn't match labels")
                negative_label = self.unique_labels[0]
                positive_label = self.unique_labels[1]

            self.label_translator.append_pair(negative_label, -1)
            self.label_translator.append_pair(positive_label, +1)

            self.weights = np.zeros((self.n + 1, 1))

    def train(self, iterations: int = 10000, accuracy_threshold: float = 1.0):
        if self.multi_class:

            for sub_classifier in self.sub_classifiers.values():
                sub_classifier.train(iterations, accuracy_threshold)

            print("Multi-Class training over...")
        else:
            for _i in range(iterations):
                miss_classification_count = 0

                for row in range(self.data_set.sample_count(DataSubSetType.TRAINING)):
                    x = self.data_set.sample_at(row, DataSubSetType.TRAINING).reshape((1, -1))
                    label = self.data_set.label_at(row, DataSubSetType.TRAINING)

                    label = self.label_translator.translate(label)

                    if label * np.dot(x, self.weights)[0, 0] <= 0:  # Wrong classification
                        miss_classification_count += 1
                        self.weights = self.weights + label * np.transpose(x)

                accuracy = classification_accuracy(
                    self.classify(self.data_set.get_training_set(), raw=False), self.data_set.training_set_labels)

                if miss_classification_count == 0:
                    print("Perfect classification...")
                    break
                elif accuracy >= accuracy_threshold:
                    print("Accuracy threshold reached...")
                    break
                else:
                    print("More training is required, error count = {}".format(miss_classification_count))

            print("Training over")

    def classify(self, x: np.ndarray, raw: bool = True) -> np.ndarray:
        if self.multi_class:
            prediction_matrix = self.predict(x, raw=raw)

            max_predictions = np.max(prediction_matrix, axis=1, keepdims=True)

            labels = np.zeros_like(max_predictions)

            for row in range(prediction_matrix.shape[0]):
                for column in range(prediction_matrix.shape[1]):
                    if max_predictions[row, 0] == prediction_matrix[row, column]:
                        labels[row, 0] = list(self.sub_classifiers.keys())[column]
                        break

            return labels
        else:
            predictions = self.predict(x, raw=raw)

            labels = np.zeros_like(predictions)

            labels[predictions > 0] = self.label_translator.original(+1)
            labels[predictions <= 0] = self.label_translator.original(-1)

            return labels

    def predict(self, unseen_data: np.ndarray, raw: bool = True) -> np.ndarray:
        """
            Predicts the values for the given unseen values. Does not round to +1 or -1.
        :param unseen_data: The data array to be classified
        :param raw:
        :return: The matrix vector product of the unseen data with the current weights vector
        """
        if self.multi_class:

            # Compute the prediction matrix. A matrix where each row corresponds to a sample and each column to a label.
            prediction_matrix = np.zeros((unseen_data.shape[0], int(self.unique_labels.shape[0])))

            for i in range(len(self.sub_classifiers.keys())):
                label = list(self.sub_classifiers.keys())[i]
                sub_classifier = self.sub_classifiers[label]

                prediction_matrix[:, i] = sub_classifier.predict(unseen_data, raw=raw).flatten()
            return prediction_matrix
        else:
            if raw:
                unseen_data = self.data_set.process_data(unseen_data)
            return np.dot(unseen_data, self.weights)

    def confidence(self, x: np.ndarray, raw: bool = True):
        # predictions = self.predict(self.training_set)
        # predicted_labels = self.classify(self.training_set)
        # actual_labels = self.training_set_labels
        #
        # x = np.linspace(0, predictions.shape[0] - 1, predictions.shape[0]).reshape(predictions.shape)
        #
        # # Plot correctly classified items
        # plt.scatter(x[predicted_labels == actual_labels], predictions[predicted_labels == actual_labels], color="green")
        #
        # # Plot wrong classified items
        # plt.scatter(x[predicted_labels != actual_labels], predictions[predicted_labels != actual_labels], color="red")
        #
        # mean = utils.arithmetic_mean(predictions)
        # std = utils.sample_standard_deviation(predictions)
        #
        # miss_classified_mean = utils.arithmetic_mean(predictions[predicted_labels != actual_labels])
        # miss_classified_std = utils.sample_standard_deviation(predictions[predicted_labels != actual_labels])
        #
        # plt.plot(x, np.ones_like(predictions) * mean)
        # plt.plot(x, np.ones_like(predictions) * (mean + std))
        # plt.plot(x, np.ones_like(predictions) * (mean - std))
        #
        # plt.plot(x, np.ones_like(predictions) * miss_classified_mean)
        # plt.plot(x, np.ones_like(predictions) * (miss_classified_mean + miss_classified_std))
        # plt.plot(x, np.ones_like(predictions) * (miss_classified_mean - miss_classified_std))
        #
        # plt.legend(["Training Average", "Training Average + std", "Training Average - std",
        #            "Miss Classified Average", "Miss Classified Average + std", "Miss Classified Average - std",
        #             "Correct Samples", "Wrong Samples"])
        #
        # plt.show()

        training_set_predictions = self.predict(self.data_set.get_training_set())
        training_set_predicted_labels = self.classify(self.data_set.get_training_set())
        training_set_actual_labels = self.data_set.training_set_labels

        miss_classified_predictions = \
            training_set_predictions[training_set_predicted_labels != training_set_actual_labels]

        mean = statistics.arithmetic_mean(miss_classified_predictions)
        std = statistics.sample_standard_deviation(miss_classified_predictions)

        prediction = self.predict(x, raw)

        return shifted_sigmoid(np.abs(prediction - mean), std)

    def test_set_accuracy(self) -> float:
        return classification_accuracy(self.classify(self.data_set.get_test_set(), False), self.data_set.test_set_labels)

    def cross_validation_accuracy(self):
        return classification_accuracy(
            self.classify(self.data_set.get_cross_validation_set(), False), self.data_set.cross_validation_labels)

    def training_set_accuracy(self):
        return classification_accuracy(
            self.classify(self.data_set.get_training_set(), False), self.data_set.training_set_labels
        )


class SvmClassifier:
    NAME = "svm"

    def __init__(self, data_matrix: np.ndarray, labels: np.ndarray, regularization: float = 1):
        self.data_set = DataSet(data_matrix, labels, add_x0=False, shuffle=True)
        self.classifier = svm.LinearSVC(C=regularization, max_iter=100000)

    def train(self):
        self.classifier.fit(self.data_set.get_training_set(), self.data_set.training_set_labels.flatten())

    def classify(self, x: np.ndarray):
        return self.classifier.predict(x)

    def training_set_accuracy(self):
        return self.classifier.score(self.data_set.get_training_set(), self.data_set.training_set_labels.flatten())

    def test_set_accuracy(self) -> float:
        return self.classifier.score(self.data_set.get_test_set(), self.data_set.test_set_labels.flatten())

    def cross_validation_accuracy(self) -> float:
        return self.classifier.score(self.data_set.get_cross_validation_set(), self.data_set.cross_validation_labels.flatten())


class LdaClassifier:
    NAME = "lda"

    def __init__(self, data_matrix: np.ndarray, labels: np.ndarray):
        self.data_set = DataSet(data_matrix, labels, add_x0=False, shuffle=True)
        self.classifier = LDA()

    def train(self):
        self.classifier.fit(self.data_set.get_training_set(), self.data_set.training_set_labels.flatten())

    def classify(self, x: np.ndarray):
        return self.classifier.predict(x)

    def training_set_accuracy(self):
        return self.classifier.score(self.data_set.get_training_set(), self.data_set.training_set_labels.flatten())

    def test_set_accuracy(self) -> float:
        return self.classifier.score(self.data_set.get_test_set(), self.data_set.test_set_labels.flatten())

    def cross_validation_accuracy(self) -> float:
        return self.classifier.score(self.data_set.get_cross_validation_set(), self.data_set.cross_validation_labels.flatten())


class LabelTranslator:

    def __init__(self):
        self.original_labels = []
        self.translated_labels = []

    def append_pair(self, original: float, translated: float):
        self.original_labels.append(original)
        self.translated_labels.append(translated)

    def translate(self, label: float):
        for i in range(len(self.original_labels)):
            if self.original_labels[i] == label:
                return self.translated_labels[i]
        return label

    def original(self, translated_label: float):
        for i in range(len(self.translated_labels)):
            if self.translated_labels[i] == translated_label:
                return self.original_labels[i]

    def translate_all(self, labels: np.ndarray):
        flat_labels = labels.flatten()
        translated = np.zeros_like(flat_labels)

        for i in range(flat_labels.shape[0]):
            translated[i] = self.translate(flat_labels[i])

        return translated.reshape(labels.shape)

    def original_all(self, translated_labels: np.ndarray):
        flat_translated = translated_labels.flatten()
        original = np.zeros_like(flat_translated)

        for i in range(flat_translated.shape[0]):
            original[i] = self.original(flat_translated[i])

        return original.reshape(translated_labels)


class NeuralNetworkClassifier:
    NAME = "neural_network"

    def __init__(self, data_matrix: np.ndarray, labels: np.ndarray, structure: NeuralNetworkStructure):

        if data_matrix.shape[1] != structure.input_neurons_count:
            raise ValueError

        self.structure = structure

        self.weights_matrices = []

        for layer_index in range(structure.hidden_layers_count() + 1):
            # TODO: Consider random initialization
            # self.weights_matrices.append(np.ones(structure.weights_matrix_size(layer_index + 1)))
            self.weights_matrices.append(np.random.random(structure.weights_matrix_size(layer_index + 1)))

        print("Weights initialized")

        self.data_set = DataSet(data_matrix, labels, add_x0=False, shuffle=True)

    def neuron_activations(self, data_set: np.ndarray, layer_index: int, sample_index: int):
        """"
            Computes the neuron activations of the given layer using the specified sample.
            Return the as a column vector
        """

        if layer_index == 1:
            x = data_set[sample_index, :]
            neuron_count = x.shape[0]
            x = x.reshape((neuron_count, 1))
            return x
        elif 2 <= layer_index <= self.structure.hidden_layers_count() + 2:
            previous_activations = self.neuron_activations(data_set, layer_index - 1, sample_index)
            neuron_count = previous_activations.shape[0]
            previous_activations = np.concatenate((np.ones((1, 1)), previous_activations), axis=0).reshape(
                (neuron_count + 1, 1))
            return sigmoid(np.dot(self.weights_matrices[layer_index - 2], previous_activations))
        else:
            raise IndexError

# if __name__ == "__main__":
#     data = np.arange(40).reshape((20, 2))
#     labels = np.reshape(np.arange(4).repeat(5), (20, 1))
#     np.random.shuffle(labels)
#     print(data)
#     print(labels)
#     Classifier(data, labels)
