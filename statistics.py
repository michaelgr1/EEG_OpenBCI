import numpy as np
import math


def arithmetic_mean(samples: np.ndarray) -> float:
    samples = samples.flatten()
    return np.sum(samples) / samples.shape[0]


def sample_variance(samples: np.ndarray) -> float:
    samples = samples.flatten()
    mean = arithmetic_mean(samples)

    distance_from_mean = samples - mean

    return np.sum(np.power(distance_from_mean, 2)) / (samples.shape[0] - 1)


def sample_standard_deviation(samples: np.ndarray) -> float:
    variance = sample_variance(samples)
    return math.sqrt(variance)


def sample_covariance(x: np.ndarray, y: np.ndarray) -> float:
    x = x.flatten()
    y = y.flatten()

    x_mean = arithmetic_mean(x)
    y_mean = arithmetic_mean(y)

    return np.sum((x - x_mean) * (y - y_mean)) / (x.shape[0] - 1)


def sample_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return sample_covariance(x, y) / (sample_standard_deviation(x) * sample_standard_deviation(y))


def sample_covariance_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """
        Computes the sample covariance matrix.
        Feature matrix should contain rows as samples and columns as features
    :param feature_matrix:
    :return:
    """

    n = feature_matrix.shape[1]

    covariance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            covariance_matrix[i, j] = sample_covariance(feature_matrix[:, i], feature_matrix[:, j])

    return covariance_matrix


def population_variance(data: np.ndarray) -> float:
    data = data.flatten()
    mean = arithmetic_mean(data)

    distance_from_mean = data - mean

    return np.sum(np.power(distance_from_mean, 2)) / data.shape[0]


def population_standard_deviation(data: np.ndarray) -> float:
    variance = population_variance(data)
    return math.sqrt(variance)


def population_covariance(x: np.ndarray, y: np.ndarray) -> float:
    x = x.flatten()
    return sample_covariance(x, y) * (x.shape[0] - 1) / x.shape[0]


def population_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return population_covariance(x, y) / (population_standard_deviation(x) * population_standard_deviation(y))


def population_covariance_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """
        Computes the population covariance matrix.
        Feature matrix should contain rows as samples and columns as features
    :param feature_matrix:
    :return: Covariance matrix
    """

    n = feature_matrix.shape[1]

    covariance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            covariance_matrix[i, j] = population_covariance(feature_matrix[:, i], feature_matrix[:, j])

    return covariance_matrix
