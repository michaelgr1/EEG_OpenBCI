import math
from enum import Enum

import numpy as np

import statistics as stats

DEFAULT_TRAINING_SPLIT = 0.6
DEFAULT_CROSS_VALIDATION_SPLIT = 0.2
DEFAULT_TEST_SET_SPLIT = 0.2


class FeatureScalingType(Enum):

	MIN_MAX_SCALING = 0

	STANDARDIZATION = 1

	MEAN_NORMALIZATION = 2

	UNIT_LENGTH = 3

	NO_SCALING = 4


class FeatureScalar:
	"""
		This class provides a way to apply feature scaling of different types.
		All the values which are used during the calculations are based on the feature matrix
		passed in the constructor.
	"""

	def __init__(self, scaling_type: FeatureScalingType, feature_matrix: np.ndarray):

		# Feature matrix should contain rows as samples and columns as features
		self.feature_matrix = feature_matrix
		self.feature_count = feature_matrix.shape[1]

		self.features_minimum = []
		self.features_maximum = []

		self.features_mean = []
		self.features_standard_deviation = []

		self.scaling_type = scaling_type

		for n in range(self.feature_count):
			feature_data = feature_matrix[:, n]

			self.features_minimum.append(np.min(feature_data))
			self.features_maximum.append(np.max(feature_data))

			self.features_mean.append(stats.arithmetic_mean(feature_data))
			self.features_standard_deviation.append(stats.sample_standard_deviation(feature_data))

	def scaled_feature_matrix(self):
		return self.scale_feature_matrix(self.feature_matrix)

	def scale_feature_matrix(self, feature_matrix: np.ndarray):

		normalized_feature_matrix = np.zeros_like(feature_matrix)

		for m in range(feature_matrix.shape[0]):
			sample = feature_matrix[m, :]
			normalized_feature_matrix[m, :] = self.scale_sample(sample)

		return normalized_feature_matrix

	def scale_sample(self, sample: np.ndarray) -> np.ndarray:
		original_shape = sample.shape
		sample = sample.flatten()  # Array should represent only a single feature so it should not have more than 1 dimension

		if sample.shape[0] != self.feature_count:
			print("Unexpected size for sample")
			raise ValueError()
		if self.scaling_type == FeatureScalingType.NO_SCALING:
			sample = sample  # Do nothing!
		if self.scaling_type == FeatureScalingType.MIN_MAX_SCALING:

			if len(self.features_minimum) != self.feature_count or len(self.features_maximum) != self.feature_count:
				raise ValueError()

			for n in range(self.feature_count):
				sample[n] = (sample[n] - self.features_minimum[n]) / (self.features_maximum[n] - self.features_minimum[n])

		elif self.scaling_type == FeatureScalingType.STANDARDIZATION:

			if len(self.features_standard_deviation) != self.feature_count or len(self.features_mean) != self.feature_count:
				raise ValueError()

			for n in range(self.feature_count):
				sample[n] = (sample[n] - self.features_mean[n]) / self.features_standard_deviation[n]

		elif self.scaling_type == FeatureScalingType.MEAN_NORMALIZATION:

			if len(self.features_mean) != self.feature_count or len(self.features_minimum) != self.feature_count or\
				len(self.features_maximum) != self.feature_count:

				raise ValueError()

			for n in range(self.feature_count):
				sample[n] = (sample[n] - self.features_mean[n]) / (self.features_maximum[n] - self.features_minimum[n])

		elif self.scaling_type == FeatureScalingType.UNIT_LENGTH:
			sample = sample / np.linalg.norm(sample)

		return sample.reshape(original_shape)  # Reshapes the data into it original shape.


def append_x0(data: np.ndarray) -> np.ndarray:
	if data.ndim == 1:
		return np.append(1, data)
	else:
		m = data.shape[0]
		# Add X0 -> column of ones to be multiplied with bias term
		return np.append(np.ones((m, 1)), data, axis=1)


def is_vector(arr: np.ndarray) -> bool:
	"""
		Returns true if the given array is either 1 dimensional, or it has 2 dimensions with one of their lengths equals 1
	:param arr:
	:return: True if vector, False otherwise
	"""

	return arr.ndim == 1 or (arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1))


def construct_feature_matrix(feature_vectors: []):
	"""
		This method constructs a feature matrix from the given feature vectors.
		The features vectors are added as rows in the matrix. Their size should be the same.
		The matrix will have a size of m by n where m is the length of the provided list and n is the feature count
	"""

	feature_matrix = np.empty((len(feature_vectors), feature_vectors[0].data.shape[1]))

	for i in range(len(feature_vectors)):
		feature = feature_vectors[i].data
		if feature.shape[1] == feature_matrix.shape[1]:
			feature_matrix[i, :] = feature

	return feature_matrix


class DataSubSetType(Enum):

	TRAINING = 0

	CROSS_VALIDATION = 1

	TESTING = 2


class DataSet:
	"""
		A class for holding data sets for machine learning models.
		Data matrix passed in the constructor should contain rows as samples and columns as features.
	"""

	def __init__(self, data_matrix: np.ndarray, labels: np.ndarray, add_x0: bool, shuffle: bool = True,
				training_split: float = DEFAULT_TRAINING_SPLIT,
				cross_validation_split: float = DEFAULT_CROSS_VALIDATION_SPLIT,
				test_set_split: float = DEFAULT_TEST_SET_SPLIT):

		if data_matrix.shape[0] != labels.shape[0]:
			print("Size of data set doesn't match size of labels")
			raise ValueError

		if not (0 < (training_split + cross_validation_split + test_set_split) <= 1):
			raise ValueError("Invalid split values...")

		data_matrix = data_matrix.copy()

		if labels.ndim == 1:
			labels = labels.reshape(-1, 1)
		else:
			labels = labels.copy()

		self.add_x0 = add_x0

		# Number of samples in the given data matrix
		m = data_matrix.shape[0]

		training_set_size = int(math.floor(m * training_split))
		cross_validation_size = int(math.floor(m * cross_validation_split))
		test_set_size = int(math.floor(m * test_set_split))

		leftover = m - (training_set_size + cross_validation_size + test_set_size)
		print("Leftover after data set divided = {}".format(leftover))

		set_assignment_arr = np.empty((m - leftover, 1))

		# Zeros for training set samples
		set_assignment_arr[0:training_set_size, 0] = 0

		# Ones for cross validation samples
		set_assignment_arr[training_set_size:training_set_size + cross_validation_size, 0] = 1

		# Twos for test set samples
		set_assignment_arr[training_set_size + cross_validation_size:, 0] = 2

		# Add the leftovers to the training set
		set_assignment_arr = np.append(set_assignment_arr, np.zeros(leftover))

		if shuffle:
			np.random.shuffle(set_assignment_arr)

		# Assign training set
		self.raw_training_set = data_matrix[set_assignment_arr == 0, :]
		self.training_set_labels = labels[set_assignment_arr == 0, :]

		# Create a feature scalar with NO_SCALING as its type. Can be used later to apply different scaling techniques
		# Learn the feature scaling parameters only from the training set
		self.feature_scalar = FeatureScalar(FeatureScalingType.NO_SCALING, self.raw_training_set)

		self.scaled_training_set = self.process_data(self.raw_training_set)

		# Assign cross validation set
		self.raw_cross_validation_set = data_matrix[set_assignment_arr == 1, :]
		self.cross_validation_labels = labels[set_assignment_arr == 1, :]

		self.scaled_cross_validation_set = self.process_data(self.raw_cross_validation_set)

		# Assign test set
		self.raw_test_set = data_matrix[set_assignment_arr == 2, :]
		self.test_set_labels = labels[set_assignment_arr == 2, :]

		self.scaled_test_set = self.process_data(self.raw_test_set)

	def apply_feature_scaling(self, scaling_type: FeatureScalingType):
		if scaling_type != self.feature_scalar.scaling_type:
			self.feature_scalar.scaling_type = scaling_type
			self.reprocess_subsets()

	def reprocess_subsets(self):
		self.scaled_training_set = self.process_data(self.raw_training_set)
		self.scaled_cross_validation_set = self.process_data(self.raw_cross_validation_set)
		self.scaled_test_set = self.process_data(self.raw_test_set)

	def get_training_set(self) -> np.ndarray:
		"""
		Get the scaled training set
		:return: Scaled training set
		"""
		return self.scaled_training_set

	def get_cross_validation_set(self) -> np.ndarray:
		"""
		Get the scaled cross validation set
		:return: Scaled cross validation set
		"""
		return self.scaled_cross_validation_set

	def get_test_set(self) -> np.ndarray:
		"""
		Get the scaled test set
		:return: Scaled test set
		"""
		return self.scaled_test_set

	def feature_count(self) -> int:
		"""
		Returns the number of features in the data set not including the X0 term.
		"""
		return self.raw_training_set.shape[1]

	def sample_count(self, subset_type: DataSubSetType):
		if subset_type == DataSubSetType.TRAINING:
			return self.raw_training_set.shape[0]
		elif subset_type == DataSubSetType.CROSS_VALIDATION:
			return self.raw_cross_validation_set.shape[0]
		elif subset_type == DataSubSetType.TESTING:
			return self.raw_test_set.shape[0]
		else:
			return -1

	def sample_at(self, index: int, subset_type: DataSubSetType) -> np.ndarray:
		if subset_type == DataSubSetType.TRAINING:
			return self.get_training_set()[index, :]
		elif subset_type == DataSubSetType.CROSS_VALIDATION:
			return self.get_cross_validation_set()[index, :]
		elif subset_type == DataSubSetType.TESTING:
			return self.get_test_set()[index, :]

	def label_at(self, index: int, subset_type: DataSubSetType) -> float:
		if subset_type == DataSubSetType.TRAINING:
			return self.training_set_labels[index, 0]
		elif subset_type == DataSubSetType.CROSS_VALIDATION:
			return self.cross_validation_labels[index, 0]
		elif subset_type == DataSubSetType.TESTING:
			return self.test_set_labels[index, 0]

	def unique_labels(self):
		return np.unique(self.feature_matrix_labels())

	def process_data(self, data: np.ndarray) -> np.ndarray:
		"""
			Applies feature scaling and adds x0 if necessary
		:param data: data to be processed
		:return: processed data according to the applied feature scaling type.
		"""
		if is_vector(data):
			scaled_sample = self.feature_scalar.scale_sample(data)
			if self.add_x0:
				return append_x0(scaled_sample)
			else:
				return scaled_sample
		else:
			scaled_data = self.feature_scalar.scale_feature_matrix(data)
			if self.add_x0:
				return append_x0(scaled_data)
			else:
				return scaled_data

	def raw_feature_matrix(self):
		return np.vstack((self.raw_training_set, self.raw_cross_validation_set, self.raw_test_set))

	def scaled_feature_matrix(self):
		return self.feature_scalar.scale_feature_matrix(self.raw_feature_matrix())

	def feature_matrix_labels(self):
		return np.vstack((self.training_set_labels, self.cross_validation_labels, self.test_set_labels))

	def append_to(self, data: np.ndarray, labels: np.ndarray, target: DataSubSetType):
		"""
			Appends the given data to the desired sub set (training/cross validation/testing)
		:param data: Data to be added, should not be normalized and should not contain X0 column (raw).
		:param labels: Corresponding labels to the given data
		:param target: Type of target subset
		:return: None
		"""

		data = self.process_data(data)

		if data.ndim == 1:
			data = data.reshape((1, -1))  # Reshape data as a row

		if labels.ndim == 1:
			labels = labels.reshape((-1, 1))  # Reshape labels as a column vector

		if target == DataSubSetType.TRAINING:
			self.raw_training_set = np.append(self.raw_training_set, data, axis=0)
			self.training_set_labels = np.append(self.training_set_labels, labels, axis=0)

			# Update feature scaling according to the new samples
			self.feature_scalar = FeatureScalar(self.feature_scalar.scaling_type, self.raw_training_set)

		elif target == DataSubSetType.CROSS_VALIDATION:
			self.raw_cross_validation_set = np.append(self.raw_cross_validation_set, data, axis=0)
			self.cross_validation_labels = np.append(self.cross_validation_labels, labels, axis=0)
		elif target == DataSubSetType.TESTING:
			self.raw_test_set = np.append(self.raw_test_set, data, axis=0)
			self.test_set_labels = np.append(self.test_set_labels)

		self.reprocess_subsets()
