import math
import sys
import time
from enum import Enum

import PyQt5.QtCore
import numpy as np
import serial.tools.list_ports
from PyQt5.QtChart import QValueAxis, QBarSet
from PyQt5.QtWidgets import QWidget, QHBoxLayout
from brainflow import BoardShim
from brainflow.data_filter import DataFilter, WindowFunctions, FilterTypes

import global_config
import numerical


class AccumulatingAverage:
	""""
		This class can be used to keep track of an average that is changing.
		You can add items and get the average of all the previously added items.
		A copy of the items is not saved, but rather their sum.
	"""

	def __init__(self):
		self.sum = 0
		self.count = 0

	def add_value(self, value: float):
		self.sum += value
		self.count += 1

	def compute_average(self):
		if self.count == 0:
			return 0

		return self.sum / self.count

	def reset(self):
		self.sum = 0
		self.count = 0


class AccumulatingAverages:
	""""
		Same as accumulating average above, but for numpy arrays.
	"""

	def __init__(self):
		self.sum = None
		self.count = 0

	def add_values(self, values: np.ndarray):
		if self.sum is None:
			self.sum = values.copy()
		else:
			self.sum = self.sum + values
		self.count += 1

	def compute_average(self) -> float:
		if self.count == 0:
			return 0
		return self.sum / self.count

	def reset(self):
		self.sum = None
		self.count = 0


class EegChannelConfigurations:

	def __init__(self, board_id: int, channel,
				visible_seconds=5, vertical_scale=400, notch_freq=50, bandpass_min_freq=5.0, bandpass_max_freq=50.0):
		self.change_listener = None
		self.channel = channel

		self.visible_seconds = visible_seconds  # In seconds
		self.vertical_scale = vertical_scale  # In μV
		self.notch_freq = notch_freq  # In Hz
		self.bandpass_min_freq = bandpass_min_freq  # In Hz
		self.bandpass_max_freq = bandpass_max_freq  # In Hz
		self.board_id = board_id

	def set_visible_seconds(self, visible_seconds: int, notify_owner=False, notify_listener=True):
		self.visible_seconds = visible_seconds

		if notify_owner:
			self.channel.horizontal_scale_changed(index=-1)

		if notify_listener:
			self.__on_change()

	def set_vertical_scale(self, vertical_scale, notify_owner=False, notify_listener=True):
		self.vertical_scale = vertical_scale
		if notify_owner:
			self.channel.vertical_scale_changed(index=-1)

		if notify_listener:
			self.__on_change()

	def set_notch_freq(self, notch_freq, notify_owner=False, notify_listener=True):
		self.notch_freq = notch_freq
		if notify_listener:
			self.__on_change()

	def set_bandpass_min_freq(self, bandpass_min_freq, notify_owner=False, notify_listener=True):
		self.bandpass_min_freq = bandpass_min_freq

		if notify_owner:
			self.channel.bandpass_min_changed(index=-1)

		if notify_listener:
			self.__on_change()

	def set_bandpass_max_freq(self, bandpass_max_freq, notify_owner=False, notify_listener=True):
		self.bandpass_max_freq = bandpass_max_freq

		if notify_owner:
			self.channel.bandpass_max_changed(index=-1)

		if notify_listener:
			self.__on_change()

	def __on_change(self):
		if self.change_listener is not None:
			self.change_listener(self)

	def window_size(self):
		""""
			Returns the window size in samples
		"""
		return self.visible_seconds * BoardShim.get_sampling_rate(self.board_id)

	def match_config(self, other):
		""""
			Matches the current configurations to those in the given object
		"""
		if type(other) == EegChannelConfigurations:
			self.set_visible_seconds(other.visible_seconds, notify_owner=True, notify_listener=False)
			self.set_vertical_scale(other.vertical_scale, notify_owner=True, notify_listener=False)
			self.set_notch_freq(other.notch_freq, notify_owner=True, notify_listener=False)
			self.set_bandpass_min_freq(other.bandpass_min_freq, notify_owner=True, notify_listener=False)
			self.set_bandpass_max_freq(other.bandpass_max_freq, notify_owner=True, notify_listener=False)
			self.board_id = other.board_id


class FrequencyBand:

	def __init__(self, min_frequency: float, max_frequency: float):
		self.min_frequency = abs(min_frequency)
		self.max_frequency = abs(max_frequency)

		if self.min_frequency > self.max_frequency:
			self.min_frequency, self.max_frequency = self.max_frequency, self.min_frequency

	def contains_freq(self, frequency: float):
		return self.min_frequency <= frequency <= self.max_frequency

	@staticmethod
	def delta_freq_band():
		return FrequencyBand(1, 3)

	@staticmethod
	def theta_freq_band():
		return FrequencyBand(4, 7)

	@staticmethod
	def alpha_freq_band():
		return FrequencyBand(8, 13)

	@staticmethod
	def beta_freq_band():
		return FrequencyBand(13, 30)

	@staticmethod
	def gama_freq_band():
		return FrequencyBand(30, 100)


class FeatureExtractor:
	""""
		This class defines several useful methods for extracting eeg features from a given sample.
	"""

	def __init__(self, channel_data: np.ndarray, sampling_rate: int):
		self.channel_data = channel_data
		self.sampling_rate = sampling_rate

	def fft(self, window_size: float = 0) -> (np.ndarray, np.ndarray):
		""""
			This method computes the fft values for the given data.
			The window size (in seconds) specifies how to divide the data, compute the fft values for each
			window and then average all. By default, the windows overlap in half their size.
			The default window size is zero, in that case, no windows are used!
			It returns two arrays with the same size.
			The first one contains the frequencies and the second
			contains their corresponding power.
		"""
		if window_size != 0:

			window_size_in_samples = math.floor(self.sampling_rate * window_size)

			if window_size_in_samples >= self.channel_data.shape[0]:
				print("Window to big for data, not using!")
				return self.fft(window_size=0)

			window_length = pow(2, closest_power_of_two(window_size_in_samples))

			half_window_size_in_samples = math.floor(window_size_in_samples / 2)

			windows_count = math.floor(self.channel_data.shape[0] / window_size_in_samples)
			windows_count += windows_count - 1  # Add overlaps. Each non last window has an overlapping window

			fft_average = AccumulatingAverages()

			for window_index in range(0, windows_count):
				start_index = half_window_size_in_samples * window_index
				end_index = start_index + window_size_in_samples

				window_data = self.channel_data[start_index:end_index]

				# TODO: Find out whether a window function should be used
				amplitudes = abs(DataFilter.perform_fft(window_data[:window_length], WindowFunctions.NO_WINDOW.value))
				amplitudes = amplitudes * (1 / window_length)
				fft_average.add_values(amplitudes)

			base_freq = self.sampling_rate / window_length

			frequencies = np.linspace(0, window_length - 1, window_length) * base_freq

			frequencies = frequencies[0:int(window_length / 2 + 1)]

			return frequencies, fft_average.compute_average()
		else:
			length = pow(2, closest_power_of_two(self.channel_data.shape[0]))
			# frequencies = np.linspace(0, self.sampling_rate / 2, int(length / 2 + 1))

			base_freq = self.sampling_rate / length

			frequencies = np.linspace(0, length - 1, length) * base_freq

			frequencies = frequencies[0:int(length / 2 + 1)]

			powers = abs(DataFilter.perform_fft(self.channel_data[:length], WindowFunctions.NO_WINDOW.value))
			return frequencies, powers * (1 / length)

	def frequency_amplitude(self, frequency: float, window_size: float = 0) -> float:
		""""
			Computes and returns the amplitude of the frequency closest to the given one using fft.
		"""
		frequencies, amplitudes = self.fft(window_size)

		for i in range(len(frequencies)):
			if frequencies[i] == frequency:
				return amplitudes[i]
			if frequencies[i] > frequency:  # First larger frequency
				distance_to_larger_freq = abs(frequencies[i] - frequency)
				if i - 1 > 0:
					distance_to_smaller_freq = abs(frequencies[i - 1] - frequency)
				else:
					print("Closest freq = {}".format(frequencies[i]))
					return amplitudes[i]
				if distance_to_larger_freq < distance_to_smaller_freq:
					print("Closest freq = {}".format(frequencies[i]))
					return amplitudes[i]
				else:
					print("Closest freq = {}".format(frequencies[i - 1]))
					return amplitudes[i - 1]

	def average_band_amplitude(self, freq_band: FrequencyBand, window_size: float = 0) -> float:
		""""
			Computes the average frequency amplitude for all the frequencies between
			min_freq and max_freq using fft.
		"""

		min_freq = freq_band.min_frequency
		max_freq = freq_band.max_frequency

		frequencies, amplitudes = self.fft(window_size)

		min_freq_index = -1
		max_freq_index = -1

		for i in range(len(frequencies)):
			if frequencies[i] == min_freq:
				min_freq_index = i

			if frequencies[i] > min_freq and min_freq_index == -1:
				if i - 1 >= 0:
					min_freq_index = i - 1
				else:
					min_freq_index = i

			if frequencies[i] == max_freq:
				max_freq_index = i
				break

			if frequencies[i] > max_freq and max_freq_index == -1:
				max_freq_index = i

		average = AccumulatingAverage()

		for i in range(len(frequencies)):
			if min_freq_index <= i <= max_freq_index:
				average.add_value(amplitudes[i])

		return average.compute_average()


class EegData:
	""""
		This class holds eeg data and provides access to each channel.
		Provided data should have rows as channels and columns as samples.
	"""

	def __init__(self, data: np.ndarray = None):
		self.channels = []
		if data is not None:
			for row in range(data.shape[0]):
				self.channels.append(data[row, :])

	def append_data(self, data: np.ndarray):

		if len(self.channels) == 0:
			for row in range(data.shape[0]):
				self.channels.append(data[row, :])
			return

		# columns = data.shape[1]
		for row in range(data.shape[0]):
			# channel_data = data[row, :].reshape((1, columns))
			channel_data = data[row, :]

			if row < len(self.channels):
				self.channels[row] = np.concatenate((self.channels[row], channel_data))

	def clear(self):
		self.channels.clear()

	def get_channel_data(self, channel: int) -> np.ndarray:
		""""
			Get the channel data from the specified channel.
			zero indexed! empty array when channel is out of bounds
		"""
		if channel < len(self.channels):
			return self.channels[channel]
		else:
			return np.empty(1)

	def filter_all_channels(self, sampling_rate: int, bandpass_min: float, bandpass_max: float,
							subtract_average: bool, notch_filter: bool = True, notch_freq: float = 50):

		for i in range(len(self.channels)):
			channel_data = self.channels[i]
			self.channels[i] = filter_data(channel_data, sampling_rate, bandpass_min, bandpass_max, subtract_average, notch_filter, notch_freq)

	def feature_extractor(self, channel: int, sampling_rate: int) -> FeatureExtractor:
		return FeatureExtractor(self.get_channel_data(channel), sampling_rate)

	def to_row_array(self) -> np.ndarray:
		"""
			Combines the data into a 2d array with rows as channels and columns as samples. Filtered if filters were applied.
		:return: A 2d array with all the eeg data.
		"""

		data = np.zeros((0, self.channels[0].shape[0]))

		for channel_data in self.channels:
			channel_data = channel_data.reshape((1, channel_data.shape[0]))
			data = np.concatenate((data, channel_data), axis=0)

		return data


class FeatureVector:

	def __init__(self):
		self.data = None

	def append_feature(self, feature: float):
		if self.data is None:
			self.data = np.empty((1, 1))
			self.data[0][0] = feature
		else:
			self.data = np.append(self.data, np.array([[feature]]), axis=1)

	def append_features(self, features: np.ndarray):
		if self.data is None:
			self.data = features.copy()
		else:
			self.data = np.concatenate((self.data, features), axis=1)


class FilterSettings:

	def __init__(self, sampling_rate: int, low_pass: float, high_pass, subtract_average: bool = True, notch_filter: bool = True, notch_freq: float = 50):
		self.sampling_rate = sampling_rate
		self.low_pass = low_pass
		self.high_pass = high_pass
		self.subtract_average = subtract_average
		self.notch_filter = notch_filter
		self.notch_freq = notch_freq

	def apply(self, data: np.ndarray) -> np.ndarray:
		eeg_data = EegData(data)
		eeg_data.filter_all_channels(self.sampling_rate, self.low_pass, self.high_pass, self.subtract_average,
										self.notch_filter, self.notch_freq)
		return eeg_data.to_row_array()


class AverageBandAmplitudeFeature:
	"""
		This class holds the necessary information in order to extract the average band amplitude feature from the desired electrodes.
		It doesn't provide any way to compute it, thus, it's a container more than a functional class.
	"""

	def __init__(self, frequency_band: FrequencyBand, fft_window_size: float):
		self.frequency_band = frequency_band
		self.fft_window_size = fft_window_size


class FrequencyBandsAmplitudeFeature:
	"""
		This class holds the necessary information in order to extract multiple frequency band features from multiple electrodes.
		It does not provide any way to compute it.
	"""

	def __init__(self, center_frequencies: [], band_width: float, fft_window_size: float):
		self.fft_window_size = fft_window_size
		self.center_frequencies = center_frequencies
		self.band_width = band_width

	def frequency_band_at(self, index: int) -> FrequencyBand:
		min_freq = self.center_frequencies[index] - self.band_width / 2
		max_freq = self.center_frequencies[index] + self.band_width / 2

		return FrequencyBand(min_freq, max_freq)


class FeatureExtractionInfo:

	def __init__(self, sampling_rate: int, first_channel: int, last_channel: int):
		self.sampling_rate = sampling_rate
		self.first_channel = first_channel
		self.last_channel = last_channel


class Direction(Enum):
	LEFT = 0
	RIGHT = 1
	FORWARD = 2
	BACKWARD = 3


class TrialClass:

	def __init__(self, name: str, image_path: str, label: int, direction: Direction):
		self.name = name
		self.image_path = image_path
		self.label = label
		self.direction = direction

	def __eq__(self, other):
		if type(other) == TrialClass:
			return self.name == other.name and self.image_path == other.image_path and self.label == other.label
		else:
			return self == other


class VibroTactileClasses:
	LEFT_CLASS = TrialClass(
		"SS-L", global_config.IMAGES_SSD_DRIVER_LETTER + ":/EEG_GUI_OpenBCI/class_images/SS-L.png", 0, Direction.LEFT
	)

	RIGHT_CLASS = TrialClass(
		"SS-R", global_config.IMAGES_SSD_DRIVER_LETTER + ":/EEG_GUI_OpenBCI/class_images/SS-R.png", 1, Direction.RIGHT
	)

	BOTH_CLASS = TrialClass(
		"SS-B", global_config.IMAGES_SSD_DRIVER_LETTER + ":/EEG_GUI_OpenBCI/class_images/SS-B.png", 2, Direction.FORWARD
	)

	NONE_CLASS = TrialClass(
		"SS-S", global_config.IMAGES_SSD_DRIVER_LETTER + ":/EEG_GUI_OpenBCI/class_images/SS-S.png", 3, Direction.BACKWARD
	)

	ALL = [LEFT_CLASS, RIGHT_CLASS, BOTH_CLASS, NONE_CLASS]


class AlphaRhythmClasses:

	OPENED_EYES_CLASS = \
		TrialClass("Opened Eyes", global_config.IMAGES_SSD_DRIVER_LETTER + ":/EEG_GUI_OpenBCI/class_images/eyes/eyes_opened.jpg", 0, Direction.FORWARD)

	CLOSED_EYES_CLASS = \
		TrialClass("Closed Eyes", global_config.IMAGES_SSD_DRIVER_LETTER + ":/EEG_GUI_OpenBCI/class_images/eyes/eyes_closed.jpg", 1, Direction.BACKWARD)

	ALL = [OPENED_EYES_CLASS, CLOSED_EYES_CLASS]


class SsvepClasses:

	LEFT_LED = TrialClass("Left LED", global_config.IMAGES_SSD_DRIVER_LETTER + ":/EEG_GUI_OpenBCI/class_images/left_arrow.png", 0, Direction.LEFT)

	RIGHT_LED = TrialClass("Right LED", global_config.IMAGES_SSD_DRIVER_LETTER + ":/EEG_GUI_OpenBCI/class_images/right_arrow.png", 1, Direction.RIGHT)

	ALL = [LEFT_LED, RIGHT_LED]


class SliceIndexGenerator:

	def __init__(self, sampling_rate: int, trial_classes: [TrialClass]):
		self.sampling_rate = sampling_rate
		self.trial_classes = trial_classes
		# This array will contain tuples in the form of (str,int,int) where the str value is the label,
		# the first int value is the first sample index, and the second int value is the last sample index.
		self.slices = []

	def add_slice(self, class_label: int, start_sample_index: int, end_sample_index: int):
		for trial_class in self.trial_classes:
			if trial_class.label == class_label:
				self.slices.append((class_label, start_sample_index, end_sample_index))
				break
		else:
			print("Class label not in trial classes, ignoring...")

	def write_to_file(self, root_directory_path: str, append: bool = False):

		with open(root_directory_path + f"/{global_config.SLICE_INDEX_FILE_NAME}", "a") as file:
			if not append:
				file.write(f"{self.sampling_rate}\n")
				for trial_class in self.trial_classes:
					if trial_class is not self.trial_classes[-1]:
						file.write(f"{trial_class_as_string(trial_class)},")
					else:
						file.write(f"{trial_class_as_string(trial_class)}")
				file.write("\n")

			for slice_tuple in self.slices:
				file.write(f"{slice_tuple[0]},{slice_tuple[1]},{slice_tuple[2]}\n")


def obtain_sampling_rate_from_slice_index(root_directory_path: str) -> int:
	file = open(root_directory_path + f"/{global_config.SLICE_INDEX_FILE_NAME}", "r")
	sampling_rate = int(file.readline())
	file.close()
	return sampling_rate


def obtain_trial_classes_from_slice_index(root_directory_path: str) -> [TrialClass]:
	file = open(root_directory_path + f"/{global_config.SLICE_INDEX_FILE_NAME}", "r")
	file.readline()  # Discard sampling rate

	trial_classes = []

	trial_classes_str = file.readline().split(",")

	for string in trial_classes_str:
		trial_classes.append(trial_class_from_string(string))

	return trial_classes


def obtain_last_trial_index_from_slice(root_directory_path: str) -> int:
	file = open(root_directory_path + f"/{global_config.SLICE_INDEX_FILE_NAME}", "r")

	line = ""
	last_line = ""

	while True:
		last_line = line
		line = file.readline()
		if not line:
			break
	return int(last_line.split(",")[-1])


def obtain_trial_length_from_slice_index(root_directory_path: str) -> float:
	"""
	Obtains the trial length from the slice index. Calculates it by looking at the first record and subtracting
	the last index from the first. The value returned is in seconds not in samples!
	:param root_directory_path: The path to the root directory which contains the slice index file.
	:return: The trial length in seconds.
	"""
	file = open(root_directory_path + f"/{global_config.SLICE_INDEX_FILE_NAME}", "r")

	sampling_rate = int(file.readline())

	file.readline()  # Discard trial classes line

	elements = file.readline().split(",")
	start_index = int(elements[1])
	end_index = int(elements[2])

	length_in_samples = end_index - start_index

	return length_in_samples / sampling_rate


def load_data(root_directory_path: str) -> np.ndarray:
	data_file_path = root_directory_path + f"/{global_config.EEG_DATA_FILE_NAME}"

	print(data_file_path)

	raw_unsliced_data = DataFilter.read_file(data_file_path)

	return raw_unsliced_data


def slice_and_filter_data(root_directory_path: str, filter_settings: FilterSettings,
							raw_eeg_data: np.ndarray = None) -> ([EegData], [int], int, [TrialClass]):

	""""
		Reads the data from the root directory and slices it according to the instructions in the slice index file.
		Returns a tuple contain a list of EegData object as its first element, another list of the same size containing
		their corresponding labels, an integer specifying the sampling rate which was used to record the given data, and a list
		or trial class objects which correspond to the used trial classes and contain their names and image paths.
		The root directory should contain two files, one which contains all the data, and another which specifies how to
		slice it according to a custom format.
	"""

	if raw_eeg_data is not None:
		raw_unsliced_data = raw_eeg_data
	else:
		raw_unsliced_data = load_data(root_directory_path)

	slice_index_file = open(root_directory_path + f"/{global_config.SLICE_INDEX_FILE_NAME}", "r")

	sampling_rate = int(slice_index_file.readline())

	filter_settings.sampling_rate = sampling_rate

	filtered_unsliced_data = filter_settings.apply(raw_unsliced_data)

	data_list = []
	classes_list = []

	trial_classes = []

	row_count = 1  # Sampling rate has already been read

	while True:

		row_count += 1

		line = slice_index_file.readline()

		if not line:
			break

		if row_count == 2:  # Trial Classes
			trial_classes_str = line.split(",")

			for string in trial_classes_str:
				trial_classes.append(trial_class_from_string(string))

		if row_count >= 3:
			elements = line.split(",")
			label = int(elements[0])
			start_index = int(elements[1])
			end_index = int(elements[2])

			data_list.append(EegData(filtered_unsliced_data[:, start_index:end_index]))
			classes_list.append(label)

	slice_index_file.close()

	return data_list, classes_list, sampling_rate, trial_classes


def extract_features(data_list: [EegData], info: FeatureExtractionInfo, feature_types: []) -> [FeatureVector]:
	"""
	This method takes in a list of EegData objects. For each data object it extracts the features as specified in
	the second argument which is a list which should contain supported class instances.
	Currently, the only possible feature types are AverageBAndAmplitudeFeature and FrequencyBandsAmplitudeFeature.
	:param info: An object containing some info about the data and the channels from which features
					should be extract.
	:param data_list: A list containing EegData objects. Data should be filtered!
	:param feature_types: A list of feature types.
	:return: A list of FeatureVectors corresponding to each EegData object.
	"""

	# Extract features

	# Prepare list with empty feature vectors

	extracted_data = []

	for i in range(len(data_list)):  # For each EegData item in the list
		feature_vector = FeatureVector()

		for channel in range(info.first_channel - 1, info.last_channel):  # For each channel

			# Data is assumed to be in a format where a row represent all the data from a given channel.
			channel_data = data_list[i].get_channel_data(channel)
			feature_extractor = FeatureExtractor(channel_data, info.sampling_rate)

			for feature_type in feature_types:  # Extract and append each

				if type(feature_type) == AverageBandAmplitudeFeature:

					fft_window_size = feature_type.fft_window_size

					channel_band_amplitude = feature_extractor.average_band_amplitude(feature_type.frequency_band, fft_window_size)

					feature_vector.append_feature(channel_band_amplitude)

				if type(feature_type) == FrequencyBandsAmplitudeFeature:

					fft_window_size = feature_type.fft_window_size

					for freq_index in range(len(feature_type.center_frequencies)):

						freq_band = feature_type.frequency_band_at(freq_index)

						band_amplitude = feature_extractor.average_band_amplitude(freq_band, fft_window_size)

						feature_vector.append_feature(band_amplitude)

		extracted_data.append(feature_vector)

	return extracted_data


def filter_data(data: np.ndarray, sampling_rate: int,
				bandpass_min: float, bandpass_max: float,
				subtract_average: bool, notch_filter: bool = True, notch_freq: float = 50):

	if subtract_average:
		data = data - np.average(data)

	DataFilter.perform_bandpass(data, sampling_rate, (bandpass_max + bandpass_min) / 2,
								bandpass_max - bandpass_min, 4, FilterTypes.BUTTERWORTH.value, 0)
	if notch_filter:
		DataFilter.perform_bandstop(data, sampling_rate, notch_freq, 2, 4, FilterTypes.BUTTERWORTH.value, 0)

	return data


def construct_horizontal_box(widgets: [QWidget]) -> QWidget:
	widget = QWidget()
	layout = QHBoxLayout()
	layout.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
	widget.setLayout(layout)

	for item in widgets:
		layout.addWidget(item)
	return widget


def closest_power_of_two(number: int):
	""""
		Returns x where: 2 ** x <= number and 2 ** (x + 1) > number
	"""

	for i in range(0, number):
		if pow(2, i) > number:
			return i - 1


def sample_to_second(sample: int, board_id: int):
	""""
		This function takes a sample number and a board id.
		It returns the seconds that have passed from sample 0 to the given sample,
		based on the sampling frequency of the given board.
	"""

	return sample / BoardShim.get_sampling_rate(board_id)


def samples_to_seconds(samples: np.ndarray, board_id: int) -> np.ndarray:
	""""
		Same as sample to second but with an array
	"""
	return samples / BoardShim.get_sampling_rate(board_id)


def distance_between(x1: np.ndarray, x2: np.ndarray):
	""""
		Computes the euclidean distance between two vectors in n dimensional space.
	"""

	if x1.shape != x2.shape:
		raise ValueError

	diff = x2 - x1

	return np.sqrt(np.dot(np.transpose(diff), diff)[0, 0])


def auto_adjust_axis(axis: QValueAxis, bar_sets: [QBarSet], padding: float = 2):

	# Adjust the range so that everything is visible and add some gaps

	set_count = len(bar_sets)

	minimums = []

	maximums = []

	for i in range(set_count):
		minimums.append(sys.maxsize)
		maximums.append(-sys.maxsize)

	for set_index in range(set_count):

		bar_set = bar_sets[set_index]

		for i in range(bar_set.count()):
			minimums[set_index] = min(minimums[set_index], bar_set.at(i))
			maximums[set_index] = max(maximums[set_index], bar_set.at(i))

	minimums.append(0)
	axis_min = min(minimums) - padding

	maximums.append(0)
	axis_max = max(maximums) + padding

	print("axis min = {}, axis max = {}".format(axis_min, axis_max))

	axis.setMin(axis_min)
	axis.setMax(axis_max)


def is_integer(text: str) -> bool:
	try:
		a = int(text)
		return True
	except ValueError:
		return False


def is_float(text: str) -> bool:
	try:
		f = float(text)
		return True
	except ValueError:
		return False


def extract_eeg_data(raw_data, board_id: int) -> np.ndarray:
	"""
		Given a board id and a 2d array containing retrieved data from board, this method extracts only the eeg data
		into a separate 2d array containing eeg channels as rows and samples as columns.
	"""

	eeg_indexes = BoardShim.get_eeg_channels(board_id)
	eeg_channels = np.empty(shape=(len(eeg_indexes), len(raw_data[eeg_indexes[0]])), dtype=float)

	eeg_channel_index = 0

	for row in range(len(raw_data)):
		if row in eeg_indexes:
			eeg_channels[eeg_channel_index] = raw_data[row]
			eeg_channel_index += 1

	return eeg_channels


def available_com_ports() -> []:
	""""
		Retrieve a list with all open COM ports
	"""

	ports = serial.tools.list_ports.comports()

	return ports


def cyton_port() -> str:

	ports = available_com_ports()

	for port in ports:
		if "DQ007MTCA" == port.serial_number:
			return port.device
	return ""


def vibration_port() -> str:
	ports = available_com_ports()

	for port in ports:
		if "AL01BFIPA" == port.serial_number:
			return port.device
	return ""


def frequency_packet_in_decimal(frequency):
	conv_factor = (2 ** 29 / (10 ** 6))

	decimal = conv_factor * frequency
	return int(decimal + 4 * 16 ** 3)


def stop_vibration(vibration_serial):
	if vibration_serial is not None and vibration_serial.isOpen():
		packet = bytearray()
		packet.append(0x80)
		packet.append(0x51)
		packet.append(0x02)
		packet.append(0x40)
		packet.append(0x00)
		packet.append(0x00)
		packet.append(0xcc)
		vibration_serial.write(packet)


def start_vibration(vibration_serial, left_frequency, right_frequency):
	if vibration_serial is not None and vibration_serial.isOpen():

		print(f"Starting vibration with left freq of {left_frequency} Hz and right freq of {right_frequency} Hz")

		left_freq = frequency_packet_in_decimal(left_frequency)
		right_freq = frequency_packet_in_decimal(right_frequency)

		left_hex = numerical.to_hex(left_freq, 4)
		right_hex = numerical.to_hex(right_freq, 4)

		# Send left command
		packet = bytearray()
		packet.append(0x81)

		packet.append(int(left_hex[0:2], 16))
		packet.append(int(left_hex[2:4], 16))

		packet.append(0x40)
		packet.append(0x00)
		packet.append(0x00)
		packet.append(0xcc)
		vibration_serial.write(packet)

		time.sleep(0.1)

		# Send right command
		packet = bytearray()
		packet.append(0x82)

		packet.append(int(right_hex[0:2], 16))
		packet.append(int(right_hex[2:4], 16))

		packet.append(0x40)
		packet.append(0x00)
		packet.append(0x00)
		packet.append(0xff)
		vibration_serial.write(packet)


def trial_class_as_string(trial_class: TrialClass) -> str:
	return f"{trial_class.name}|{trial_class.image_path}|{trial_class.label}|{trial_class.direction}"


def trial_class_from_string(string: str) -> TrialClass:
	string = string.replace("\n", "")
	items = string.split("|")
	return TrialClass(items[0], items[1], int(items[2]), Direction[items[3].split(".")[1]])
