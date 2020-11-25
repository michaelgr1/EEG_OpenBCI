import random
import sys

import PyQt5.QtCore
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QCheckBox, QGridLayout, QMainWindow, QComboBox, QLabel, QPushButton, \
	QLineEdit, QFileDialog, QHBoxLayout, QPlainTextEdit, QGroupBox, QRadioButton
from brainflow.board_shim import BoardShim, BrainFlowInputParams

import classification
import data
import global_config
import utils
from data import DataSet

AVAILABLE_CLASSIFIERS = [
	classification.LogisticRegressionClassifier.NAME,
	classification.KNearestNeighborsClassifier.NAME,
	classification.PerceptronClassifier.NAME,
	classification.SvmClassifier.NAME,
	classification.LdaClassifier.NAME
]


FFT_WINDOW_SIZES = \
	[
		0,
		pow(2, 7) / global_config.SAMPLING_RATE,
		pow(2, 8) / global_config.SAMPLING_RATE,
		pow(2, 9) / global_config.SAMPLING_RATE,
		pow(2, 10) / global_config.SAMPLING_RATE
	]


class ClassifierTrainer(QMainWindow):

	data_set: DataSet

	def __init__(self):
		super().__init__()
		self.setWindowTitle("Classifier Trainer")

		self.classifier = None
		self.data_set = None
		self.filter_settings = None
		self.feature_extraction_info = None
		self.feature_types = []
		self.trial_classes = []

		self.root_widget = QWidget()
		self.root_layout = QGridLayout()
		self.root_layout.setAlignment(PyQt5.QtCore.Qt.AlignTop | PyQt5.QtCore.Qt.AlignVCenter)
		self.root_widget.setLayout(self.root_layout)
		self.setCentralWidget(self.root_widget)

		# Data which should not get loaded every train. Saved globally to avoid redundancy.
		self.loaded_eeg_data = np.empty(0)

		# Title
		title = QLabel("<h1> Train A Classifier </h1>")
		title.setMargin(20)
		title.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(title, 0, 0, 1, 3)

		# Load Training Data
		load_training_data_label = QLabel("<h2> Load Training Data </h2>")
		load_training_data_label.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(load_training_data_label, 1, 0, 1, 3)

		self.root_layout.addWidget(QLabel("Root Directory: "), 2, 0, 1, 1)

		self.root_directory_label = QLabel("path to directory")
		self.root_layout.addWidget(self.root_directory_label, 2, 1, 1, 1)

		self.select_root_directory = QPushButton("Select/Change")
		self.root_directory_changed = True
		self.select_root_directory.clicked.connect(self.select_root_directory_path)
		self.root_layout.addWidget(self.select_root_directory, 2, 2, 1, 1)

		pre_processing_label = QLabel("<h2> Pre-Process Data </h2>")
		pre_processing_label.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(pre_processing_label, 6, 0, 1, 3)

		self.bandpass_min_edit = QLineEdit()
		self.bandpass_max_edit = QLineEdit()

		self.notch_filter_checkbox = QCheckBox("Notch Filter")
		self.notch_filter_checkbox.setChecked(True)

		self.root_layout.addWidget(utils.construct_horizontal_box(
			[QLabel("Bandpass Filter: "), QLabel("from "), self.bandpass_min_edit, QLabel(" to "),
			 self.bandpass_max_edit,
			 self.notch_filter_checkbox]), 7, 0, 1, 3)

		self.feature_scaling_radio_group = QGroupBox()

		self.selected_feature_scaling_type = data.FeatureScalingType.NO_SCALING

		self.min_max_scaling_radio_btn = QRadioButton("MinMax Scaling")
		self.min_max_scaling_radio_btn.setChecked(True)
		self.min_max_scaling_radio_btn.clicked.connect\
			(lambda: self.set_feature_scaling_type(data.FeatureScalingType.MIN_MAX_SCALING))

		self.standardization_scaling_radio_btn = QRadioButton("Standardization (Z-Score)")
		self.standardization_scaling_radio_btn.clicked.connect\
			(lambda: self.set_feature_scaling_type(data.FeatureScalingType.STANDARDIZATION))

		self.mean_normalization_radio_btn = QRadioButton("Mean Normalization")
		self.mean_normalization_radio_btn.clicked.connect\
			(lambda: self.set_feature_scaling_type(data.FeatureScalingType.MEAN_NORMALIZATION))

		self.unit_length_scaling_radio_btn = QRadioButton("Unit Length")
		self.unit_length_scaling_radio_btn.clicked.connect\
			(lambda: self.set_feature_scaling_type(data.FeatureScalingType.UNIT_LENGTH))

		self.feature_scaling_radio_layout = QHBoxLayout()
		self.feature_scaling_radio_layout.addWidget(self.min_max_scaling_radio_btn)
		self.feature_scaling_radio_layout.addWidget(self.standardization_scaling_radio_btn)
		self.feature_scaling_radio_layout.addWidget(self.mean_normalization_radio_btn)
		self.feature_scaling_radio_layout.addWidget(self.unit_length_scaling_radio_btn)

		self.feature_scaling_radio_group.setLayout(self.feature_scaling_radio_layout)
		self.feature_scaling_radio_group.setCheckable(True)
		self.feature_scaling_radio_group.setChecked(False)
		self.feature_scaling_radio_group.setTitle("Feature Scaling")

		self.root_layout.addWidget(self.feature_scaling_radio_group, 8, 0, 1, 3)

		feature_extraction_label = QLabel("<h2> Extract Features </h2>")
		feature_extraction_label.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(feature_extraction_label, 9, 0, 1, 3)

		self.first_electrode_edit = QLineEdit()
		self.last_electrode_edit = QLineEdit()

		self.root_layout.addWidget(utils.construct_horizontal_box([
			QLabel("Include data from electrode "), self.first_electrode_edit, QLabel(" up to electrode "),
			self.last_electrode_edit]), 10, 0, 1, 3)

		self.band_amplitude_checkbox = QCheckBox("Average Band Amplitude")

		self.band_amplitude_min_edit = QLineEdit()
		self.band_amplitude_max_edit = QLineEdit()

		self.fft_window_combo = QComboBox()

		for window_size in FFT_WINDOW_SIZES:
			self.fft_window_combo.addItem(str(window_size))

		self.k_value_edit = QLineEdit()

		self.accuracy_threshold_edit = QLineEdit()

		self.regularization_edit = QLineEdit()

		self.root_layout.addWidget(utils.construct_horizontal_box([
			QLabel("FFT Window Size:"), self.fft_window_combo, QLabel("K value:"), self.k_value_edit,
			QLabel("Accuracy Threshold (0 - 1):"), self.accuracy_threshold_edit,
			QLabel("Regularization Parameter:"), self.regularization_edit
		]), 11, 0, 1, 3)

		self.root_layout.addWidget(utils.construct_horizontal_box([
			self.band_amplitude_checkbox, QLabel("Frequency band from "), self.band_amplitude_min_edit,
			QLabel(" up to "), self.band_amplitude_max_edit
		]), 12, 0, 1, 3)

		# Extract features as frequency band width and multiple frequency band centers.

		self.frequency_bands_checkbox = QCheckBox("Multiple Frequency Bands")

		self.band_width_edit = QLineEdit()

		self.center_frequencies_edit = QLineEdit()

		self.root_layout.addWidget(utils.construct_horizontal_box([
			self.frequency_bands_checkbox, QLabel("Bandwidth: "), self.band_width_edit,
			QLabel("Center Frequencies (comma separated): "), self.center_frequencies_edit
		]), 13, 0, 1, 3)

		classifier_type_label = QLabel("<p>Classifier Type:</p>")

		self.classifier_type_combo = QComboBox()
		self.classifier_type_combo.addItems(AVAILABLE_CLASSIFIERS)

		self.root_layout.addWidget(classifier_type_label, 14, 0, 1, 1)
		self.root_layout.addWidget(self.classifier_type_combo, 14, 1, 1, 1)

		self.extract_features_btn = QPushButton("Extract Features")
		self.extract_features_btn.clicked.connect(self.extract_features_clicked)
		self.shuffle_data_set = QPushButton("Shuffle DataSet")
		self.shuffle_data_set.clicked.connect(self.shuffle_data_set_clicked)
		self.train_classifier_btn = QPushButton("Train Classifier")
		self.train_classifier_btn.clicked.connect(self.train_classifier_clicked)
		self.test_classifier_btn = QPushButton("Test Classifier")
		self.test_classifier_btn.clicked.connect(self.test_classifier_clicked)

		self.root_layout.addWidget(utils.construct_horizontal_box([
			self.extract_features_btn, self.shuffle_data_set, self.train_classifier_btn, self.test_classifier_btn
		]), 15, 0, 1, 3)

		self.performance_report_btn = QPushButton("Performance Report")
		self.performance_report_btn.clicked.connect(self.generate_performance_report)

		self.root_layout.addWidget(utils.construct_horizontal_box([
			self.performance_report_btn
		]), 16, 0, 1, 3)

	def set_feature_scaling_type(self, feature_scaling_type: data.FeatureScalingType):
		self.selected_feature_scaling_type = feature_scaling_type
		print("Setting selected feature scaling type to {}".format(feature_scaling_type))

	def extract_features_clicked(self):
		# Construct filter settings to loaded data.

		bandpass_min = -1
		bandpass_max = -1

		notch_filter = self.notch_filter_checkbox.isChecked()

		if utils.is_float(self.bandpass_min_edit.text()):
			bandpass_min = float(self.bandpass_min_edit.text())

		if utils.is_float(self.bandpass_max_edit.text()):
			bandpass_max = float(self.bandpass_max_edit.text())

		filter_settings = utils.FilterSettings(global_config.SAMPLING_RATE, bandpass_min, bandpass_max, notch_filter)

		if self.root_directory_changed:
			self.loaded_eeg_data = utils.load_data(self.root_directory_label.text())

		eeg_data, classes, sampling_rate, self.trial_classes = \
			utils.slice_and_filter_data(self.root_directory_label.text(), filter_settings, self.loaded_eeg_data)

		labels = np.array(classes).reshape((-1, 1))

		if len(eeg_data) != 0 and len(classes) != 0:
			print("Data loaded successfully")
		else:
			print("Could not load data")
			return

		# Construct feature descriptors.

		# Obtain the range of channels to be included
		first_electrode = -1
		last_electrode = -1

		if utils.is_integer(self.first_electrode_edit.text()):
			first_electrode = int(self.first_electrode_edit.text())

		if utils.is_integer(self.last_electrode_edit.text()):
			last_electrode = int(self.last_electrode_edit.text())

		fft_window_size = float(self.fft_window_combo.currentText())

		feature_types = []

		if self.band_amplitude_checkbox.isChecked():
			band_amplitude_min_freq = -1
			band_amplitude_max_freq = -1

			if utils.is_float(self.band_amplitude_min_edit.text()):
				band_amplitude_min_freq = float(self.band_amplitude_min_edit.text())

			if utils.is_float(self.band_amplitude_max_edit.text()):
				band_amplitude_max_freq = float(self.band_amplitude_max_edit.text())

			if band_amplitude_min_freq != -1 and band_amplitude_max_freq != -1:
				feature_types.append(
					utils.AverageBandAmplitudeFeature(
						utils.FrequencyBand(band_amplitude_min_freq, band_amplitude_max_freq), fft_window_size))

		if self.frequency_bands_checkbox.isChecked():
			band_width = -1

			if utils.is_float(self.band_width_edit.text()):
				band_width = float(self.band_width_edit.text())

			center_frequencies_str_list = self.center_frequencies_edit.text().split(",")

			center_frequencies = []

			for center_freq_str in center_frequencies_str_list:
				if utils.is_float(center_freq_str):
					center_frequencies.append(float(center_freq_str))

			if len(center_frequencies) != 0 and band_width != -1:
				feature_types.append(
					utils.FrequencyBandsAmplitudeFeature(center_frequencies, band_width, fft_window_size))

		feature_extraction_info = utils.FeatureExtractionInfo(sampling_rate, first_electrode, last_electrode)

		self.filter_settings = filter_settings
		self.feature_extraction_info = feature_extraction_info
		self.feature_types = feature_types

		# Extract features

		extracted_data = utils.extract_features(
			eeg_data, feature_extraction_info, feature_types)

		feature_matrix = data.construct_feature_matrix(extracted_data)

		self.data_set = DataSet(feature_matrix, labels, add_x0=False, shuffle=True)

		print("Features extracted successfully...")

	def shuffle_data_set_clicked(self):
		if self.data_set is not None:
			self.data_set = \
				DataSet(self.data_set.raw_feature_matrix(), self.data_set.feature_matrix_labels(), False, True)
			print("Data set shuffled!!!")

	def train_classifier_clicked(self):
		print("train classifier clicked")

		if self.data_set is None:
			self.extract_features_clicked()
		else:
			print("*"*10 + "Using existing feature matrix, re-extract if changes were made" + "*"*10)

		accuracy_threshold = self.get_accuracy_threshold()
		regularization_param = self.get_regularization_param()

		selected_classifier = self.classifier_type_combo.currentText()

		feature_matrix = self.data_set.raw_feature_matrix()
		labels = self.data_set.feature_matrix_labels()

		# Train Classifier
		if selected_classifier == classification.LogisticRegressionClassifier.NAME:
			classifier = classification.LogisticRegressionClassifier(feature_matrix, labels, shuffle=False)
			classifier.data_set.apply_feature_scaling(self.selected_feature_scaling_type)
			self.classifier = classifier
			cost = classifier.train(accuracy_threshold=accuracy_threshold)

			plt.plot(cost)
			plt.xlabel("Iteration Number")
			plt.ylabel("Training Set Cost")
			plt.title("Gradient Descent Cost Curve")
			plt.show()

			print("Training set accuracy = {}".format(classifier.training_set_accuracy()))
			print("Logistic Regression trained successfully, test set accuracy = {}".format(classifier.test_set_accuracy()))
			print("Cross validation accuracy = {}".format(classifier.test_set_accuracy()))
		elif selected_classifier == classification.KNearestNeighborsClassifier.NAME:

			k_value = self.get_k_value()

			print("Using K value of {}".format(k_value))

			classifier = classification.KNearestNeighborsClassifier(feature_matrix, labels, k_value, shuffle=False)
			classifier.data_set.apply_feature_scaling(self.selected_feature_scaling_type)
			self.classifier = classifier

			print("KNN training set accuracy = {}".format(classifier.training_set_accuracy()))
			print("KNN test set accuracy = {}".format(classifier.test_set_accuracy()))
			print("KNN cross validation accuracy = {}".format(classifier.cross_validation_accuracy()))

			k_values, accuracy = classifier.cross_validation_learning_curve()
			plt.plot(k_values, accuracy)
			plt.xlabel("K Neighbors")
			plt.ylabel("Accuracy")
			plt.title("Accuracy graph for K values")
			plt.show()
		elif selected_classifier == classification.PerceptronClassifier.NAME:
			classifier = classification.PerceptronClassifier(feature_matrix, labels, shuffle=False)
			classifier.data_set.apply_feature_scaling(self.selected_feature_scaling_type)
			self.classifier = classifier
			classifier.train(accuracy_threshold=accuracy_threshold)

			print("Perceptron training set accuracy = {}".format(classifier.training_set_accuracy()))
			print("Perceptron test set accuracy = {}".format(classifier.test_set_accuracy()))
			print("Perceptron Cross validation accuracy = {}".format(classifier.cross_validation_accuracy()))
		elif selected_classifier == classification.SvmClassifier.NAME:
			classifier = classification.SvmClassifier(feature_matrix, labels, regularization_param, shuffle=False)
			classifier.data_set.apply_feature_scaling(self.selected_feature_scaling_type)
			self.classifier = classifier
			classifier.train()

			print("SVM training set accuracy = {}".format(classifier.training_set_accuracy()))
			print("SVM test set accuracy = {}".format(classifier.test_set_accuracy()))
			print("SVM cross validation accuracy = {}".format(classifier.cross_validation_accuracy()))
		elif selected_classifier == classification.LdaClassifier.NAME:
			classifier = classification.LdaClassifier(feature_matrix, labels, shuffle=False)
			classifier.data_set.apply_feature_scaling(self.selected_feature_scaling_type)
			self.classifier = classifier
			classifier.train()

			print("LDA training set accuracy = {}".format(classifier.training_set_accuracy()))
			print("LDA test set accuracy = {}".format(classifier.test_set_accuracy()))
			print("LDA cross validation accuracy = {}".format(classifier.cross_validation_accuracy()))

			# TODO: Learning curves

		self.root_directory_changed = False

	def generate_performance_report(self):
		feature_matrix = self.data_set.raw_feature_matrix()
		labels = self.data_set.feature_matrix_labels()

		# Logistic Regression
		cls1 = classification.LogisticRegressionClassifier(feature_matrix, labels, shuffle=False)
		cls1.data_set.apply_feature_scaling(self.selected_feature_scaling_type)
		cls1.train(accuracy_threshold=self.get_accuracy_threshold())

		# KNN
		k_value = self.get_k_value()
		print("Using K value of {}".format(k_value))

		cls2 = classification.KNearestNeighborsClassifier(feature_matrix, labels, k_value, shuffle=False)
		cls2.data_set.apply_feature_scaling(self.selected_feature_scaling_type)

		# Perceptron
		cls3 = classification.PerceptronClassifier(feature_matrix, labels, shuffle=False)
		cls3.data_set.apply_feature_scaling(self.selected_feature_scaling_type)
		cls3.train(accuracy_threshold=self.get_accuracy_threshold())

		# SVM
		cls4 = classification.SvmClassifier(feature_matrix, labels, self.get_regularization_param(), shuffle=False)
		cls4.data_set.apply_feature_scaling(self.selected_feature_scaling_type)
		cls4.train()

		# LDA
		cls5 = classification.LdaClassifier(feature_matrix, labels, shuffle=False)
		cls5.data_set.apply_feature_scaling(self.selected_feature_scaling_type)
		cls5.train()

		# Get performance measure from each
		prm1 = cls1.performance_measure()
		prm2 = cls2.performance_measure()
		prm3 = cls3.performance_measure()
		prm4 = cls4.performance_measure()
		prm5 = cls5.performance_measure()

		plot_data = np.vstack((
			prm1.as_row_array(),
			prm2.as_row_array(),
			prm3.as_row_array(),
			prm4.as_row_array(),
			prm5.as_row_array()
		))

		plot_data = np.transpose(plot_data)

		x = np.arange(5)

		plt.bar(x + 0.0, plot_data[0], width=0.25, label="Training Accuracy")
		plt.bar(x + 0.25, plot_data[1], width=0.25, label="Cross Validation Accuracy")
		plt.bar(x + 0.5, plot_data[2], width=0.25, label="Testing Accuracy")

		plt.xticks(x + 0.125, ("Logistic Regression", "kNN", "Perceptron", "SVM", "LDA"))
		plt.ylabel("Accuracy %")

		plt.legend(loc="best")
		plt.title("Classifiers' Accuracy")

		plt.show()

	def generate_error_descriptions(self):
		# TODO: Implement
		pass

	def get_k_value(self) -> int:
		k_value = 5
		if utils.is_integer(self.k_value_edit.text()):
			k_value = int(self.k_value_edit.text())
			if k_value <= 0:
				k_value = 1
		return k_value

	def get_accuracy_threshold(self) -> float:
		accuracy_threshold = 1
		if utils.is_float(self.accuracy_threshold_edit.text()):
			accuracy_threshold = float(self.accuracy_threshold_edit.text())
		return accuracy_threshold

	def get_regularization_param(self) -> float:
		c = 1.0
		if utils.is_float(self.regularization_edit.text()):
			c = float(self.regularization_edit.text())
		return c

	def test_classifier_clicked(self):
		if self.classifier is not None and self.filter_settings is not None and self.feature_extraction_info is not None\
				and len(self.feature_types) != 0 and len(self.trial_classes) != 0:
			trial_length = utils.obtain_trial_length_from_slice_index(self.root_directory_label.text())

			online_config = OnlineClassifierConfigurations()
			online_config.feature_window_size = trial_length

			OnlineClassifierGui(self.classifier,
								self.filter_settings,
								self.feature_extraction_info,
								self.feature_types,
								self.trial_classes, self, online_config)
		else:
			print("Train Classifier Before Testing!")

	@staticmethod
	def validate_keywords(keywords: [str]) -> bool:
		for keyword in keywords:
			if len(keyword) == 0:
				return False
		return True

	def select_root_directory_path(self):
		path = QFileDialog.getExistingDirectory(self, "Select Root Directory...")
		self.root_directory_label.setText(path)
		self.root_directory_changed = True


class OnlineClassifierConfigurations:

	def __init__(self, feature_window_size: float = 5, repetition_interval: float = 0.2, detection_threshold: int = 1):
		self.feature_window_size = feature_window_size
		self.repetition_interval = repetition_interval
		self.detection_threshold = detection_threshold


class TrainingTaskTile(QWidget):

	def __init__(self, trial_class: utils.TrialClass, image_height: float):
		super().__init__()
		self.highlighted = False

		self.trial_class = trial_class

		# self.setMaximumSize(200, 200)

		self.root_layout = QGridLayout()
		self.root_layout.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.setLayout(self.root_layout)

		self.class_image_label = QLabel()
		self.class_image_pixmap = QPixmap(self.trial_class.image_path).scaledToHeight(int(image_height), Qt.FastTransformation)
		self.class_image_label.setPixmap(self.class_image_pixmap)

		self.root_layout.addWidget(self.class_image_label, 0, 0, 3, 3)

	def mouseDoubleClickEvent(self, a0: QtGui.QMouseEvent) -> None:
		if self.highlighted:
			self.highlight()
		else:
			self.disable_highlight()

	def highlight(self):
		self.highlighted = True
		self.setStyleSheet("border: 5px solid blue;")

	def disable_highlight(self):
		self.highlighted = False
		self.setStyleSheet("border: none;")


class OnlineClassifierGui(QMainWindow):
	"""
		A GUI for testing and training classifiers online.
		This class can be used alone, but is really meant to be the stage after training which
		is done by the Classifier Trainer class.
	"""

	INTERNAL_BUFFER_EXTRA_DURATION = 2

	INTERNAL_BUFFER_EXTRA_SIZE = 0

	CLASS_IMAGE_HEIGHT = 400

	MENTAL_TASK_DELAY = 4000

	def __init__(self, classifier, filter_settings: utils.FilterSettings,
				feature_extraction_info: utils.FeatureExtractionInfo,
				feature_types: [],
				trial_classes: [utils.TrialClass],
				parent=None,
				config: OnlineClassifierConfigurations = OnlineClassifierConfigurations()):
		super().__init__(parent)
		self.setWindowTitle("Online Classifier")
		self.setWindowModality(PyQt5.QtCore.Qt.ApplicationModal)

		self.classifier = classifier
		self.filter_settings = filter_settings
		self.feature_extraction_info = feature_extraction_info
		self.feature_types = feature_types
		self.trial_classes = trial_classes
		self.config = config

		self.INTERNAL_BUFFER_EXTRA_SIZE = self.INTERNAL_BUFFER_EXTRA_DURATION * self.feature_extraction_info.sampling_rate

		self.data_buffer = np.array([])

		self.root_layout = QGridLayout()
		self.root_layout.setAlignment(PyQt5.QtCore.Qt.AlignTop)
		self.root_widget = QWidget()
		self.root_widget.setLayout(self.root_layout)
		self.setCentralWidget(self.root_widget)

		title = QLabel("<h1>Online Classifier</h1>")
		title.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(title, 0, 0, 1, 3)

		self.feature_window_edit = QLineEdit()
		self.feature_window_edit.setText(str(self.config.feature_window_size))

		self.repetition_interval_edit = QLineEdit()
		self.repetition_interval_edit.setText(str(self.config.repetition_interval))

		self.detection_threshold_edit = QLineEdit()
		self.detection_threshold_edit.setText(str(self.config.detection_threshold))

		self.online_training_checkbox = QCheckBox("Online Training")
		self.online_training = False

		self.previous_test_set_accuracy = 0.0
		self.previous_cross_validation_accuracy = 0.0

		self.root_layout.addWidget(utils.construct_horizontal_box([
			QLabel("Feature Window Size (sec): "), self.feature_window_edit,
			QLabel("Repetition Interval (sec): "), self.repetition_interval_edit,
			QLabel("Detection Threshold: "), self.detection_threshold_edit,
			self.online_training_checkbox
		]), 1, 0, 1, 3)

		self.start_btn = QPushButton("Start Streaming")
		self.start_btn.clicked.connect(self.start_clicked)
		self.stop_btn = QPushButton("Stop Streaming")
		self.stop_btn.clicked.connect(self.stop_clicked)

		self.root_layout.addWidget(utils.construct_horizontal_box([
			self.stop_btn, self.start_btn
		]), 2, 0, 1, 3)

		self.class_label = QLabel("Start stream to see result")
		self.class_label.setAlignment(PyQt5.QtCore.Qt.AlignCenter)

		self.root_layout.addWidget(self.class_label, 3, 0, 1, 3)

		self.root_layout.addWidget(utils.construct_horizontal_box([
			QLabel("<h2>Mental Tasks:</h2>")
		]), 4, 0, 1, 3)

		self.training_tasks_widget = QWidget()
		self.training_tasks_layout = QHBoxLayout()
		self.training_tasks_widget.setLayout(self.training_tasks_layout)

		self.class_tiles_list = []

		for trial_class in trial_classes:
			self.class_tiles_list.append(TrainingTaskTile(trial_class, self.CLASS_IMAGE_HEIGHT / 2))
			self.training_tasks_layout.addWidget(self.class_tiles_list[-1])

		self.root_layout.addWidget(self.training_tasks_widget, 5, 0, 1, 3)

		self.class_pixmap = QPixmap()

		self.log_textarea = QPlainTextEdit()

		self.root_layout.addWidget(self.log_textarea, 6, 0, 1, 3)

		self.board = self.initialize_board()
		self.board.prepare_session()

		self.reading_timer = None
		self.samples_push_count = 0
		self.online_training_samples_push_count = 0
		self.online_training_timer = None
		self.current_mental_task = None

		self.show()

	@staticmethod
	def initialize_board() -> BoardShim:
		params = BrainFlowInputParams()
		params.serial_port = utils.cyton_port()

		board = BoardShim(global_config.BOARD_ID, params)
		return board

	def initialize_data_buffer(self):
		self.data_buffer = np.zeros((
				int(self.feature_extraction_info.last_channel - self.feature_extraction_info.first_channel + 1),
				int(self.INTERNAL_BUFFER_EXTRA_SIZE + self.config.feature_window_size * self.feature_extraction_info.sampling_rate)
		), dtype=float)

	def start_clicked(self):

		if utils.is_float(self.feature_window_edit.text()):
			self.config.feature_window_size = float(self.feature_window_edit.text())

		if utils.is_float(self.repetition_interval_edit.text()):
			self.config.repetition_interval = float(self.repetition_interval_edit.text())

		if utils.is_float(self.detection_threshold_edit.text()):
			self.config.detection_threshold = float(self.detection_threshold_edit.text())

		self.online_training = self.online_training_checkbox.isChecked()

		self.log(f"Starting data stream, online training? {self.online_training}")

		# if self.online_training:
		# 	if type(self.classifier) == classification.LogisticRegressionClassifier:
		# 		self.previous_cross_validation_accuracy = self.classifier.cross_validation_accuracy()
		# 		self.previous_test_set_accuracy = self.classifier.test_set_accuracy()
		# 	elif type(self.classifier) == classification.KNearestNeighborsClassifier:
		# 		self.previous_cross_validation_accuracy = self.classifier.cross_validation_accuracy()
		# 		self.previous_test_set_accuracy = self.classifier.test_set_accuracy()
		# 	elif type(self.classifier) == classification.PerceptronClassifier:
		# 		self.previous_cross_validation_accuracy = self.classifier.cross_validation_accuracy()
		# 		self.previous_test_set_accuracy = self.classifier.test_set_accuracy()
		# 	self.online_training_timer = QTimer()
		# 	self.online_training_timer.singleShot(self.MENTAL_TASK_DELAY, self.next_mental_task)

		self.initialize_data_buffer()

		self.samples_push_count = 0
		self.online_training_samples_push_count = 0

		if self.reading_timer is None:
			print("Starting data stream...")
			self.board.start_stream()
			self.reading_timer = QTimer()
			self.reading_timer.timeout.connect(self.read_data)
			# TODO: Replace with a config variable
			self.reading_timer.start(100)

	def stop_clicked(self):
		if self.reading_timer is not None:
			self.log("Stopping stream...")
			print("Stopping...")
			self.board.stop_stream()
			self.reading_timer.deleteLater()
			self.reading_timer = None
			# if self.online_training:
			# 	if self.online_training_timer is not None:
			# 		self.online_training_timer.deleteLater()
			# 		self.online_training_timer = None
			#
			# 	self.log("Retraining classifier...")
			#
			# 	if type(self.classifier) == classification.LogisticRegressionClassifier:
			# 		self.classifier.train()
			# 	elif type(self.classifier) == classification.KNearestNeighborsClassifier:
			# 		self.log("No training for KNN...")
			# 	elif type(self.classifier) == classification.PerceptronClassifier:
			# 		self.classifier.train()
			#
			# 	self.log("Training over...")
			#
			# 	test_set_accuracy = -1
			# 	cross_validation_set_accuracy = -1
			#
			# 	if type(self.classifier) == classification.LogisticRegressionClassifier:
			# 		cross_validation_set_accuracy = self.classifier.cross_validation_accuracy()
			# 		test_set_accuracy = self.classifier.test_set_accuracy()
			# 	elif type(self.classifier) == classification.KNearestNeighborsClassifier:
			# 		cross_validation_set_accuracy = self.classifier.cross_validation_accuracy()
			# 		test_set_accuracy = self.classifier.test_set_accuracy()
			# 	elif type(self.classifier) == classification.PerceptronClassifier:
			# 		cross_validation_set_accuracy = self.classifier.cross_validation_accuracy()
			# 		test_set_accuracy = self.classifier.test_set_accuracy()
			#
			# 	self.log(f"Cross validation accuracy: {100 * cross_validation_set_accuracy}%")
			# 	self.log(f"Test set accuracy: {100 * test_set_accuracy}%")
			#
			# 	self.log(f"Cross validation accuracy change: {self.previous_cross_validation_accuracy - cross_validation_set_accuracy}")
			# 	self.log(f"Test set accuracy change: {self.previous_test_set_accuracy - test_set_accuracy}")

	def read_data(self):
		if self.board.get_board_data_count() > 0:
			raw_data = self.board.get_board_data()
			raw_eeg_data = utils.extract_eeg_data(raw_data, global_config.BOARD_ID)

			# Make room for new samples, discard the oldest
			self.data_buffer = np.roll(self.data_buffer, shift=-raw_eeg_data.shape[1], axis=1)

			# Insert new samples
			first_channel = self.feature_extraction_info.first_channel - 1
			last_channel = self.feature_extraction_info.last_channel
			self.data_buffer[:, self.data_buffer.shape[1] - raw_eeg_data.shape[1]:] = raw_eeg_data[first_channel:last_channel, :]

			self.samples_push_count += raw_eeg_data.shape[1]
			if self.online_training and self.online_training_timer is None:
				self.online_training_samples_push_count += raw_eeg_data.shape[1]

			if self.samples_push_count >= self.config.repetition_interval * self.feature_extraction_info.sampling_rate:
				self.classify_data()
				self.samples_push_count = 0

			if self.online_training_samples_push_count >= self.config.feature_window_size * self.feature_extraction_info.sampling_rate:
				self.classify_data(online_training=True)
				self.online_training_samples_push_count = 0
				self.online_training_timer = QTimer()
				self.online_training_timer.singleShot(self.MENTAL_TASK_DELAY, self.next_mental_task)

				for tile in self.class_tiles_list:
					if tile.highlighted:
						tile.disable_highlight()
						break

	def classify_data(self, online_training: bool = False):
		filtered_data = self.filter_settings.apply(self.data_buffer[:, self.INTERNAL_BUFFER_EXTRA_SIZE:])

		feature_vector = \
			utils.extract_features([utils.EegData(filtered_data)], self.feature_extraction_info, self.feature_types)[0]

		feature_data = feature_vector.data

		print("Feature Vector extracted successfully...")

		label = -sys.maxsize

		if type(self.classifier) == classification.LogisticRegressionClassifier:
			label = self.classifier.classify(feature_data)
			self.class_label.setText(f"Current data is classified as {label} using Logistic Regression")
			self.log(f"Data classified as {label}, confidence = {self.classifier.confidence(feature_data)}")

		if type(self.classifier) == classification.KNearestNeighborsClassifier:
			label = self.classifier.classify(feature_data)
			self.class_label.setText(f"Current data is classified as {label} using KNN")
			self.log(f"Data classified as {label}, confidence = {self.classifier.confidence(np.transpose(feature_vector.data))}")

		if type(self.classifier) == classification.PerceptronClassifier:
			label = self.classifier.classify(feature_data)
			self.class_label.setText(f"Current data is classified as {label} using the Perceptron")
			self.log(f"Data classified as {label}, confidence = {self.classifier.confidence(feature_vector.data)}")

		if type(self.classifier) == classification.SvmClassifier:
			label = self.classifier.classify(feature_data)
			self.class_label.setText(f"Current data is classified as {label} using Linear SVM")
			self.log(f"Data classified as {label}")

		if type(self.classifier) == classification.LdaClassifier:
			label = self.classifier.classify(feature_data)
			self.class_label.setText(f"Current data is classified as {label} using LDA")
			self.log(f"Data classified as {label}")

		print(f"label = {label}")

		if label != -sys.maxsize:

			# if online_training:
			# 	correct_label = self.current_mental_task.label
			#
			# 	if label != correct_label:
			# 		self.log("Wrong classification!")
			# 	else:
			# 		self.log("Correct Classification")
			#
			# 	# Add the sample to one of the classifier's data sets
			# 	chance = random.random()
			#
			# 	# TODO: Implement online training
			# 	self.log("Not Implemented")

			path = ""
			for trial_class in self.trial_classes:
				if trial_class.label == label:
					path = trial_class.image_path
					break
			self.class_pixmap = QPixmap(path).scaledToHeight(self.CLASS_IMAGE_HEIGHT, Qt.FastTransformation)
			self.class_label.setPixmap(self.class_pixmap)

	def next_mental_task(self):
		self.current_mental_task = self.random_class()

		for tile in self.class_tiles_list:
			if tile.trial_class == self.current_mental_task:
				tile.highlight()
			else:
				tile.disable_highlight()

		if self.online_training_timer is not None:
			self.online_training_timer.deleteLater()
			self.online_training_timer = None

	def random_class(self) -> utils.TrialClass:
		index = random.randint(0, len(self.trial_classes) - 1)
		return self.trial_classes[index]

	def log(self, text: str):
		self.log_textarea.insertPlainText(f"{text}\n")

	def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
		self.board.release_session()


def main():
	app = QApplication([])
	app.setStyle(global_config.APP_STYLE)

	window = ClassifierTrainer()
	window.show()

	app.exec()


if __name__ == "__main__":
	main()
