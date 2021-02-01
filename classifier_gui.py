import random

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
from sklearn.decomposition import PCA

import classification as cls
import data
import ev3
import global_config
import utils
from data import DataSet

AVAILABLE_CLASSIFIERS = [
	cls.LogisticRegressionClassifier.NAME,
	cls.KNearestNeighborsClassifier.NAME,
	cls.PerceptronClassifier.NAME,
	cls.SvmClassifier.NAME,
	cls.LdaClassifier.NAME,
	cls.ANNClassifier.NAME,
	cls.VotingClassifier.NAME
]

DEFAULT_VOTING_CLASSIFIERS = [
	cls.LogisticRegressionClassifier.NAME,
	cls.SvmClassifier.NAME,
	cls.LdaClassifier.NAME
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

	DEFAULT_K_FOLD_REPETITIONS = 1000

	data_set: DataSet
	classifier: cls.SimpleClassifier

	def __init__(self):
		super().__init__()
		self.setWindowTitle("Classifier Trainer")

		self.classifier = None
		self.data_set = None
		self.filter_settings = None
		self.feature_extraction_info = None
		self.feature_types = []
		self.trial_classes = []

		self.root_directories = []
		self.root_directories.append("F:\\EEG_GUI_OpenBCI\\eeg_recordings\\vibro_tactile_27_12_2020\\trial_02")

		self.root_widget = QWidget()
		self.root_layout = QGridLayout()
		self.root_layout.setAlignment(PyQt5.QtCore.Qt.AlignTop | PyQt5.QtCore.Qt.AlignVCenter)
		self.root_widget.setLayout(self.root_layout)
		self.setCentralWidget(self.root_widget)

		# Data which should not get loaded every train. Saved globally to avoid redundancy.
		self.loaded_eeg_data = []

		# Title
		title = QLabel("<h1> Train A Classifier </h1>")
		title.setMargin(20)
		title.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(title, 0, 0, 1, 3)

		# Load Training Data
		load_training_data_label = QLabel("<h2> Load Training Data </h2>")
		load_training_data_label.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(load_training_data_label, 1, 0, 1, 3)

		self.root_directory_label = QLabel("path to directories")

		self.add_root_directory = QPushButton("Add path")
		self.pop_root_directory = QPushButton("Pop path")
		self.root_directory_changed = True
		self.add_root_directory.clicked.connect(self.add_root_directory_clicked)
		self.pop_root_directory.clicked.connect(self.pop_root_directory_clicked)
		self.root_layout.addWidget(utils.construct_horizontal_box([
			QLabel("Root Directories: "), self.root_directory_label, self.add_root_directory, self.pop_root_directory
		]), 2, 0, 2, 3)

		pre_processing_label = QLabel("<h2> Pre-Process Data </h2>")
		pre_processing_label.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(pre_processing_label, 6, 0, 1, 3)

		self.bandpass_min_edit = QLineEdit("15")
		self.bandpass_max_edit = QLineEdit("30")

		self.notch_filter_checkbox = QCheckBox("Notch Filter")
		self.notch_filter_checkbox.setChecked(True)

		self.root_layout.addWidget(utils.construct_horizontal_box(
			[QLabel("Bandpass Filter: "), QLabel("from "), self.bandpass_min_edit, QLabel(" to "),
			 self.bandpass_max_edit,
			 self.notch_filter_checkbox]), 7, 0, 1, 3)

		self.adaptive_filtering_checkbox = QCheckBox("Adaptive Filtering")
		self.adaptive_reference_electrode = QLineEdit()
		self.adaptive_frequencies_edit = QLineEdit()
		self.adaptive_bandwidths_edit = QLineEdit()

		self.root_layout.addWidget(utils.construct_horizontal_box(
			[
				self.adaptive_filtering_checkbox,
				QLabel("Reference Electrode: "), self.adaptive_reference_electrode,
				QLabel("Frequencies (comma separated):"), self.adaptive_frequencies_edit,
				QLabel("Bandwidths (comma separated):"), self.adaptive_bandwidths_edit
			 ],
		), 8, 0, 1, 3)

		self.re_reference_checkbox = QCheckBox("Re-Reference data")
		self.reference_electrode_edit = QLineEdit("3")

		self.root_layout.addWidget(utils.construct_horizontal_box([
			self.re_reference_checkbox, QLabel("New reference electrode: "), self.reference_electrode_edit
		]), 9, 0, 1, 3)

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

		self.root_layout.addWidget(self.feature_scaling_radio_group, 10, 0, 1, 3)

		feature_extraction_label = QLabel("<h2> Extract Features </h2>")
		feature_extraction_label.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(feature_extraction_label, 11, 0, 1, 3)

		self.electrodes_edit = QLineEdit("5,4,2,1")

		self.root_layout.addWidget(utils.construct_horizontal_box([
			QLabel("Include data from electrodes (comma separated):"), self.electrodes_edit]), 12, 0, 1, 2)

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
		]), 13, 0, 1, 3)

		self.root_layout.addWidget(utils.construct_horizontal_box([
			self.band_amplitude_checkbox, QLabel("Frequency band from "), self.band_amplitude_min_edit,
			QLabel(" up to "), self.band_amplitude_max_edit
		]), 14, 0, 1, 3)

		# Extract features as frequency band width and multiple frequency band centers.

		self.frequency_bands_checkbox = QCheckBox("Multiple Frequency Bands")

		self.band_width_edit = QLineEdit("1")

		self.center_frequencies_edit = QLineEdit("20,24")

		self.peak_frequency_checkbox = QCheckBox("Peak Frequency")

		self.root_layout.addWidget(utils.construct_horizontal_box([
			self.frequency_bands_checkbox, QLabel("Bandwidth: "), self.band_width_edit,
			QLabel("Center Frequencies (comma separated): "), self.center_frequencies_edit,
			self.peak_frequency_checkbox
		]), 15, 0, 1, 3)

		classifier_type_label = QLabel("<p>Classifier Type:</p>")

		self.classifier_type_combo = QComboBox()
		self.classifier_type_combo.addItems(AVAILABLE_CLASSIFIERS)

		self.root_layout.addWidget(utils.construct_horizontal_box([
			classifier_type_label, self.classifier_type_combo
		]), 16, 0, 1, 3)

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
		]), 17, 0, 1, 3)

		self.performance_report_btn = QPushButton("Performance Report")
		self.performance_report_btn.clicked.connect(self.generate_performance_report)

		self.error_description_btn = QPushButton("Error Description")
		self.error_description_btn.clicked.connect(self.generate_error_descriptions)

		self.visualize_data_btn = QPushButton("Visualize Data")
		self.visualize_data_btn.clicked.connect(self.visualize_data)

		self.k_fold_edit = QLineEdit()
		self.k_fold_btn = QPushButton("k fold cross validation")
		self.k_fold_btn.clicked.connect(self.k_fold_cross_validation_clicked)

		self.repeated_k_fold_btn = QPushButton("repeated k fold")
		self.repeated_k_fold_btn.clicked.connect(self.repeated_k_fold_clicked)

		self.root_layout.addWidget(utils.construct_horizontal_box([
			self.performance_report_btn, self.error_description_btn, self.visualize_data_btn,
			self.k_fold_edit, self.k_fold_btn, self.repeated_k_fold_btn
		]), 18, 0, 1, 3)

		self.update_root_directories_label()

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

		adaptive_settings = None

		if self.adaptive_filtering_checkbox.isChecked():
			reference_electrode = int(self.adaptive_reference_electrode.text())
			frequencies = []
			widths = []

			for freq_str in self.adaptive_frequencies_edit.text().split(","):
				frequencies.append(float(freq_str))

			for width_str in self.adaptive_bandwidths_edit.text().split(","):
				widths.append(float(width_str))

			adaptive_settings = utils.AdaptiveFilterSettings(reference_electrode, frequencies, widths)

		reference_electrode = 0

		if self.re_reference_checkbox.isChecked() and utils.is_integer(self.reference_electrode_edit.text()):
			reference_electrode = int(self.reference_electrode_edit.text())

		filter_settings = utils.FilterSettings(global_config.SAMPLING_RATE, bandpass_min, bandpass_max, notch_filter=notch_filter,
											   adaptive_filter_settings=adaptive_settings, reference_electrode=reference_electrode)

		if self.root_directory_changed:
			self.loaded_eeg_data = utils.load_data(self.root_directories)

		eeg_data, classes, sampling_rate, self.trial_classes = \
			utils.slice_and_filter_data(self.root_directories, filter_settings, self.loaded_eeg_data)

		labels = np.array(classes).reshape((-1, 1))

		if len(eeg_data) != 0 and len(classes) != 0:
			print("Data loaded successfully")
		else:
			print("Could not load data")
			return

		# Construct feature descriptors.

		# Obtain the range of channels to be included
		electrode_list = []

		for electrode_str in self.electrodes_edit.text().split(","):
			electrode_list.append(int(electrode_str))

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

			peak_frequency = self.peak_frequency_checkbox.isChecked()

			if utils.is_float(self.band_width_edit.text()):
				band_width = float(self.band_width_edit.text())

			center_frequencies_str_list = self.center_frequencies_edit.text().split(",")

			center_frequencies = []

			for center_freq_str in center_frequencies_str_list:
				if utils.is_float(center_freq_str):
					center_frequencies.append(float(center_freq_str))

			if len(center_frequencies) != 0 and band_width != -1:
				feature_types.append(
					utils.FrequencyBandsAmplitudeFeature(center_frequencies, band_width, fft_window_size, peak_frequency))

		feature_extraction_info = utils.FeatureExtractionInfo(sampling_rate, electrode_list)

		self.filter_settings = filter_settings
		self.feature_extraction_info = feature_extraction_info
		self.feature_types = feature_types

		# Extract features

		extracted_data = utils.extract_features(
			eeg_data, feature_extraction_info, feature_types)

		feature_matrix = data.construct_feature_matrix(extracted_data)

		self.data_set = DataSet(feature_matrix, labels, add_x0=False, shuffle=False)

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
		if selected_classifier == cls.LogisticRegressionClassifier.NAME:
			classifier = cls.LogisticRegressionClassifier(feature_matrix, labels, shuffle=False)
			classifier.apply_feature_scaling(self.selected_feature_scaling_type)
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
		elif selected_classifier == cls.KNearestNeighborsClassifier.NAME:

			k_value = self.get_k_value()

			print("Using K value of {}".format(k_value))

			classifier = cls.KNearestNeighborsClassifier(feature_matrix, labels, k_value, shuffle=False)
			classifier.apply_feature_scaling(self.selected_feature_scaling_type)
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
		elif selected_classifier == cls.PerceptronClassifier.NAME:
			classifier = cls.PerceptronClassifier(feature_matrix, labels, shuffle=False)
			classifier.apply_feature_scaling(self.selected_feature_scaling_type)
			self.classifier = classifier
			classifier.train(accuracy_threshold=accuracy_threshold)

			print("Perceptron training set accuracy = {}".format(classifier.training_set_accuracy()))
			print("Perceptron test set accuracy = {}".format(classifier.test_set_accuracy()))
			print("Perceptron Cross validation accuracy = {}".format(classifier.cross_validation_accuracy()))
		elif selected_classifier == cls.SvmClassifier.NAME:
			classifier = cls.SvmClassifier(feature_matrix, labels, regularization_param, shuffle=False)
			classifier.apply_feature_scaling(self.selected_feature_scaling_type)
			self.classifier = classifier
			classifier.train()

			print("SVM training set accuracy = {}".format(classifier.training_set_accuracy()))
			print("SVM test set accuracy = {}".format(classifier.test_set_accuracy()))
			print("SVM cross validation accuracy = {}".format(classifier.cross_validation_accuracy()))
		elif selected_classifier == cls.LdaClassifier.NAME:
			classifier = cls.LdaClassifier(feature_matrix, labels, shuffle=False)
			classifier.apply_feature_scaling(self.selected_feature_scaling_type)
			self.classifier = classifier
			classifier.train()

			print("LDA training set accuracy = {}".format(classifier.training_set_accuracy()))
			print("LDA test set accuracy = {}".format(classifier.test_set_accuracy()))
			print("LDA cross validation accuracy = {}".format(classifier.cross_validation_accuracy()))

			# TODO: Learning curves
		elif selected_classifier == cls.ANNClassifier.NAME:
			classifier = cls.ANNClassifier(feature_matrix, labels, shuffle=False)
			classifier.apply_feature_scaling(self.selected_feature_scaling_type)
			self.classifier = classifier
			classifier.train()

			print("MLP training set accuracy = {}".format(classifier.training_set_accuracy()))
			print("MLP test set accuracy = {}".format(classifier.test_set_accuracy()))
			print("MLP cross validation accuracy = {}".format(classifier.cross_validation_accuracy()))
		elif selected_classifier == cls.VotingClassifier.NAME:
			classifier = cls.VotingClassifier(feature_matrix, labels, DEFAULT_VOTING_CLASSIFIERS, shuffle=False)
			classifier.apply_feature_scaling(self.selected_feature_scaling_type)
			self.classifier = classifier
			self.classifier.train()

			print("VOTING training set accuracy = {}".format(classifier.training_set_accuracy()))
			print("VOTING test set accuracy = {}".format(classifier.test_set_accuracy()))
			print("VOTING cross validation accuracy = {}".format(classifier.cross_validation_accuracy()))

		self.root_directory_changed = False

	def k_fold_cross_validation_clicked(self):
		if self.classifier is not None and self.classifier.get_data_set() is not None:
			try:
				k = int(self.k_fold_edit.text())
				print(self.classifier.k_fold_cross_validation(k))
			except ValueError:
				pass

	def repeated_k_fold_clicked(self):
		if self.classifier is not None:
			k = int(self.k_fold_edit.text())

			average, std = self.classifier.repeated_k_fold_cross_validation(k)

			print("Average = {}, std = {}".format(average, std))

	def generate_performance_report(self):
		feature_matrix = self.data_set.raw_feature_matrix()
		labels = self.data_set.feature_matrix_labels()

		# Logistic Regression
		cls1 = cls.LogisticRegressionClassifier(feature_matrix, labels, shuffle=False)
		cls1.apply_feature_scaling(self.selected_feature_scaling_type)
		cls1.train(accuracy_threshold=self.get_accuracy_threshold())

		# KNN
		k_value = self.get_k_value()
		print("Using K value of {}".format(k_value))

		cls2 = cls.KNearestNeighborsClassifier(feature_matrix, labels, k_value, shuffle=False)
		cls2.apply_feature_scaling(self.selected_feature_scaling_type)

		# Perceptron
		cls3 = cls.PerceptronClassifier(feature_matrix, labels, shuffle=False)
		cls3.apply_feature_scaling(self.selected_feature_scaling_type)
		cls3.train(accuracy_threshold=self.get_accuracy_threshold())

		# SVM
		cls4 = cls.SvmClassifier(feature_matrix, labels, self.get_regularization_param(), shuffle=False)
		cls4.apply_feature_scaling(self.selected_feature_scaling_type)
		cls4.train()

		# LDA
		cls5 = cls.LdaClassifier(feature_matrix, labels, shuffle=False)
		cls5.apply_feature_scaling(self.selected_feature_scaling_type)
		cls5.train()

		# MLP
		cls6 = cls.ANNClassifier(feature_matrix, labels, shuffle=False)
		cls6.apply_feature_scaling(self.selected_feature_scaling_type)
		cls6.train()

		# Get performance measure from each
		prm1 = cls1.performance_measure()
		prm2 = cls2.performance_measure()
		prm3 = cls3.performance_measure()
		prm4 = cls4.performance_measure()
		prm5 = cls5.performance_measure()
		prm6 = cls6.performance_measure()

		plot_data = np.vstack((
			prm1.as_row_array(),
			prm2.as_row_array(),
			prm3.as_row_array(),
			prm4.as_row_array(),
			prm5.as_row_array(),
			prm6.as_row_array()
		))

		plot_data = np.transpose(plot_data)

		x = np.arange(6)

		plt.title("Classifiers' Accuracy")

		plt.bar(x + 0.0, plot_data[0], width=0.25, label="Training Accuracy")
		plt.bar(x + 0.25, plot_data[1], width=0.25, label="Cross Validation Accuracy")
		plt.bar(x + 0.5, plot_data[2], width=0.25, label="Testing Accuracy")

		plt.xticks(x + 0.125, ("Logistic Regression", "kNN", "Perceptron", "SVM", "LDA", "MLP"))
		plt.ylabel("Accuracy %")

		plt.legend(loc="best")

		plt.show()

	def generate_error_descriptions(self):
		# TODO: Fix the error description method in all the classifiers. Problem with sample size and shape.
		feature_matrix = self.data_set.raw_feature_matrix()
		labels = self.data_set.feature_matrix_labels()

		# Logistic Regression
		cls1 = cls.LogisticRegressionClassifier(feature_matrix, labels, shuffle=False)
		cls1.apply_feature_scaling(self.selected_feature_scaling_type)
		cls1.train(accuracy_threshold=self.get_accuracy_threshold())

		# KNN
		k_value = self.get_k_value()
		print("Using K value of {}".format(k_value))

		cls2 = cls.KNearestNeighborsClassifier(feature_matrix, labels, k_value, shuffle=False)
		cls2.apply_feature_scaling(self.selected_feature_scaling_type)

		# Perceptron
		cls3 = cls.PerceptronClassifier(feature_matrix, labels, shuffle=False)
		cls3.apply_feature_scaling(self.selected_feature_scaling_type)
		cls3.train(accuracy_threshold=self.get_accuracy_threshold())

		# SVM
		cls4 = cls.SvmClassifier(feature_matrix, labels, self.get_regularization_param(), shuffle=False)
		cls4.apply_feature_scaling(self.selected_feature_scaling_type)
		cls4.train()

		# LDA
		cls5 = cls.LdaClassifier(feature_matrix, labels, shuffle=False)
		cls5.apply_feature_scaling(self.selected_feature_scaling_type)
		cls5.train()

		# MLP
		cls6 = cls.ANNClassifier(feature_matrix, labels, shuffle=False)
		cls6.apply_feature_scaling(self.selected_feature_scaling_type)
		cls6.train()

		errd1 = cls1.error_description()
		errd2 = cls2.error_description()
		errd3 = cls3.error_description()
		errd4 = cls4.error_description()
		errd5 = cls5.error_description()
		errd6 = cls6.error_description()

		unique_labels = self.data_set.unique_labels().flatten()
		text_labels = []

		for label in unique_labels:
			for trial_class in self.trial_classes:
				if trial_class.label == label:
					text_labels.append(trial_class.name)

		label_count = unique_labels.size

		plot_data = np.vstack((
			errd1.as_row_array(),
			errd2.as_row_array(),
			errd3.as_row_array(),
			errd4.as_row_array(),
			errd5.as_row_array(),
			errd6.as_row_array()
		))

		plot_data = plot_data.transpose()

		x = np.arange(6)

		plt.title("Error Description")

		for i in range(label_count):
			plt.bar(x + 0.25 * i, plot_data[i], width=0.25, label=text_labels[i])

		plt.xticks(x + 0.125, ("Logistic Regression", "kNN", "Perceptron", "SVM", "LDA", "MLP"))
		plt.ylabel("Error Percent")

		plt.legend(loc="best")
		plt.show()

	def visualize_data(self):
		if self.data_set is None:
			print("Please extract features before trying to visualize...")
			return
		self.data_set.apply_feature_scaling(self.selected_feature_scaling_type)
		feature_matrix = self.data_set.scaled_feature_matrix()
		labels = self.data_set.feature_matrix_labels()

		pca = PCA(n_components=2)
		x = pca.fit_transform(feature_matrix)

		plt.figure()
		plt.title("Data reduced to 2D using PCA")

		unique_labels = np.unique(labels)

		label_to_name = {}

		for label in unique_labels:
			for trial_class in self.trial_classes:
				if trial_class.label == label:
					label_to_name[label] = trial_class.name

		for label in unique_labels:
			x_values = x[labels.flatten() == label, :]
			plt.scatter(x_values[:, 0], x_values[:, 1], label=f"{label_to_name[label]}")

		ratio1 = int(pca.explained_variance_ratio_[0] * 100 * 100) / 100
		ratio2 = int(pca.explained_variance_ratio_[1] * 100 * 100) / 100

		plt.xlabel(f"PC1 %{ratio1}")
		plt.ylabel(f"PC2 %{ratio2}")

		plt.legend(loc="best")
		plt.show()

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
			trial_length = utils.obtain_trial_length_from_slice_index(self.root_directories[0])

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

	def update_root_directories_label(self):
		label_str = ""

		for directory_str in self.root_directories:
			label_str += directory_str + "\n"

		self.root_directory_label.setText(label_str)

	def add_root_directory_clicked(self):
		path = QFileDialog.getExistingDirectory(self, "Add Root Directory...")
		self.root_directories.append(path)
		self.update_root_directories_label()
		self.root_directory_changed = True

	def pop_root_directory_clicked(self):
		if len(self.root_directories) != 0:
			self.root_directories.pop()
			self.update_root_directories_label()
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
		self.class_image_pixmap = QPixmap(self.trial_class.get_image_path()).scaledToHeight(int(image_height), Qt.FastTransformation)
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

	INTERNAL_BUFFER_EXTRA_SIZE = INTERNAL_BUFFER_EXTRA_DURATION * global_config.SAMPLING_RATE

	CLASS_IMAGE_HEIGHT = 400

	MENTAL_TASK_DELAY = 1000

	EV3_MAC_ADDRESS = "00:16:53:4f:bd:54"

	ROBOT_CONTROL = False

	DEFAULT_ROBOT_SPEED = 10

	classifier: cls.SimpleClassifier

	def __init__(self, classifier: cls.SimpleClassifier, filter_settings: utils.FilterSettings,
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

		self.robot_connect_btn = QPushButton("Connect to EV3")
		self.robot_connect_btn.clicked.connect(self.connect_clicked)

		self.manual_control_checkbox = QCheckBox("Manual Control")

		self.ev3 = ev3.EV3(self.EV3_MAC_ADDRESS)
		self.motor_control = ev3.MotorControl(1, 8, self.ev3)
		self.previous_direction = None

		self.previous_test_set_accuracy = 0.0
		self.previous_cross_validation_accuracy = 0.0

		self.root_layout.addWidget(utils.construct_horizontal_box([
			QLabel("Feature Window Size (sec): "), self.feature_window_edit,
			QLabel("Repetition Interval (sec): "), self.repetition_interval_edit,
			QLabel("Detection Threshold: "), self.detection_threshold_edit,
			self.online_training_checkbox,
			self.robot_connect_btn, self.manual_control_checkbox
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
				self.feature_extraction_info.range_size(),
				int(self.INTERNAL_BUFFER_EXTRA_SIZE + self.config.feature_window_size * self.feature_extraction_info.sampling_rate)
		), dtype=float)

	def connect_clicked(self):
		self.ev3.connect()

	def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
		print("key pressed")
		if self.manual_control_checkbox.isChecked():
			if event.key() == Qt.Key_A:
				self.motor_control.turn_left_from_middle(90, self.DEFAULT_ROBOT_SPEED)
			elif event.key() == Qt.Key_D:
				self.motor_control.turn_right_from_middle(90, self.DEFAULT_ROBOT_SPEED)
			elif event.key() == Qt.Key_W:
				self.motor_control.forward(self.DEFAULT_ROBOT_SPEED)
			elif event.key() == Qt.Key_S:
				self.motor_control.backward(self.DEFAULT_ROBOT_SPEED)

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
		# 	if type(self.classifier) == cls.LogisticRegressionClassifier:
		# 		self.previous_cross_validation_accuracy = self.classifier.cross_validation_accuracy()
		# 		self.previous_test_set_accuracy = self.classifier.test_set_accuracy()
		# 	elif type(self.classifier) == cls.KNearestNeighborsClassifier:
		# 		self.previous_cross_validation_accuracy = self.classifier.cross_validation_accuracy()
		# 		self.previous_test_set_accuracy = self.classifier.test_set_accuracy()
		# 	elif type(self.classifier) == cls.PerceptronClassifier:
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
			# 	if type(self.classifier) == cls.LogisticRegressionClassifier:
			# 		self.classifier.train()
			# 	elif type(self.classifier) == cls.KNearestNeighborsClassifier:
			# 		self.log("No training for KNN...")
			# 	elif type(self.classifier) == cls.PerceptronClassifier:
			# 		self.classifier.train()
			#
			# 	self.log("Training over...")
			#
			# 	test_set_accuracy = -1
			# 	cross_validation_set_accuracy = -1
			#
			# 	if type(self.classifier) == cls.LogisticRegressionClassifier:
			# 		cross_validation_set_accuracy = self.classifier.cross_validation_accuracy()
			# 		test_set_accuracy = self.classifier.test_set_accuracy()
			# 	elif type(self.classifier) == cls.KNearestNeighborsClassifier:
			# 		cross_validation_set_accuracy = self.classifier.cross_validation_accuracy()
			# 		test_set_accuracy = self.classifier.test_set_accuracy()
			# 	elif type(self.classifier) == cls.PerceptronClassifier:
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
			first_index = self.feature_extraction_info.first_electrode() - 1
			last_index = self.feature_extraction_info.last_electrode()  # Not including
			self.data_buffer[:, self.data_buffer.shape[1] - raw_eeg_data.shape[1]:] = raw_eeg_data[first_index:last_index, :]

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

		label = self.classifier.classify(feature_data)

		while type(label) == np.ndarray:
			label = label[0]

		print(f"label = {label}")

		direction = None

		if online_training:
			correct_label = self.current_mental_task.label

			if label != correct_label:
				self.log("Wrong classification!")
			else:
				self.log("Correct Classification")

			self.classifier.get_data_set().append_to(feature_data, np.array([correct_label]), data.DataSubSetType.TRAINING)
			# TODO: Might block execution for too long
			self.classifier.train()
			self.log("Training accuracy: " + self.classifier.training_set_accuracy())

		path = ""
		for trial_class in self.trial_classes:
			if trial_class.label == label:
				path = trial_class.get_image_path()
				direction = trial_class.direction
				break
		self.class_pixmap = QPixmap(path).scaledToHeight(self.CLASS_IMAGE_HEIGHT, Qt.FastTransformation)
		self.class_label.setPixmap(self.class_pixmap)

		if self.ROBOT_CONTROL:
			if direction == utils.Direction.LEFT:
				self.motor_control.turn_left_from_middle(90, self.DEFAULT_ROBOT_SPEED)
				print("left")
			elif direction == utils.Direction.RIGHT:
				self.motor_control.turn_right_from_middle(90, self.DEFAULT_ROBOT_SPEED)
				print("right")
			elif direction == utils.Direction.FORWARD:
				self.motor_control.forward(self.DEFAULT_ROBOT_SPEED)
				print("forward")
			elif direction == utils.Direction.BACKWARD:
				self.motor_control.backward(self.DEFAULT_ROBOT_SPEED)
				print("backward")

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
