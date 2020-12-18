import math
import os.path
import random
import time
from os import listdir

import PyQt5.QtCore
import serial
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QMainWindow, QComboBox, \
	QHBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QDialog, QSlider, QErrorMessage, QMessageBox, QCheckBox
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter

import global_config
import utils


class FileNameFormatter:
	BASE_NAME_PLACEHOLDER = "{b}"

	CLASS_NAME_PLACEHOLDER = "{c}"

	TRIAL_NUMBER_PLACEHOLDER = "{t}"

	@staticmethod
	def format(placeholder_str: str, base_name: str, class_name: str, trial_number: int) -> str:
		placeholder_str = placeholder_str.replace(FileNameFormatter.BASE_NAME_PLACEHOLDER, base_name)
		placeholder_str = placeholder_str.replace(FileNameFormatter.CLASS_NAME_PLACEHOLDER, class_name)
		placeholder_str = placeholder_str.replace(FileNameFormatter.TRIAL_NUMBER_PLACEHOLDER, str(trial_number))
		return placeholder_str


class FeedbackProvider:

	image_paths = [
		global_config.IMAGES_SSD_DRIVER_LETTER + ":/EEG_GUI_OpenBCI/class_images/checkmark.jpg",
		global_config.IMAGES_SSD_DRIVER_LETTER + ":/EEG_GUI_OpenBCI/class_images/thumbs_up.png",
		global_config.IMAGES_SSD_DRIVER_LETTER + ":/EEG_GUI_OpenBCI/class_images/thumbs_down.png"
				   ]

	proportions = [0.4, 0.4, 0.2]

	def get_image(self) -> str:
		d = random.random()

		print(d)

		sum = 0

		for i in range(len(self.proportions)):
			p = self.proportions[i]

			if sum <= d < (sum + p):
				return self.image_paths[i]

			sum += p


class TrialConfigurations:

	DEFAULT_START_DELAY = 4

	DEFAULT_TRIAL_DURATION = 5

	DEFAULT_REPETITIONS = 10

	DEFAULT_RELAXATION_PERIOD = 3

	DEFAULT_CLASSES = [utils.VibroTactileClasses.LEFT_CLASS, utils.VibroTactileClasses.RIGHT_CLASS]

	DEFAULT_ROOT_DIRECTORY = ""

	def __init__(self, start_delay: int = DEFAULT_START_DELAY,
					trial_duration: int = DEFAULT_TRIAL_DURATION,
					repetitions: int = DEFAULT_REPETITIONS,
					relaxation_period: int = DEFAULT_RELAXATION_PERIOD,
					classes: [utils.TrialClass] = DEFAULT_CLASSES,
					root_directory: str = DEFAULT_ROOT_DIRECTORY,
					feedback_provider: FeedbackProvider = FeedbackProvider(),
				 	vibration_control: bool = False,
				 	left_frequency: int = 0,
				 	right_frequency: int = 0):
		self.start_delay = start_delay
		self.trial_duration = trial_duration
		self.repetitions = repetitions
		self.relaxation_period = relaxation_period
		self.classes = classes
		self.root_directory = root_directory
		self.feedback_provider = feedback_provider
		self.vibration_control = vibration_control
		self.left_frequency = left_frequency
		self.right_frequency = right_frequency
		self.overwrite_files = True
		self.non_empty_root_directory = False

	def validate_saving_info(self) -> bool:
		return self.root_directory != "" and os.path.isdir(self.root_directory)

	def eeg_data_path(self) -> str:
		full_path = self.root_directory + "/" + global_config.EEG_DATA_FILE_NAME
		return full_path

	def slice_index_path(self) -> str:
		full_path = self.root_directory + "/" + global_config.SLICE_INDEX_FILE_NAME
		return full_path

	def labels(self):
		labels = []
		for trial_class in self.classes:
			labels.append(trial_class.label)
		return labels


class TrialConfigDialog(QDialog):

	def __init__(self, config: TrialConfigurations = TrialConfigurations()):
		super().__init__()
		self.config = config
		self.available_classes = self.config.classes.copy()

		self.setWindowTitle("Edit Trial Configurations")

		self.root_layout = QGridLayout()
		self.setLayout(self.root_layout)
		self.root_layout.setAlignment(PyQt5.QtCore.Qt.AlignTop | PyQt5.QtCore.Qt.AlignLeft)

		# Add title
		title = QLabel("<h1>Edit Trial Configurations</h1>")
		title.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(title, 0, 0, 1, 3)

		# Start delay edit
		self.start_delay_edit = QLineEdit()
		self.start_delay_edit.setText(str(self.config.start_delay))
		self.start_delay_edit.textChanged.connect(self.update_total_trial_duration_label)  # Update info label
		self.root_layout.addWidget(self.label_widgets_row("Start Delay (sec): ", [self.start_delay_edit]), 1, 0, 1, 1)

		# Trial duration edit
		self.trial_duration_edit = QLineEdit()
		self.trial_duration_edit.setText(str(self.config.trial_duration))
		self.trial_duration_edit.textChanged.connect(self.update_total_trial_duration_label)  # Update info label
		self.root_layout.addWidget(self.label_widgets_row("Trial Duration (sec): ", [self.trial_duration_edit]), 1, 1, 1, 1)

		# Repetitions edit
		self.repetitions_edit = QLineEdit()
		self.repetitions_edit.setText(str(self.config.repetitions))
		self.repetitions_edit.textChanged.connect(self.update_total_trial_duration_label)  # Update info label
		self.root_layout.addWidget(self.label_widgets_row("Repetitions: ", [self.repetitions_edit]), 2, 0, 1, 1)

		# Relaxation period
		self.relaxation_period_edit = QLineEdit()
		self.relaxation_period_edit.setText(str(self.config.relaxation_period))
		self.relaxation_period_edit.textChanged.connect(self.update_total_trial_duration_label)  # Update info label
		self.root_layout.addWidget(self.label_widgets_row("Relaxation Period (sec): ", [self.relaxation_period_edit]), 2, 1, 1, 1)

		# Class count
		self.class_count_slider = QSlider()
		self.class_count_slider.setOrientation(PyQt5.QtCore.Qt.Horizontal)
		self.class_count_slider.setRange(2, len(self.available_classes))
		self.class_count_slider.setValue(len(self.config.classes))
		self.class_count_slider.valueChanged.connect(self.class_count_update)

		self.root_layout.addWidget(self.label_widgets_row("Number of Classes: ", [self.class_count_slider]), 3, 0, 1, 1)

		self.class_picker_widget = QWidget()
		self.class_picker_layout = QHBoxLayout()
		self.class_picker_widget.setLayout(self.class_picker_layout)

		self.classes_combo_boxes = []

		self.root_layout.addWidget(self.class_picker_widget, 3, 1, 1, 1)

		self.class_count_update()

		# Vibration control
		self.vibration_control_checkbox = QCheckBox("Vibration Control")
		self.vibration_control_checkbox.setChecked(self.config.vibration_control)
		self.left_freq_edit = QLineEdit()
		self.left_freq_edit.setText(str(self.config.left_frequency))
		self.right_freq_edit = QLineEdit()
		self.right_freq_edit.setText(str(self.config.right_frequency))

		self.root_layout.addWidget(utils.construct_horizontal_box([
			self.vibration_control_checkbox, QLabel("Left Frequency (Hz):"), self.left_freq_edit, QLabel("Right Frequency (Hz):"),
			self.right_freq_edit
		]), 4, 0, 1, 3)

		# File saving settings
		second_title = QLabel("<h2>Save Files</h2>")
		second_title.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(second_title, 5, 0, 1, 3)

		self.root_directory_path_label = QLabel(self.config.root_directory)
		self.change_root_directory_btn = QPushButton("Select/Change")
		self.change_root_directory_btn.clicked.connect(self.change_root_directory_clicked)

		self.root_layout.addWidget(self.label_widgets_row("Root Directory: ",
									[self.root_directory_path_label, self.change_root_directory_btn]), 6, 0, 1, 3)

		info_title = QLabel("<h3>Info</h3>")
		info_title.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(info_title, 7, 0, 1, 3)

		self.total_trial_duration_label = QLabel()
		self.total_trial_duration_label.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(self.total_trial_duration_label, 8, 0, 1, 3)

		self.trials_for_class_label = QLabel()
		self.trials_for_class_label.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(self.trials_for_class_label, 9, 0, 1, 3)

		self.update_total_trial_duration_label()
		self.update_trials_for_class_label()

		self.save_btn = QPushButton("Save")
		self.save_btn.clicked.connect(self.save)
		self.cancel_btn = QPushButton("Cancel")
		self.cancel_btn.clicked.connect(self.cancel)

		self.root_layout.addWidget(self.save_btn, 10, 0, 1, 1)
		self.root_layout.addWidget(self.cancel_btn, 10, 1, 1, 1)

	def obtain_info_from_line_edits(self):
		if utils.is_integer(self.start_delay_edit.text()):
			self.config.start_delay = int(self.start_delay_edit.text())

		if utils.is_integer(self.trial_duration_edit.text()):
			self.config.trial_duration = int(self.trial_duration_edit.text())

		if utils.is_integer(self.repetitions_edit.text()):
			self.config.repetitions = int(self.repetitions_edit.text())

		if utils.is_integer(self.relaxation_period_edit.text()):
			self.config.relaxation_period = int(self.relaxation_period_edit.text())

		if utils.is_integer(self.left_freq_edit.text()):
			self.config.left_frequency = int(self.left_freq_edit.text())

		if utils.is_integer(self.right_freq_edit.text()):
			self.config.right_frequency = int(self.right_freq_edit.text())

		self.config.vibration_control = self.vibration_control_checkbox.isChecked()

	def save(self):
		print("Save clicked")

		self.obtain_info_from_line_edits()

		class_count = len(self.classes_combo_boxes)

		self.config.classes.clear()

		for i in range(class_count):
			class_name = self.classes_combo_boxes[i].currentText()

			for available_class in self.available_classes:
				if available_class.name == class_name:
					print("Adding class named {}".format(available_class.name))
					self.config.classes.append(available_class)

		self.config.root_directory = self.root_directory_path_label.text()

		if self.config.root_directory == "":
			print("Please select a root directory")
			return

		content = listdir(self.config.root_directory)

		if global_config.EEG_DATA_FILE_NAME in content or global_config.SLICE_INDEX_FILE_NAME in content:
			print("Directory contains data from a previous trial...")

			msg = QMessageBox()
			msg.setIcon(QMessageBox.Warning)
			msg.setText("The selected directory contains data from a previous trial.\n")

			msg.setInformativeText("You can choose to override the existing data or append to it.")
			msg.setWindowTitle("Warning")
			msg.setDetailedText("Do you want to overwrite the files in the selected directory or append the data to them?")

			overwrite_btn = QPushButton("Overwrite Files")
			msg.addButton(overwrite_btn, QMessageBox.DestructiveRole)

			append_btn = QPushButton("Append To Files")
			msg.addButton(append_btn, QMessageBox.YesRole)

			cancel_btn = QPushButton("Cancel")
			msg.addButton(cancel_btn, QMessageBox.RejectRole)

			# msg.setDefaultButton(append_btn)

			msg.exec()

			if msg.clickedButton() == cancel_btn:
				return
			elif msg.clickedButton() == overwrite_btn:
				self.config.overwrite_files = True
			elif msg.clickedButton() == append_btn:
				self.config.overwrite_files = False

			self.config.non_empty_root_directory = True
			print("Saving and closing, overwrite? {}".format(self.config.overwrite_files))

			self.close()
		else:
			print("Saving and closing")

			self.close()

	def cancel(self):
		self.close()

	def update_total_trial_duration_label(self):
		previous_repetitions = self.config.repetitions
		self.obtain_info_from_line_edits()

		if previous_repetitions != self.config.repetitions:
			self.update_trials_for_class_label()

		duration = self.config.start_delay + \
						self.config.repetitions * (self.config.trial_duration + self.config.relaxation_period)

		minutes = duration // 60
		seconds = duration - minutes * 60

		if minutes == 1:
			self.total_trial_duration_label.setText(
				"<h3>Estimated time to complete trial: {} minute and {} seconds</h3>".format(minutes, seconds))
		else:
			self.total_trial_duration_label.setText(
				"<h3>Estimated time to complete trial: {} minutes and {} seconds</h3>".format(minutes, seconds))

	def update_trials_for_class_label(self):
		repetitions = self.config.repetitions
		class_count = self.class_count_slider.value()

		try:
			self.trials_for_class_label.setText(
				"<h3>Expected trials for each class: {}</h3>".format(1 / class_count * repetitions))
		except AttributeError:
			pass

	def class_count_update(self):
		class_count = self.class_count_slider.value()

		if len(self.classes_combo_boxes) != 0:
			self.clear_class_picker_layout()
			self.classes_combo_boxes.clear()

		for i in range(class_count):
			self.classes_combo_boxes.append(self.create_class_combo_box())
			if i < len(self.available_classes):
				self.classes_combo_boxes[-1].setCurrentText(self.available_classes[i].name)
			self.class_picker_layout.addWidget(self.label_widgets_row(str(self.available_classes[i].label),
												[self.classes_combo_boxes[-1]]))
		self.update_trials_for_class_label()

	def change_root_directory_clicked(self):
		path_to_directory = QFileDialog.getExistingDirectory(self, "Root Directory")
		self.root_directory_path_label.setText(path_to_directory)

	def create_class_combo_box(self):
		combo_box = QComboBox()
		for i in range(len(self.available_classes)):
			combo_box.addItem(self.available_classes[i].name)
		return combo_box

	def clear_class_picker_layout(self):
		while self.class_picker_layout.count() > 0:
			child = self.class_picker_layout.takeAt(0)
			if child.widget():
				child.widget().deleteLater()

	@staticmethod
	def label_widgets_row(label_text: str, widgets: [QWidget]):
		layout = QHBoxLayout()
		layout.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		widget = QWidget()
		widget.setLayout(layout)

		layout.addWidget(QLabel(label_text))
		for item in widgets:
			layout.addWidget(item)
		return widget

	def edit_config(self) -> TrialConfigurations:
		self.show()
		self.exec()
		return self.config


class TrialConductor(QMainWindow):

	vibration_serial: serial.Serial
	HOME_IMAGE_PATH = global_config.IMAGES_SSD_DRIVER_LETTER + ":/EEG_GUI_OpenBCI/class_images/home.png"

	# TRIAL_CLASSES = utils.AlphaRhythmClasses.ALL.copy()
	# TRIAL_CLASSES = utils.SsvepClasses.ALL.copy()
	TRIAL_CLASSES = utils.VibroTactileClasses.ALL.copy()

	def __init__(self, board: BoardShim):
		super(TrialConductor, self).__init__()
		self.setWindowTitle("Trial Conductor")

		self.board = board

		self.vibration_serial = None

		self.start_timer = QTimer()
		self.start_timer.setSingleShot(True)

		self.next_trial_timer = QTimer()
		self.next_trial_timer.setSingleShot(True)

		self.relaxation_timer = QTimer()
		self.relaxation_timer.setSingleShot(True)

		self.eeg_recording_timer = QTimer()

		self.running = False
		self.trial_count = 0
		self.sample_count = 0
		self.slice_generator = utils.SliceIndexGenerator(0, [])

		# Initialize default trial configurations
		self.configurations = TrialConfigurations(classes=self.TRIAL_CLASSES)

		# Initialize root layout and root widget
		self.root_layout = QGridLayout()
		# self.root_layout.setAlignment(PyQt5.QtCore.Qt.AlignLeft | PyQt5.QtCore.Qt.AlignTop)
		self.root_widget = QWidget()
		self.root_widget.setLayout(self.root_layout)

		self.setCentralWidget(self.root_widget)

		# A parent widget for the top button bar
		self.button_bar_widget = QWidget()
		self.button_bar_layout = QHBoxLayout()
		self.button_bar_widget.setLayout(self.button_bar_layout)

		self.begin_trial_btn = QPushButton("Begin Trial")
		self.begin_trial_btn.clicked.connect(self.begin_trial)

		self.terminate_trial_btn = QPushButton("Terminate Trial")
		self.terminate_trial_btn.setEnabled(False)
		self.terminate_trial_btn.clicked.connect(self.terminate_trial)

		self.config_btn = QPushButton("Edit Configurations")
		self.config_btn.clicked.connect(self.configure)

		self.button_bar_layout.addWidget(self.begin_trial_btn)
		self.button_bar_layout.addWidget(self.terminate_trial_btn)
		self.button_bar_layout.addWidget(self.config_btn)

		self.root_layout.addWidget(self.button_bar_widget, 0, 0, 1, 5)

		self.timer_label = QLabel("")
		self.timer_label.setFont(QFont("Serif", 20))

		self.timer_start_time = time.time()

		self.timer_label.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.root_layout.addWidget(self.timer_label, 1, 0, 1, 5)

		self.class_image_pixmap = QPixmap()
		self.class_image_pixmap.load(self.HOME_IMAGE_PATH)

		self.class_image_view = QLabel()
		# self.class_image_view.setScaledContents(True)
		self.class_image_view.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
		self.class_image_view.setPixmap(self.class_image_pixmap)
		self.root_layout.addWidget(self.class_image_view, 2, 0, 5, 5)

	def begin_trial(self):
		if not self.running:

			if not self.configurations.validate_saving_info():
				dialog = QErrorMessage()
				dialog.showMessage("Please select a directory in which the data will be saved before starting!")
				dialog.exec()
				return

			self.slice_generator = utils.SliceIndexGenerator(global_config.SAMPLING_RATE, self.configurations.classes)

			self.start_eeg_recording()

			self.create_timers()

			self.terminate_trial_btn.setEnabled(True)
			self.running = True
			self.start_timer.timeout.connect(self.next_class)
			self.start_timer.start(self.configurations.start_delay * 1000)

	def start_eeg_recording(self):

		self.board.start_stream()

		if self.eeg_recording_timer is None:
			self.eeg_recording_timer = QTimer()
		self.eeg_recording_timer.timeout.connect(self.read_eeg_data)
		# TODO: Introduce a config variable specifying how many times a second should the data be read.
		self.eeg_recording_timer.start(200)

	def read_eeg_data(self):
		seconds_since_start = time.time() - self.timer_start_time
		seconds = math.floor(self.configurations.trial_duration + self.configurations.relaxation_period - seconds_since_start)
		self.timer_label.setText("{} sec".format(seconds))
		if self.board.get_board_data_count() > 0:
			raw_data = self.board.get_board_data()
			raw_eeg_data = utils.extract_eeg_data(raw_data, global_config.BOARD_ID)

			self.sample_count += raw_eeg_data.shape[1]  # Add the number of columns present in the newly read data to the count.

			if self.configurations.validate_saving_info():
				# file_name = FileNameFormatter.format(self.configurations.file_name_template,
				# 										self.configurations.file_basename, self.current_class.name, self.trial_count)
				# full_path = self.configurations.root_directory + "/" + file_name + ".csv"
				# print("Saving to {}".format(full_path))

				full_path = self.configurations.root_directory + "/" + global_config.EEG_DATA_FILE_NAME

				# Currently saves in append mode. If their are files in the directory and they have the same name,
				# the new data would be appended to the previous, instead of overwriting them.
				DataFilter.write_file(raw_eeg_data, full_path, "a")

	def stop_eeg_recording(self):
		self.eeg_recording_timer.deleteLater()
		self.eeg_recording_timer = None
		self.board.stop_stream()

	def create_timers(self):
		if self.start_timer is None:
			self.start_timer = QTimer()
			self.start_timer.setSingleShot(True)

		if self.next_trial_timer is None:
			self.next_trial_timer = QTimer()
			self.next_trial_timer.setSingleShot(True)

		if self.relaxation_timer is None:
			self.relaxation_timer = QTimer()
			self.relaxation_timer.setSingleShot(True)

	def destroy_timers(self):
		self.start_timer.deleteLater()
		self.next_trial_timer.deleteLater()
		self.relaxation_timer.deleteLater()

		self.start_timer = None
		self.next_trial_timer = None
		self.relaxation_timer = None

	def next_class(self):
		utils.start_vibration(self.vibration_serial, self.configurations.left_frequency, self.configurations.right_frequency)
		if self.trial_count >= self.configurations.repetitions:
			self.terminate_trial()
			return
		self.destroy_timers()
		self.create_timers()
		self.timer_start_time = time.time()
		index = random.randrange(len(self.configurations.classes))

		self.slice_generator.add_slice(
			self.configurations.labels()[index], self.sample_count,
			self.sample_count + global_config.SAMPLING_RATE * self.configurations.trial_duration
		)

		self.class_image_pixmap.load(self.configurations.classes[index].image_path)
		self.class_image_view.setPixmap(self.class_image_pixmap)
		self.trial_count += 1

		print("Next class, count = {}".format(self.trial_count))
		self.setWindowTitle("Trial {} out of {}".format(self.trial_count, self.configurations.repetitions))

		self.relaxation_timer.timeout.connect(self.show_relaxation_image)
		self.relaxation_timer.start(self.configurations.trial_duration * 1000)

	def show_relaxation_image(self):
		utils.stop_vibration(self.vibration_serial)
		self.destroy_timers()
		self.create_timers()
		print("Relax")
		QApplication.beep()
		self.class_image_pixmap.load(self.configurations.feedback_provider.get_image())
		self.class_image_view.setPixmap(self.class_image_pixmap)
		self.next_trial_timer.timeout.connect(self.next_class)
		self.next_trial_timer.start(self.configurations.relaxation_period * 1000)

	def terminate_trial(self):
		self.destroy_timers()
		self.stop_eeg_recording()
		utils.stop_vibration(self.vibration_serial)

		self.timer_label.setText("Press Start")

		append = not self.configurations.overwrite_files
		self.slice_generator.write_to_file(self.configurations.root_directory, append=append)

		self.class_image_pixmap.load(self.HOME_IMAGE_PATH)
		self.class_image_view.setPixmap(self.class_image_pixmap)
		self.terminate_trial_btn.setEnabled(False)
		self.running = False
		self.trial_count = 0
		self.configurations.overwrite_files = False
		self.configurations.non_empty_root_directory = True

	def configure(self):
		if self.vibration_serial is not None:
			self.vibration_serial.close()
			self.vibration_serial = None
		self.configurations = TrialConfigDialog(config=self.configurations).edit_config()
		if self.configurations.non_empty_root_directory:
			if not self.configurations.overwrite_files:
				self.sample_count = utils.obtain_last_trial_index_from_slice(self.configurations.root_directory)
				classes_from_file = utils.obtain_trial_classes_from_slice_index(self.configurations.root_directory)

				invalid = False

				for trial_class in classes_from_file:
					if trial_class not in self.configurations.classes:
						invalid = True
						break

				if invalid:
					msg = QMessageBox()
					msg.setText("Cannot append new data to existing one.")
					msg.setInformativeText("The existing data and the new data cannot be concatenated together.")
					msg.setDetailedText("This might occur duo to an attempt to save")
					msg.setIcon(QMessageBox.Critical)
					msg.setStandardButtons(QMessageBox.Ok)

					msg.exec()
					self.sample_count = 0
					self.configure()
				else:  # Append files
					self.sample_count = utils.obtain_last_trial_index_from_slice(self.configurations.root_directory)

			else:  # Overwrite files
				print("*" * 10 + "WARNING: OVERWRITING FILES" + "*"*10)

				answer = input("Are you sure? (Y/N)")

				if answer == "Y":
					if os.path.exists(self.configurations.eeg_data_path()):
						os.remove(self.configurations.eeg_data_path())

					if os.path.exists(self.configurations.slice_index_path()):
						os.remove(self.configurations.slice_index_path())
				else:
					print("Please select a different directory")

		if self.configurations.vibration_control:
			self.vibration_serial = serial.Serial(port=utils.vibration_port(), baudrate=115200, timeout=5000)
			if not self.vibration_serial.isOpen():
				print("Opening port")
				self.vibration_serial.open()
			else:
				print("Port is already open")

	def closeEvent(self, event) -> None:
		self.vibration_serial.close()


def main():
	app = QApplication([])
	app.setStyle(global_config.APP_STYLE)

	BoardShim.enable_board_logger()
	BoardShim.enable_dev_board_logger()

	params = BrainFlowInputParams()
	params.serial_port = utils.cyton_port()

	board = BoardShim(global_config.BOARD_ID, params)

	window = TrialConductor(board)
	window.show()

	if not board.is_prepared():
		print("Preparing board from trial gui")
		board.prepare_session()

	app.exec()

	board.release_session()


if __name__ == "__main__":
	main()
