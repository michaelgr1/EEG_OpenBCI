import sys

import pyqtgraph as pg
import serial
from PyQt5.QtChart import QChartView, QChart, QBarSet, QBarSeries, QBarCategoryAxis, QValueAxis
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QMainWindow, QLabel, QPushButton, QFileDialog, QSlider, \
	QProgressDialog, QErrorMessage, QCheckBox
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter

import global_config
import utils


class ResonanceFrequencyFinder(QMainWindow):

	# AVAILABLE_WINDOW_SIZES = ["5 Sec", "8 Sec", "10 Sec", "15 Sec", "20 Sec"]

	DEFAULT_GRAPH_PADDING = 2

	DEFAULT_FFT_WINDOW_SIZE = pow(2, utils.closest_power_of_two(5 * global_config.SAMPLING_RATE)) / global_config.SAMPLING_RATE

	DEFAULT_RECORDING_DURATION = DEFAULT_FFT_WINDOW_SIZE * 5

	DEFAULT_MIN_FREQUENCY = 17

	DEFAULT_MAX_FREQUENCY = 35

	DEFAULT_FREQUENCY_STEP = 2

	# Used to create a band for which the average frequency amplitude is computed
	DEFAULT_FREQUENCY_PADDING = 0.2

	DEFAULT_BANDPASS_MIN = 11

	DEFAULT_BANDPASS_MAX = 30

	DEFAULT_C3_CHANNEL_INDEX = 4

	DEFAULT_CZ_CHANNEL_INDEX = 3

	DEFAULT_C4_CHANNEL_INDEX = 2

	def __init__(self, board: BoardShim):
		super().__init__()
		self.setGeometry(0, 0, 1800, 900)
		self.setWindowTitle("Resonance-Like Frequency")

		self.board = board

		self.recording_progress_dialog = None
		self.eeg_data_buffer = utils.EegData()
		self.reading_timer = QTimer()
		self.recording = False

		self.recording_reference = False
		self.reference_eeg_data = utils.EegData()

		self.root_widget = QWidget()
		self.root_layout = QGridLayout()
		self.root_widget.setLayout(self.root_layout)
		self.setCentralWidget(self.root_widget)

		title = QLabel("<h1>Resonance Frequency Finder</h1>")
		title.setAlignment(Qt.AlignCenter)
		self.root_layout.addWidget(title, 0, 0, 1, 3)

		# window_size_label = QLabel("window size: ")
		# window_size_label.setAlignment(Qt.AlignRight)

		# self.window_size_combo_box = QComboBox()
		# self.window_size_combo_box.addItems(self.AVAILABLE_WINDOW_SIZES)

		self.root_directory_label = QLabel("Root Directory")
		self.select_root_directory = QPushButton("Select/Change")
		self.select_root_directory.clicked.connect(self.pick_root_directory)

		self.record_btn = QPushButton("Record")
		self.record_btn.setEnabled(False)
		self.record_btn.clicked.connect(self.record_clicked)

		self.record_reference_btn = QPushButton("Record Reference")
		self.record_reference_btn.clicked.connect(self.record_reference_clicked)

		# self.root_layout.addWidget(utils.construct_horizontal_box([
		# 	window_size_label, self.window_size_combo_box, self.record_btn
		# ]), 1, 0, 1, 3)

		self.vibration_control_checkbox = QCheckBox("Vibration Control")
		self.vibration_serial = None

		self.root_layout.addWidget(utils.construct_horizontal_box([
			self.record_btn, self.record_reference_btn, self.root_directory_label, self.select_root_directory, self.vibration_control_checkbox
		]), 1, 0, 1, 3)

		self.current_freq_label = QLabel()

		self.root_layout.addWidget(utils.construct_horizontal_box([self.current_freq_label]), 2, 0, 1, 3)

		self.frequency_slider = QSlider()
		self.frequency_slider.setRange(self.DEFAULT_MIN_FREQUENCY, self.DEFAULT_MAX_FREQUENCY)
		self.frequency_slider.setSingleStep(self.DEFAULT_FREQUENCY_STEP)
		self.frequency_slider.setTickInterval(self.DEFAULT_FREQUENCY_STEP)
		self.frequency_slider.valueChanged.connect(self.update_freq_label)
		self.frequency_slider.setTickPosition(QSlider.TicksBelow)
		self.frequency_slider.setOrientation(Qt.Horizontal)

		min_freq_label = QLabel(f"<b>{self.DEFAULT_MIN_FREQUENCY} Hz</b>")
		max_freq_label = QLabel(f"<b>{self.DEFAULT_MAX_FREQUENCY} Hz</b>")

		self.root_layout.addWidget(utils.construct_horizontal_box([
			min_freq_label, self.frequency_slider, max_freq_label
		]), 3, 0, 1, 3)

		self.c3_amplitude_bar_set = QBarSet("Electrode C3")
		self.cz_amplitude_bar_set = QBarSet("Electrode Cz")
		self.c4_amplitude_bar_set = QBarSet("Electrode C4")

		self.frequencies = []

		for freq in range(self.DEFAULT_MIN_FREQUENCY, self.DEFAULT_MAX_FREQUENCY + 1, self.DEFAULT_FREQUENCY_STEP):
			self.frequencies.append(f"{freq} Hz")
			self.c3_amplitude_bar_set.append(1)
			self.cz_amplitude_bar_set.append(1)
			self.c4_amplitude_bar_set.append(1)

		self.freq_axis = QBarCategoryAxis()
		self.freq_axis.append(self.frequencies)

		self.amplitude_axis = QValueAxis()
		self.amplitude_axis.setRange(0, 4)

		self.freq_chart = QChart()
		self.freq_chart.setAnimationOptions(QChart.SeriesAnimations)

		self.electrodes_data_series = QBarSeries()
		self.electrodes_data_series.append(self.c3_amplitude_bar_set)
		self.electrodes_data_series.append(self.cz_amplitude_bar_set)
		self.electrodes_data_series.append(self.c4_amplitude_bar_set)

		self.freq_chart.addSeries(self.electrodes_data_series)
		self.freq_chart.setTitle("<h1>Frequency Amplitude Increase</h1>")
		self.freq_chart.addAxis(self.freq_axis, Qt.AlignBottom)
		self.freq_chart.addAxis(self.amplitude_axis, Qt.AlignLeft)

		self.electrodes_data_series.attachAxis(self.amplitude_axis)
		self.electrodes_data_series.attachAxis(self.freq_axis)

		self.frequency_amplitude_graph = QChartView(self.freq_chart)
		self.frequency_amplitude_graph.setRenderHint(QPainter.Antialiasing)

		self.root_layout.addWidget(self.frequency_amplitude_graph, 4, 0, 15, 3)

		self.auto_adjust_axis()

	def update_freq_label(self):
		self.current_freq_label.setText("Selected Frequency: {} Hz".format(self.frequency_slider.value()))

	def pick_root_directory(self):
		path = QFileDialog.getExistingDirectory(self, "Root Directory...")
		self.root_directory_label.setText(path)

	def record_clicked(self, reference: bool = False):
		# selected_window_text = self.window_size_combo_box.currentText()
		# window_size_text = selected_window_text.replace(" Sec", "")

		# window_size = -1
		#
		# if utils.is_integer(window_size_text):
		# 	window_size = int(window_size_text)
		# else:
		# 	print("Invalid window size...")
		# 	return

		# window_size_in_samples = window_size * SAMPLING_RATE

		recording_duration_in_samples = self.DEFAULT_RECORDING_DURATION * global_config.SAMPLING_RATE

		if not reference and (self.frequency_slider.value() - self.DEFAULT_MIN_FREQUENCY) % self.DEFAULT_FREQUENCY_STEP != 0:
			err = QErrorMessage(self)
			err.showMessage("Invalid Frequency Selected")
			err.exec()
			return

		self.recording_progress_dialog = \
			QProgressDialog("Reading EEG data from board...", "Stop Recording", 0, int(recording_duration_in_samples), self)
		self.recording_progress_dialog.setWindowTitle("Reading Data, Please Wait...")
		self.recording_progress_dialog.setWindowModality(Qt.WindowModal)
		self.recording_progress_dialog.show()

		if reference:
			self.recording_reference = True
		else:
			self.recording = True
			self.eeg_data_buffer.clear()
			if self.vibration_control_checkbox.isChecked():
				if self.vibration_serial is None:
					self.vibration_serial = serial.Serial(port=utils.vibration_port(), baudrate=115200, timeout=5000)

				if not self.vibration_serial.isOpen():
					self.vibration_serial.open()
				frequency = self.frequency_slider.value()
				utils.start_vibration(self.vibration_serial, frequency, frequency)

		self.board.start_stream()

		self.reading_timer = QTimer()
		self.reading_timer.timeout.connect(self.read_data)
		self.reading_timer.start(100)

	def record_reference_clicked(self):
		print("Record reference clicked")
		self.record_clicked(reference=True)
		if self.vibration_control_checkbox.isChecked():
			if self.vibration_serial is None:
				self.vibration_serial = serial.Serial(port=utils.vibration_port(), baudrate=115200, timeout=5000)

			if not self.vibration_serial.isOpen():
				self.vibration_serial.open()
			frequency = self.frequency_slider.value()
			utils.start_vibration(self.vibration_serial, frequency, frequency)

	def read_data(self):
		if not self.recording and not self.recording_reference:
			return

		recording_duration_in_samples = self.recording_progress_dialog.maximum()

		if self.recording_reference:
			if self.reference_eeg_data.get_channel_data(0).shape[0] > recording_duration_in_samples or\
					self.recording_progress_dialog.wasCanceled():
				self.stop_recording(True)
				return

		if self.recording:
			if self.recording_progress_dialog.wasCanceled() or\
					self.eeg_data_buffer.get_channel_data(0).shape[0] > recording_duration_in_samples:
				self.stop_recording(self.recording_reference)
				return

		if self.board.get_board_data_count() > 0:
			raw_data = self.board.get_board_data()
			raw_eeg_data = utils.extract_eeg_data(raw_data, global_config.BOARD_ID)

			# c3 = raw_eeg_data[self.DEFAULT_C3_CHANNEL_INDEX, :]
			# cz = raw_eeg_data[self.DEFAULT_CZ_CHANNEL_INDEX, :]
			# c4 = raw_eeg_data[self.DEFAULT_C4_CHANNEL_INDEX, :]

			if self.recording_reference:
				self.reference_eeg_data.append_data(raw_eeg_data)
				self.recording_progress_dialog.setValue(self.reference_eeg_data.get_channel_data(0).shape[0])
			else:
				self.eeg_data_buffer.append_data(raw_eeg_data)
				self.recording_progress_dialog.setValue(self.eeg_data_buffer.get_channel_data(0).shape[0])

	def stop_recording(self, reference: bool = False):
		if self.reading_timer is not None:
			self.reading_timer.deleteLater()

		utils.stop_vibration(self.vibration_serial)

		self.board.stop_stream()
		self.recording = False
		self.recording_reference = False
		self.recording_progress_dialog.setValue(self.recording_progress_dialog.maximum())

		QApplication.beep()

		path = self.root_directory_label.text()

		if path != "":
			print("Saving Data")
			file_name = path + "/"
			if reference:
				file_name += "reference.csv"
			else:
				file_name += "freq_"+str(self.frequency_slider.value())+".csv"

			if reference:
				DataFilter.write_file(self.reference_eeg_data.to_row_array(), file_name, "a")
			else:
				DataFilter.write_file(self.eeg_data_buffer.to_row_array(), file_name, "a")

		if reference:
			self.record_btn.setEnabled(True)
			# self.record_reference_btn.setEnabled(False)
			self.reference_eeg_data.filter_all_channels(
				global_config.SAMPLING_RATE, self.DEFAULT_BANDPASS_MIN, self.DEFAULT_BANDPASS_MAX, True
			)
			print("Reference data saved...")
		else:
			print("Stopping the recording...")
			selected_frequency = self.frequency_slider.value()

			print("Filtering data...")

			self.eeg_data_buffer.filter_all_channels(
				global_config.SAMPLING_RATE, self.DEFAULT_BANDPASS_MIN, self.DEFAULT_BANDPASS_MAX, subtract_average=True
			)

			freq_band = utils.FrequencyBand(
				selected_frequency - self.DEFAULT_FREQUENCY_PADDING, selected_frequency + self.DEFAULT_FREQUENCY_PADDING)

			c3_freq_amplitude = self.eeg_data_buffer.feature_extractor(self.DEFAULT_C3_CHANNEL_INDEX, global_config.SAMPLING_RATE).\
				average_band_amplitude(freq_band, self.DEFAULT_FFT_WINDOW_SIZE)
			cz_freq_amplitude = self.eeg_data_buffer.feature_extractor(self.DEFAULT_CZ_CHANNEL_INDEX, global_config.SAMPLING_RATE).\
				average_band_amplitude(freq_band, self.DEFAULT_FFT_WINDOW_SIZE)
			c4_freq_amplitude = self.eeg_data_buffer.feature_extractor(self.DEFAULT_C4_CHANNEL_INDEX, global_config.SAMPLING_RATE).\
				average_band_amplitude(freq_band, self.DEFAULT_FFT_WINDOW_SIZE)

			c3_ref_freq = self.reference_eeg_data.feature_extractor(self.DEFAULT_C3_CHANNEL_INDEX, global_config.SAMPLING_RATE).\
				average_band_amplitude(freq_band, self.DEFAULT_FFT_WINDOW_SIZE)

			cz_ref_freq = self.reference_eeg_data.feature_extractor(self.DEFAULT_CZ_CHANNEL_INDEX, global_config.SAMPLING_RATE).\
				average_band_amplitude(freq_band, self.DEFAULT_FFT_WINDOW_SIZE)

			c4_ref_freq = self.reference_eeg_data.feature_extractor(self.DEFAULT_C4_CHANNEL_INDEX, global_config.SAMPLING_RATE).\
				average_band_amplitude(freq_band, self.DEFAULT_FFT_WINDOW_SIZE)

			c3_amplitude_difference = c3_freq_amplitude - c3_ref_freq
			cz_amplitude_difference = cz_freq_amplitude - cz_ref_freq
			c4_amplitude_difference = c4_freq_amplitude - c4_ref_freq

			print(f"""
				c3 amplitude diff = {c3_amplitude_difference}
				cz amplitude diff = {cz_amplitude_difference}
				c4 amplitude diff = {c4_amplitude_difference}
			""")

			# max_amplitude = max(c3_amplitude_difference, cz_amplitude_difference, c4_amplitude_difference)
			#
			# min_amplitude = min(c3_amplitude_difference, cz_amplitude_difference, c4_amplitude_difference)
			#
			# if self.amplitude_axis.max() < max_amplitude:
			# 	self.amplitude_axis.setMax(max_amplitude)
			#
			# if self.amplitude_axis.min() > min_amplitude:
			# 	self.amplitude_axis.setMin(min_amplitude)

			index = (selected_frequency - self.DEFAULT_MIN_FREQUENCY) // self.DEFAULT_FREQUENCY_STEP

			print("index = {}".format(index))

			self.c3_amplitude_bar_set.replace(index, c3_amplitude_difference)
			self.cz_amplitude_bar_set.replace(index, cz_amplitude_difference)
			self.c4_amplitude_bar_set.replace(index, c4_amplitude_difference)

			utils.auto_adjust_axis(self.amplitude_axis,
								[self.c3_amplitude_bar_set, self.cz_amplitude_bar_set, self.c4_amplitude_bar_set], self.DEFAULT_GRAPH_PADDING)

	def auto_adjust_axis(self):
		# Adjust the range so that everything is visible and add some gaps

		c3_min = sys.maxsize
		cz_min = sys.maxsize
		c4_min = sys.maxsize

		c3_max = -sys.maxsize
		cz_max = -sys.maxsize
		c4_max = -sys.maxsize

		for i in range(self.c3_amplitude_bar_set.count()):
			c3_min = min(c3_min, self.c3_amplitude_bar_set.at(i))
			cz_min = min(cz_min, self.cz_amplitude_bar_set.at(i))
			c4_min = min(c4_min, self.c4_amplitude_bar_set.at(i))

			c3_max = max(c3_max, self.c3_amplitude_bar_set.at(i))
			cz_max = max(cz_max, self.cz_amplitude_bar_set.at(i))
			c4_max = max(c4_max, self.c4_amplitude_bar_set.at(i))

		print("c3 min = {}, cz min = {}, c4 min = {}".format(c3_min, cz_min, c4_min))
		print("c3 max = {}, cz max = {}, c4 max = {}".format(c3_max, cz_max, c4_max))

		axis_min = min(0, c3_min, cz_min, c4_min) - self.DEFAULT_GRAPH_PADDING
		axis_max = max(0, c3_max, cz_max, c4_max) + self.DEFAULT_GRAPH_PADDING

		print("axis min = {}, axis max = {}".format(axis_min, axis_max))

		self.amplitude_axis.setMin(axis_min)
		self.amplitude_axis.setMax(axis_max)

	def closeEvent(self, event) -> None:
		self.vibration_serial.close()


def main():
	app = QApplication([])
	app.setStyle(global_config.APP_STYLE)

	BoardShim.enable_board_logger()

	params = BrainFlowInputParams()
	params.serial_port = utils.cyton_port()

	board = BoardShim(global_config.BOARD_ID, params)

	# Switch to using white background and black foreground
	pg.setConfigOption('background', 'w')
	pg.setConfigOption('foreground', 'k')

	window = ResonanceFrequencyFinder(board)
	window.show()

	board.prepare_session()

	app.exec()

	board.release_session()


if __name__ == "__main__":
	main()
