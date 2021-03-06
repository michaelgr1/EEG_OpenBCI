import sys

import numpy as np
import pyqtgraph as pg
from PyQt5.QtChart import QChartView, QChart, QBarSet, QBarSeries, QBarCategoryAxis, QValueAxis
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QMainWindow, QLabel, QPushButton, QFileDialog, QSlider, \
	QProgressDialog, QErrorMessage
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter
from matplotlib import pyplot as plt

import global_config
import utils


class ResonanceFrequencyFinder(QMainWindow):

	# AVAILABLE_WINDOW_SIZES = ["5 Sec", "8 Sec", "10 Sec", "15 Sec", "20 Sec"]

	DEFAULT_GRAPH_PADDING = 2

	DEFAULT_FFT_WINDOW_SIZE = 3

	DEFAULT_RECORDING_DURATION = 10

	DEFAULT_MIN_FREQUENCY = 17

	DEFAULT_MAX_FREQUENCY = 35

	DEFAULT_FREQUENCY_STEP = 2

	# Used to create a band for which the average frequency amplitude is computed
	DEFAULT_FREQUENCY_PADDING = 0.2

	DEFAULT_BANDPASS_MIN = 11

	DEFAULT_BANDPASS_MAX = 40

	DEFAULT_C3_CHANNEL_INDEX = 5

	DEFAULT_CZ_CHANNEL_INDEX = 3

	DEFAULT_C4_CHANNEL_INDEX = 1

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

		self.index_generator = utils.FrequencyIndexGenerator(global_config.SAMPLING_RATE)
		self.eeg_sample_count = 0

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

		self.load_results_btn = QPushButton("Load Existing Data")
		self.load_results_btn.clicked.connect(self.load_existing_data)

		self.root_layout.addWidget(utils.construct_horizontal_box([
			self.record_btn, self.record_reference_btn, self.root_directory_label,
			self.select_root_directory, self.load_results_btn
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

		self.board.start_stream()

		self.reading_timer = QTimer()
		self.reading_timer.timeout.connect(self.read_data)
		self.reading_timer.start(100)

	def record_reference_clicked(self):
		print("Record reference clicked")
		if self.reference_eeg_data.get_channel_data(0).shape[0] > 0:
			self.reference_eeg_data.clear()
		self.record_clicked(reference=True)

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

			self.eeg_sample_count += raw_eeg_data.shape[1]

			path = self.root_directory_label.text()

			if path != "":
				full_path = path + "/" + global_config.RESONANCE_DATA_FILE_NAME
				DataFilter.write_file(raw_eeg_data, full_path, "a")

			# c3 = raw_eeg_data[self.DEFAULT_C3_CHANNEL_INDEX, :]
			# cz = raw_eeg_data[self.DEFAULT_CZ_CHANNEL_INDEX, :]
			# c4 = raw_eeg_data[self.DEFAULT_C4_CHANNEL_INDEX, :]

			if self.recording_reference:
				self.reference_eeg_data.append_data(raw_eeg_data)
				self.recording_progress_dialog.setValue(self.reference_eeg_data.get_channel_data(0).shape[0])
			else:
				self.eeg_data_buffer.append_data(raw_eeg_data)
				self.recording_progress_dialog.setValue(self.eeg_data_buffer.get_channel_data(0).shape[0])

	def load_existing_data(self):
		path = QFileDialog.getExistingDirectory(self, "Root Directory...")

		filter_settings = utils.FilterSettings(global_config.SAMPLING_RATE, self.DEFAULT_BANDPASS_MIN, self.DEFAULT_BANDPASS_MAX)

		frequencies, eeg_data, reference_data = utils.load_slice_and_filter_resonance_data(path, filter_settings)

		print(frequencies)

		size = len(frequencies)

		x = np.arange(size)

		x_ticks = []

		plot_data = np.zeros((3, size))

		for i in range(size):
			current_eeg_data = eeg_data[i]
			freq = frequencies[i]
			x_ticks.append(f"{freq} Hz")
			freq_band = utils.FrequencyBand(
				freq - self.DEFAULT_FREQUENCY_PADDING,
				freq + self.DEFAULT_FREQUENCY_PADDING)

			reference_c3_extractor = reference_data.feature_extractor(
				self.DEFAULT_C3_CHANNEL_INDEX, global_config.SAMPLING_RATE
			)

			reference_cz_extractor = reference_data.feature_extractor(
				self.DEFAULT_CZ_CHANNEL_INDEX, global_config.SAMPLING_RATE
			)

			reference_c4_extractor = reference_data.feature_extractor(
				self.DEFAULT_C4_CHANNEL_INDEX, global_config.SAMPLING_RATE
			)

			data_c3_extractor = current_eeg_data.feature_extractor(
				self.DEFAULT_C3_CHANNEL_INDEX, global_config.SAMPLING_RATE
			)

			data_cz_extractor = current_eeg_data.feature_extractor(
				self.DEFAULT_CZ_CHANNEL_INDEX, global_config.SAMPLING_RATE
			)

			data_c4_extractor = current_eeg_data.feature_extractor(
				self.DEFAULT_C4_CHANNEL_INDEX, global_config.SAMPLING_RATE
			)

			c3_diff, cz_diff, c4_diff = self.amplitude_diff(freq_band, reference_c3_extractor, reference_cz_extractor,
															reference_c4_extractor, data_c3_extractor,
															data_cz_extractor, data_c4_extractor)

			plot_data[0, i] = c3_diff
			plot_data[1, i] = cz_diff
			plot_data[2, i] = c4_diff

		plt.figure()
		plt.title("Amplitude Increase")

		plt.bar(x, plot_data[0], width=0.25, label="C3 amplitude increase")
		plt.bar(x + 0.25, plot_data[1], width=0.25, label="Cz amplitude increase")
		plt.bar(x + 0.50, plot_data[2], width=0.25, label="C4 amplitude increase")

		plt.xticks(x + 0.25, x_ticks)
		plt.ylabel("Average Band Amplitude")

		plt.legend(loc="best")
		plt.show()

	def amplitude_diff(self, freq_band, ref_c3_extractor, ref_cz_extractor, ref_c4_extractor, data_c3_extractor, data_cz_extractor, data_c4_extractor):
		ref_c3_amplitude = ref_c3_extractor.average_band_amplitude(freq_band, self.DEFAULT_FFT_WINDOW_SIZE)
		ref_cz_amplitude = ref_cz_extractor.average_band_amplitude(freq_band, self.DEFAULT_FFT_WINDOW_SIZE)
		ref_c4_amplitude = ref_c4_extractor.average_band_amplitude(freq_band, self.DEFAULT_FFT_WINDOW_SIZE)

		data_c3_amplitude = data_c3_extractor.average_band_amplitude(freq_band, self.DEFAULT_FFT_WINDOW_SIZE)
		data_cz_amplitude = data_cz_extractor.average_band_amplitude(freq_band, self.DEFAULT_FFT_WINDOW_SIZE)
		data_c4_amplitude = data_c4_extractor.average_band_amplitude(freq_band, self.DEFAULT_FFT_WINDOW_SIZE)

		c3_diff = data_c3_amplitude - ref_c3_amplitude
		cz_diff = data_cz_amplitude - ref_cz_amplitude
		c4_diff = data_c4_amplitude - ref_c4_amplitude

		return c3_diff, cz_diff, c4_diff

	def stop_recording(self, reference: bool = False):
		if self.reading_timer is not None:
			self.reading_timer.deleteLater()

		self.board.stop_stream()
		self.recording = False
		self.recording_reference = False
		self.recording_progress_dialog.setValue(self.recording_progress_dialog.maximum())

		recording_duration_in_samples = self.recording_progress_dialog.maximum()
		selected_freq = self.frequency_slider.value()

		if reference:
			sample_count = min(self.reference_eeg_data.get_channel_data(0).shape[0], recording_duration_in_samples)
			self.index_generator.add_slice(0, self.eeg_sample_count - sample_count, self.eeg_sample_count)
		else:
			sample_count = min(self.eeg_data_buffer.get_channel_data(0).shape[0], recording_duration_in_samples)
			self.index_generator.add_slice(selected_freq, self.eeg_sample_count - sample_count, self.eeg_sample_count)

		self.index_generator.write_to_file(self.root_directory_label.text())

		QApplication.beep()

		if reference:
			self.record_btn.setEnabled(True)
			# self.record_reference_btn.setEnabled(False)
			self.reference_eeg_data.filter_all_channels(
				global_config.SAMPLING_RATE, self.DEFAULT_BANDPASS_MIN, self.DEFAULT_BANDPASS_MAX, True
			)
			print("Reference data saved...")
		else:
			print("Stopping the recording...")

			print("Filtering data...")

			self.eeg_data_buffer.filter_all_channels(
				global_config.SAMPLING_RATE, self.DEFAULT_BANDPASS_MIN, self.DEFAULT_BANDPASS_MAX, subtract_average=True
			)

			reference_c3_extractor = self.reference_eeg_data.feature_extractor(
				self.DEFAULT_C3_CHANNEL_INDEX, global_config.SAMPLING_RATE
			)

			reference_cz_extractor = self.reference_eeg_data.feature_extractor(
				self.DEFAULT_CZ_CHANNEL_INDEX, global_config.SAMPLING_RATE
			)

			reference_c4_extractor = self.reference_eeg_data.feature_extractor(
				self.DEFAULT_C4_CHANNEL_INDEX, global_config.SAMPLING_RATE
			)

			data_c3_extractor = self.eeg_data_buffer.feature_extractor(
				self.DEFAULT_C3_CHANNEL_INDEX, global_config.SAMPLING_RATE
			)

			data_cz_extractor = self.eeg_data_buffer.feature_extractor(
				self.DEFAULT_CZ_CHANNEL_INDEX, global_config.SAMPLING_RATE
			)

			data_c4_extractor = self.eeg_data_buffer.feature_extractor(
				self.DEFAULT_C4_CHANNEL_INDEX, global_config.SAMPLING_RATE
			)

			for i in range(self.c3_amplitude_bar_set.count()):
				current_freq = int(self.frequencies[i].replace(" Hz", ""))
				if current_freq == selected_freq:
					freq_band = utils.FrequencyBand(
						current_freq - self.DEFAULT_FREQUENCY_PADDING,
						current_freq + self.DEFAULT_FREQUENCY_PADDING)

					c3_diff, cz_diff, c4_diff = self.amplitude_diff(freq_band, reference_c3_extractor,reference_cz_extractor, reference_c4_extractor,
																	data_c3_extractor, data_cz_extractor, data_c4_extractor)

					self.c3_amplitude_bar_set.replace(i, c3_diff)
					self.cz_amplitude_bar_set.replace(i, cz_diff)
					self.c4_amplitude_bar_set.replace(i, c4_diff)

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
