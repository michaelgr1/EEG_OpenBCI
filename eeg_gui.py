import PyQt5.QtCore
import numpy as np
import pyqtgraph as pg
from PyQt5.QtChart import QChartView, QChart, QBarCategoryAxis, QValueAxis, QBarSeries, QBarSet
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QWidget, QApplication, QCheckBox, QGridLayout, QMainWindow, QComboBox, QVBoxLayout, \
	QHBoxLayout, QLabel, QPushButton, QColorDialog, QStackedLayout, QLineEdit, QFileDialog
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams, LogLevels
from brainflow.data_filter import DataFilter, FilterTypes, WindowFunctions

import global_config
import trial_gui
import utils
from utils import AccumulatingAverage

FFT_WINDOW_SIZES = \
	[
		pow(2, 7) / global_config.SAMPLING_RATE,
		pow(2, 8) / global_config.SAMPLING_RATE,
		pow(2, 9) / global_config.SAMPLING_RATE,
		pow(2, 10) / global_config.SAMPLING_RATE
	]

DEFAULT_CHANNEL_MAX_HEIGHT = 300


class EegChannel(QWidget):

	HORIZONTAL_SCALE_OPTIONS = ["2 Sec", "5 Sec", "8 Sec", "10 Sec", "15 Sec", "20 Sec"]
	VERTICAL_SCALE_OPTIONS = ["50 μV", "100 μV", "200 μV", "400 μV", "800 μV", "1000 μV", "2000 μV"]

	BANDPASS_MIN_OPTIONS = ["0.5 Hz", "1.0 Hz", "3.0 Hz", "5.0 Hz", "7.0 Hz", "10.0 Hz"]
	BANDPASS_MAX_OPTIONS = ["45.0 Hz", "50.0 Hz", "55.0 Hz", "60.0 Hz", "80.0 Hz", "100.0 Hz"]

	# The channel saves internally data that can cover the window plus extra for filtering
	# this is the extra size in samples
	INTERNAL_BUFFER_EXTRA_SIZE = 2 * global_config.SAMPLING_RATE

	def __init__(self, pen_color='b', title="channel", show_controls: bool = True):
		super().__init__()

		self.pen = (pg.mkPen(color=pen_color))

		self.setMaximumHeight(DEFAULT_CHANNEL_MAX_HEIGHT)

		# Holds the average of all the eeg readings, gets subtracted during filtering
		self.eeg_average = AccumulatingAverage()

		self.config = utils.EegChannelConfigurations(global_config.BOARD_ID, self)

		# This array holds the most recent eeg data, it holds only the visible data, so its size depends on the window size
		self.unfiltered_data = np.zeros(self.buffer_size())
		# Same but filtered
		self.filtered_data = np.zeros(self.buffer_size())

		# Creates a new plot widget on which data will be displayed
		self.plot_widget = pg.PlotWidget()
		self.plot_widget.getPlotItem().setLabel(axis="bottom", text="Time (sec) ")
		self.plot_widget.getPlotItem().setLabel(axis="left", text="Amplitude (μV) ")
		self.plot_widget.getPlotItem().setLabel(axis="top", text=title)

		# Gets the plot data item of the plot widget
		self.data_item = self.plot_widget.getPlotItem().plot(pen=self.pen)

		# Root layout
		self.root = QVBoxLayout()
		self.setLayout(self.root)

		# Create a widget and layout to display controllers
		self.controls_layout = QHBoxLayout()
		self.controls_widget = QWidget()
		self.controls_widget.setLayout(self.controls_layout)

		# Create combo boxes for horizontal and vertical scale
		self.horizontal_scale_combo = QComboBox()
		self.horizontal_scale_combo.addItems(self.HORIZONTAL_SCALE_OPTIONS)
		self.horizontal_scale_combo.\
			setCurrentIndex(self.HORIZONTAL_SCALE_OPTIONS.index(str(self.config.visible_seconds) + " Sec"))
		self.horizontal_scale_combo.activated.connect(self.horizontal_scale_changed)

		self.vertical_scale_combo = QComboBox()
		self.vertical_scale_combo.addItems(self.VERTICAL_SCALE_OPTIONS)
		self.vertical_scale_combo.setCurrentIndex(self.VERTICAL_SCALE_OPTIONS.index(str(self.config.vertical_scale) + " μV"))
		self.vertical_scale_combo.activated.connect(self.vertical_scale_changed)
		self.vertical_scale_changed()

		# Create combo boxes for bandpass min and max frequencies
		self.bandpass_min_combo = QComboBox()
		self.bandpass_min_combo.addItems(self.BANDPASS_MIN_OPTIONS)
		self.bandpass_min_combo.setCurrentIndex(self.BANDPASS_MIN_OPTIONS.index(str(self.config.bandpass_min_freq) + " Hz"))
		self.bandpass_min_combo.activated.connect(self.bandpass_min_changed)

		self.bandpass_max_combo = QComboBox()
		self.bandpass_max_combo.addItems(self.BANDPASS_MAX_OPTIONS)
		self.bandpass_max_combo.setCurrentIndex(self.BANDPASS_MAX_OPTIONS.index(str(self.config.bandpass_max_freq) + " Hz"))
		self.bandpass_max_combo.activated.connect(self.bandpass_max_changed)

		self.notch_filter_checkbox = QCheckBox("Notch Filter")
		self.notch_filter_checkbox.setChecked(True)

		self.change_color_btn = QPushButton("Change Color")
		self.change_color_btn.clicked.connect(self.change_color_pressed)

		# Add controls to layout
		self.controls_layout.addWidget(QLabel("Horizontal Scale:"))
		self.controls_layout.addWidget(self.horizontal_scale_combo)

		self.controls_layout.addWidget(QLabel("Vertical Scale:"))
		self.controls_layout.addWidget(self.vertical_scale_combo)

		self.controls_layout.addWidget(QLabel("Bandpass Filter:"))
		self.controls_layout.addWidget(self.bandpass_min_combo)
		self.controls_layout.addWidget(QLabel("-"))
		self.controls_layout.addWidget(self.bandpass_max_combo)

		self.controls_layout.addWidget(self.notch_filter_checkbox)

		self.controls_layout.addWidget(self.change_color_btn)

		# Add widgets to root
		if show_controls:
			self.root.addWidget(self.controls_widget, alignment=PyQt5.QtCore.Qt.AlignLeft)
		self.root.addWidget(self.plot_widget)

	def change_color_pressed(self):
		color = QColorDialog.getColor()
		self.change_pen_color(color)

	def horizontal_scale_changed(self, index: int = -1):
		if index != -1:  # The user changed the value
			print("Horizontal scale changed by user")
			visible_seconds = int(self.horizontal_scale_combo.currentText().replace(" Sec", ""))
		else:
			visible_seconds = self.config.visible_seconds
			print("Horizontal scale changed not by user")
			self.horizontal_scale_combo.setCurrentIndex(self.HORIZONTAL_SCALE_OPTIONS.index(str(visible_seconds) + " Sec"))
		self.resize_visible_window(visible_seconds, notify_listener=(index != -1))

	def vertical_scale_changed(self, index: int = -1):
		if index != -1:
			vertical_scale = int(self.vertical_scale_combo.currentText().replace(" μV", ""))
			self.config.set_vertical_scale(vertical_scale)
		else:
			vertical_scale = self.config.vertical_scale
			self.vertical_scale_combo.setCurrentIndex(self.VERTICAL_SCALE_OPTIONS.index(str(vertical_scale) + " μV"))
		self.plot_widget.setYRange(min=-vertical_scale, max=vertical_scale)

	def bandpass_min_changed(self, index: int = -1):
		if index != -1:
			freq = float(self.bandpass_min_combo.currentText().replace(" Hz", ""))
			self.config.set_bandpass_min_freq(freq)
		else:
			self.bandpass_min_combo.setCurrentIndex(self.BANDPASS_MIN_OPTIONS.index(str(self.config.bandpass_min_freq) + " Hz"))

	def bandpass_max_changed(self, index: int = -1):
		if index != -1:
			freq = float(self.bandpass_max_combo.currentText().replace(" Hz", ""))
			self.config.set_bandpass_max_freq(freq)
		else:
			self.bandpass_max_combo.setCurrentIndex(self.BANDPASS_MAX_OPTIONS.index(str(self.config.bandpass_max_freq) + " Hz"))

	def change_pen_color(self, color):
		self.data_item.clear()
		self.pen = pg.mkPen(color=color)
		self.data_item = self.plot_widget.getPlotItem().plot(pen=self.pen)
		self.update_graph()

	def add_data_point(self, value, update=False):
		self.eeg_average.add_value(value)

		# Shifts the array one to the left and adds value as the last data point
		self.unfiltered_data = np.roll(self.unfiltered_data, shift=-1, axis=0)
		self.unfiltered_data[self.buffer_size() - 1] = value
		if update:
			self.update_graph()

	def add_data_points(self, eeg_data, apply_filters=True):
		for eeg_value in eeg_data:
			self.add_data_point(eeg_value, False)

		# Filter data
		if apply_filters:
			self.filter_data()
		else:
			self.filtered_data = self.unfiltered_data

		self.update_graph()

	def buffer_size(self):
		""""
			Computes the internal buffer size.
			The size depends on the visible window size defined in the config object
			and the extra trailing data for filtering defined by a class constant
		"""
		return self.config.window_size() + self.INTERNAL_BUFFER_EXTRA_SIZE

	def visible_filtered_data(self) -> np.ndarray:
		""""
			Returns filtered eeg data from the internal before which is visible and displayed on screen.
		"""
		return self.filtered_data[self.INTERNAL_BUFFER_EXTRA_SIZE:]

	def perform_fft(self, window_size: float):
		# filtered_data = self.visible_filtered_data()
		# length = pow(2, utils.closest_power_of_two(filtered_data.shape[0]))
		# frequencies = np.linspace(0, BoardShim.get_sampling_rate(BOARD_ID) / 2, int(length / 2 + 1))
		# return frequencies, DataFilter.perform_fft(filtered_data[:length], WindowFunctions.NO_WINDOW.value)
		return utils.FeatureExtractor(self.visible_filtered_data(), global_config.SAMPLING_RATE).fft(window_size)

	def filter_data(self):
		""""
			Filters the current EEG data. Uses raw data from unfiltered_data array and updates the filtered_data array.
		"""

		# Subtract average from data
		self.filtered_data = self.unfiltered_data - self.eeg_average.compute_average()

		if self.eeg_average.count > 2 * self.config.window_size():
			print("Clearing average...")
			self.eeg_average.reset()

		if self.notch_filter_checkbox.isChecked():
			# Notch filter
			DataFilter.perform_bandstop\
				(self.filtered_data, BoardShim.get_sampling_rate(global_config.BOARD_ID),
				 self.config.notch_freq, 2, 4, FilterTypes.BUTTERWORTH.value, 0)
		# Bandpass filter
		band_width = self.config.bandpass_max_freq - self.config.bandpass_min_freq
		band_center = self.config.bandpass_min_freq + band_width / 2
		DataFilter.perform_bandpass(self.filtered_data, BoardShim.get_sampling_rate(global_config.BOARD_ID),
		                            band_center, band_width, 4,
									FilterTypes.BUTTERWORTH.value, 0)

	def update_graph(self):
		""""
			Updates the graph. Draws the filtered values on screen, discards a given amount from the beginning to get rid of filter artifacts.
		"""
		win_size = self.config.window_size()
		time_values = utils.samples_to_seconds(np.linspace(-(win_size - 1), 0, win_size), global_config.BOARD_ID)
		self.data_item.setData(time_values, self.visible_filtered_data())

	def resize_visible_window(self, visible_seconds: int, notify_listener=True):
		""""
			Change the size of the visible window
		"""
		print("Resizing window, new size = {} sec".format(visible_seconds))
		new_window_size = visible_seconds * BoardShim.get_sampling_rate(global_config.BOARD_ID)
		current_window_size = self.unfiltered_data.shape[0] - self.INTERNAL_BUFFER_EXTRA_SIZE

		self.config.set_visible_seconds(visible_seconds, notify_owner=False, notify_listener=notify_listener)

		if new_window_size == current_window_size:
			return

		difference = int(abs(new_window_size - current_window_size))

		if new_window_size > current_window_size:
			self.unfiltered_data = np.concatenate((np.zeros(difference), self.unfiltered_data))
		else:
			self.unfiltered_data = self.unfiltered_data[difference:]

		self.filter_data()


class FrequencyGraph(QWidget):
	""""
		Creates a graph and draws the fft values for each added channel.
		Update graph should be called to actually perform fft and update the graph.
	"""

	def __init__(self):
		super().__init__()
		self.channels = []

		self.setMaximumHeight(DEFAULT_CHANNEL_MAX_HEIGHT)

		self.fft_window_size = 0

		self.plot_widget = pg.PlotWidget()
		self.plot_widget.getPlotItem().setLabel(axis="bottom", text="Frequency (Hz)")
		self.plot_widget.getPlotItem().setLabel(axis="left", text="Amplitude")
		self.plot_widget.getPlotItem().setLabel(axis="top", text="Frequency Domain")
		self.plot_widget.setXRange(min=0, max=40)

		self.root_layout = QStackedLayout()
		self.setLayout(self.root_layout)

		self.root_layout.addWidget(self.plot_widget)

	def add_channel(self, channel: EegChannel):
		self.channels.append(channel)

	def add_channels(self, channels):
		for channel in channels:
			self.add_channel(channel)

	def update_graph(self):
		self.plot_widget.clear()
		for channel in self.channels:
			frequency, fft_data = channel.perform_fft(self.fft_window_size)
			# print(fft_data)
			# (x, y) = utils.complex_arr_to_xy(fft_data)
			self.plot_widget.plot(frequency, np.abs(fft_data), pen=channel.pen)


class BandPowerGraph(QWidget):

	def __init__(self, name: str):
		super().__init__()

		self.band_power_chart = QChart()
		self.band_power_chart.setAnimationOptions(QChart.SeriesAnimations)

		self.channel_band_power_set = QBarSet("Band Power")
		self.channel_band_power_set.append(1)
		self.channel_band_power_set.append(1)
		self.channel_band_power_set.append(1)
		self.channel_band_power_set.append(1)
		self.channel_band_power_set.append(1)

		self.bands_axis = QBarCategoryAxis()
		self.bands_axis.append("Delta (1 - 3 Hz)")
		self.bands_axis.append("Theta (4 - 7 Hz)")
		self.bands_axis.append("Alpha (8 - 13 Hz)")
		self.bands_axis.append("Beta (13 - 30 Hz)")
		self.bands_axis.append("Gamma (30 - 100)")

		self.power_axis = QValueAxis()

		self.chart_series = QBarSeries()
		self.chart_series.append(self.channel_band_power_set)

		self.band_power_chart.addSeries(self.chart_series)
		self.band_power_chart.setTitle(f"<h1>Band Power For {name}</h1>")
		self.band_power_chart.addAxis(self.bands_axis, Qt.AlignBottom)
		self.band_power_chart.addAxis(self.power_axis, Qt.AlignLeft)

		self.chart_series.attachAxis(self.bands_axis)
		self.chart_series.attachAxis(self.power_axis)

		self.chart_view = QChartView(self.band_power_chart)
		self.chart_view.setRenderHint(QPainter.Antialiasing)

		self.root_layout = QStackedLayout()
		self.setLayout(self.root_layout)

		self.root_layout.addWidget(self.chart_view)

	def set_name(self, name: str):
		self.band_power_chart.setTitle(f"<h1>Band Power For {name}</h1>")

	def update_values(self, data: np.ndarray, fft_window_size: float = 0):
		eeg_data = utils.EegData(data)

		feature_extractor = eeg_data.feature_extractor(0, global_config.SAMPLING_RATE)

		self.channel_band_power_set.replace(
			0, feature_extractor.average_band_amplitude(utils.FrequencyBand.delta_freq_band(), fft_window_size)
		)

		self.channel_band_power_set.replace(
			1, feature_extractor.average_band_amplitude(utils.FrequencyBand.theta_freq_band(), fft_window_size)
		)

		self.channel_band_power_set.replace(
			2, feature_extractor.average_band_amplitude(utils.FrequencyBand.alpha_freq_band(), fft_window_size)
		)

		self.channel_band_power_set.replace(
			3, feature_extractor.average_band_amplitude(utils.FrequencyBand.beta_freq_band(), fft_window_size)
		)

		self.channel_band_power_set.replace(
			4, feature_extractor.average_band_amplitude(utils.FrequencyBand.gama_freq_band(), fft_window_size)
		)

	def auto_adjust_axis(self):
		utils.auto_adjust_axis(self.power_axis, [self.channel_band_power_set], 0.1)


class MainWindow(QMainWindow):

	def __init__(self, app: QApplication, board: BoardShim, screen_width, channel_count: int = 8):
		super(MainWindow, self).__init__()
		self.app = app
		self.setGeometry(PyQt5.QtCore.QRect(int(screen_width / 2 - 400), 50, 800, 1200))
		self.setWindowTitle("OpenBCI EEG")

		self.timer = None

		self.boardShim = board

		self.save_to_file = False
		self.file_path = ""

		# Define the root widget and its layout
		self.root = QWidget()
		self.main_layout = QGridLayout()
		self.root.setLayout(self.main_layout)
		self.setCentralWidget(self.root)

		self.global_controls_layout = QHBoxLayout()
		self.global_controls_widget = QWidget()
		self.global_controls_widget.setLayout(self.global_controls_layout)

		self.start_btn = QPushButton("Start Stream")
		self.start_btn.clicked.connect(self.start_stream)

		self.stop_btn = QPushButton("Stop Stream")
		self.stop_btn.clicked.connect(self.stop_stream)

		self.begin_trial_btn = QPushButton("Begin Trial")
		self.begin_trial_btn.clicked.connect(self.begin_trial)

		self.global_controls_layout.addWidget(self.start_btn)
		self.global_controls_layout.addWidget(self.stop_btn)
		self.global_controls_layout.addWidget(self.begin_trial_btn)

		self.unified_settings_check_box = QCheckBox("Unified Settings")
		self.unified_settings_check_box.setChecked(False)
		self.unified_settings_check_box.toggled.connect(self.unified_settings_checkbox_clicked)

		self.global_controls_layout.addWidget(self.unified_settings_check_box)

		self.main_layout.addWidget(self.global_controls_widget, 0, 0)

		self.file_settings_widget = QWidget()
		self.file_settings_layout = QHBoxLayout()
		self.file_settings_widget.setLayout(self.file_settings_layout)

		self.file_path_line_edit = QLineEdit()
		self.pick_file_button = QPushButton("Select File")
		self.pick_file_button.clicked.connect(self.pick_file_clicked)
		self.save_to_file_checkbox = QCheckBox("Save to file")

		self.file_settings_layout.addWidget(self.file_path_line_edit)
		self.file_settings_layout.addWidget(self.pick_file_button)
		self.file_settings_layout.addWidget(self.save_to_file_checkbox)

		self.main_layout.addWidget(self.file_settings_widget, 0, 1)

		self.fft_settings_widget = QWidget()
		self.fft_settings_layout = QHBoxLayout()
		self.fft_settings_layout.setAlignment(Qt.AlignCenter)
		self.fft_settings_widget.setLayout(self.fft_settings_layout)

		self.fft_window_size_combo = QComboBox()

		for window_size in FFT_WINDOW_SIZES:
			self.fft_window_size_combo.addItem(str(window_size))

		self.fft_window_size_combo.currentTextChanged.connect(self.fft_window_size_changed)

		self.fft_settings_layout.addWidget(QLabel("FFT window size: "))
		self.fft_settings_layout.addWidget(self.fft_window_size_combo)

		self.band_power_channel_select = QComboBox()

		for i in range(channel_count):
			self.band_power_channel_select.addItem(str(i))

		self.band_power_channel_select.currentTextChanged.connect(self.band_power_channel_changed)

		self.fft_settings_layout.addWidget(QLabel("Channel Band Power: "))
		self.fft_settings_layout.addWidget(self.band_power_channel_select)

		self.rescale_bandpower_graph = QPushButton("Rescale Band Power Graph")
		self.rescale_bandpower_graph.clicked.connect(self.rescale_bandpower_clicked)

		self.fft_settings_layout.addWidget(self.rescale_bandpower_graph)

		self.main_layout.addWidget(self.fft_settings_widget, 1, 0, 1, 3)

		channels_baseline = 2

		self.channels = []
		colors = ['r', 'g', 'b', 'k', 'y', 'c', 'm', 'r']

		row = channels_baseline
		column = 0

		# Creates objects to represent the eeg channels, adds them to the layout
		for i in range(channel_count):
			title = "Channel {}".format(i + 1)
			self.channels.append(EegChannel(pen_color=colors[i], title=title, show_controls=(i == 0)))
			row = channels_baseline + i // 2
			column = i % 2
			self.main_layout.addWidget(self.channels[i], row, column)

		self.fft_graph = FrequencyGraph()
		self.fft_graph.add_channels(self.channels)
		# Column could be either 0 or 1. When zero, adds fft graph to the right of the last channel, when one,
		# adds graph in a new row
		fft_layout_row = row + column
		fft_layout_column = (column + 1) % 2
		self.main_layout.addWidget(self.fft_graph, fft_layout_row, fft_layout_column)

		self.band_power_channel_index = 0
		self.band_power_graph = BandPowerGraph("Channel {}".format(self.band_power_channel_index + 1))
		band_power_layout_column = fft_layout_row + fft_layout_column
		self.main_layout.addWidget\
			(self.band_power_graph, band_power_layout_column, (fft_layout_column + 1) % 2, 1, band_power_layout_column * 2 + 1)

	def pick_file_clicked(self):
		self.file_path = QFileDialog.getSaveFileName(self, "Save EEG Session", filter="*.csv")[0]
		print(self.file_path)
		self.file_path_line_edit.setText(self.file_path)
		self.save_to_file_checkbox.setChecked(True)

	def begin_trial(self):
		self.stop_stream()
		# self.close()
		print("Starting trial")
		trial_gui.main()
		# self.close()

	def unified_settings_checkbox_clicked(self):
		if self.unified_settings_check_box.isChecked():
			print("Checkbox selected, adding change listeners")
			for channel in self.channels:
				channel.config.change_listener = self.on_config_changed
		else:
			print("Checkbox unselected, removing change listeners")
			for channel in self.channels:
				channel.config.change_listener = None

	def on_config_changed(self, config: utils.EegChannelConfigurations):
		print("Config changed, updating the rest of the channels to match")
		for channel in self.channels:
			if channel.config is not config:
				channel.config.match_config(config)

	def fft_window_size_changed(self, text: str):
		self.fft_graph.fft_window_size = float(text)
		self.band_power_graph.auto_adjust_axis()

	def band_power_channel_changed(self, text: str):
		self.band_power_channel_index = int(text)
		self.band_power_graph.set_name("Channel {}".format(self.band_power_channel_index + 1))
		self.band_power_graph.auto_adjust_axis()

	def start_stream(self):
		""""
			Start the data stream. If there is already some data displayed, it won't get cleaned.
		"""
		if self.timer is None:
			self.boardShim.start_stream()
			BoardShim.log_message(LogLevels.LEVEL_INFO.value, "The data stream has started!")

			self.save_to_file = self.save_to_file_checkbox.isChecked()
			self.file_path = self.file_path_line_edit.text()

			self.timer = QTimer()
			self.timer.timeout.connect(self.read_data)

			self.timer.start(100)

	def stop_stream(self):
		""""
			Stop the data stream coming from the board, eeg will freeze
		"""
		if self.timer is not None:
			self.timer = None
			self.boardShim.stop_stream()

	def rescale_bandpower_clicked(self):
		self.band_power_graph.auto_adjust_axis()

	def read_data(self):
		""""
			If available, read data from board and update the channels
		"""
		if self.boardShim.get_board_data_count() > 0:
			raw_data = self.boardShim.get_board_data()

			raw_eeg_data = utils.extract_eeg_data(raw_data, global_config.BOARD_ID)

			if self.save_to_file and self.file_path != "" and self.file_path.endswith(".csv"):
				DataFilter.write_file(raw_eeg_data, self.file_path, "a")

			for channel_index in range(len(self.channels)):
				self.channels[channel_index].add_data_points(raw_eeg_data[channel_index])

			self.fft_graph.update_graph()
			self.band_power_graph.update_values(
				self.channels[self.band_power_channel_index]
				.filtered_data.reshape(1, self.channels[self.band_power_channel_index].filtered_data.shape[0]),
				float(self.fft_window_size_combo.currentText()))


def main():
	app = QApplication([])
	app.setStyle(global_config.APP_STYLE)

	BoardShim.enable_board_logger()
	BoardShim.enable_dev_board_logger()

	params = BrainFlowInputParams()
	params.serial_port = utils.cyton_port()

	board = BoardShim(global_config.BOARD_ID, params)

	# Switch to using white background and black foreground
	pg.setConfigOption('background', 'w')
	pg.setConfigOption('foreground', 'k')

	window = MainWindow(app, board=board, screen_width=app.primaryScreen().size().width(), channel_count=5)
	window.show()

	board.prepare_session()

	app.exec()

	print("Release session")
	board.release_session()


if __name__ == "__main__":
	main()
