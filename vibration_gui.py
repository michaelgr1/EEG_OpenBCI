import serial
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QGridLayout, QLineEdit, QLabel, QPushButton

import global_config
import utils


class VibrationController(QMainWindow):

	def __init__(self):
		super().__init__()
		self.setWindowTitle("Vibration Control")
		self.vibration_serial = serial.Serial(port=utils.vibration_port(), baudrate=115200, timeout=5000)
		if not self.vibration_serial.isOpen():
			self.vibration_serial.open()

		self.root_widget = QWidget()
		self.root_layout = QGridLayout()
		self.root_widget.setLayout(self.root_layout)
		self.setCentralWidget(self.root_widget)

		self.left_freq_edit = QLineEdit()
		self.right_freq_edit = QLineEdit()

		self.root_layout.addWidget(utils.construct_horizontal_box([
			QLabel("Left Frequency (Hz):"), self.left_freq_edit,
			QLabel("Right Frequency (Hz):"), self.right_freq_edit
		]), 0, 0, 1, 3)

		self.left_power_edit = QLineEdit("204")
		self.right_power_edit = QLineEdit("204")

		self.root_layout.addWidget(utils.construct_horizontal_box([
			QLabel("Left Power 0-255:"), self.left_power_edit,
			QLabel("Right Power 0-255:"), self.right_power_edit
		]), 1, 0, 1, 3)

		self.stop_btn = QPushButton("Stop Vibration")
		self.stop_btn.clicked.connect(self.stop_vibration)
		self.start_btn = QPushButton("Start Vibration")
		self.start_btn.clicked.connect(self.start_vibration)

		self.root_layout.addWidget(utils.construct_horizontal_box([
			self.stop_btn, self.start_btn
		]), 2, 1, 1, 1)

	def start_vibration(self):
		left_freq = -1
		if utils.is_integer(self.left_freq_edit.text()):
			left_freq = int(self.left_freq_edit.text())

		right_freq = -1
		if utils.is_integer(self.right_freq_edit.text()):
			right_freq = int(self.right_freq_edit.text())

		left_power = 204
		if utils.is_integer(self.left_power_edit.text()):
			left_power = int(self.left_power_edit.text())

		right_power = 204
		if utils.is_integer(self.right_power_edit.text()):
			right_power = int(self.right_power_edit.text())

		if left_freq != -1 and right_freq != -1:
			utils.start_vibration(self.vibration_serial, left_freq, right_freq, left_power, right_power)

	def stop_vibration(self):
		utils.stop_vibration(self.vibration_serial)

	def closeEvent(self, event) -> None:
		self.vibration_serial.close()


def main():
	app = QApplication([])
	app.setStyle(global_config.APP_STYLE)

	window = VibrationController()
	window.show()

	app.exec()


if __name__ == "__main__":
	main()
