from matplotlib import pyplot as plt
import numpy as np
import PyQt5.QtCore
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSlider, QApplication
from pyqtgraph import PlotWidget

# t = np.linspace(0, 1, 10000)
# g_2hz = np.sin(2*2*np.pi*t)
# g_5hz = np.sin(5*2*np.pi*t)
# g_2hz_5hz = g_2hz * g_5hz
#
# winding_2hz = g_2hz_5hz * np.exp(2*np.pi*1j*t)
#
# plt.figure()
# plt.plot(g_2hz)
# plt.figure()
# plt.plot(g_5hz)
# plt.figure()
# plt.plot(g_2hz_5hz)
# plt.figure()
# plt.plot(np.real(winding_2hz), np.imag(winding_2hz))
# plt.show()


def plot_01_update():
	global slider_01, plot_01, t
	g = np.sin(slider_01.value() / 10 * 2 * np.pi * t)
	plot_01.clear()
	plot_01.plot(t, g)
	update_sum_plot()


def plot_02_update():
	global slider_02, plot_02, t
	g = np.sin(slider_02.value() / 10 * 2 * np.pi * t)
	plot_02.clear()
	plot_02.plot(t, g)
	update_sum_plot()


def wave_sum():
	global slider_01, slider_02, t
	return np.sin(slider_01.value() / 10 * 2 * np.pi * t) + np.sin(slider_02.value() * 2 * np.pi * t)


def update_sum_plot():
	global sum_plot
	sum_plot.clear()
	sum_plot.plot(t, wave_sum())
	update_winding_plot()


def update_winding_plot():
	global slider_03, t, winding_plot
	f = slider_03.value() / 10
	winding = wave_sum() * np.exp(-2 * np.pi * f * 1j * t)
	winding_avg = np.average(winding)
	winding_plot.clear()
	winding_plot.plot(np.real(winding), np.imag(winding))
	winding_plot.plot([np.real(winding_avg)], [np.imag(winding_avg)], pen=None, symbol='o')


t = np.linspace(0, 1, 1000000)

app = QApplication([])

root = QWidget()
layout = QVBoxLayout()
root.setLayout(layout)

slider_01 = QSlider()
slider_01.setOrientation(PyQt5.QtCore.Qt.Horizontal)
slider_01.setRange(0, 200)
slider_01.valueChanged.connect(plot_01_update)
layout.addWidget(slider_01)

plot_01 = PlotWidget()
layout.addWidget(plot_01)

slider_02 = QSlider()
slider_02.setOrientation(PyQt5.QtCore.Qt.Horizontal)
slider_02.setRange(0, 200)
slider_02.valueChanged.connect(plot_02_update)
layout.addWidget(slider_02)

plot_02 = PlotWidget()
layout.addWidget(plot_02)

sum_plot = PlotWidget()
layout.addWidget(sum_plot)

slider_03 = QSlider()
slider_03.setOrientation(PyQt5.QtCore.Qt.Horizontal)
slider_03.setRange(0, 200)
slider_03.valueChanged.connect(update_winding_plot)
layout.addWidget(slider_03)

winding_plot = PlotWidget()
layout.addWidget(winding_plot)

root.show()
app.exec()
