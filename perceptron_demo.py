import numpy as np

from matplotlib import pyplot as plt

import utils

import pyqtgraph as pg

from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication
from PyQt5.QtGui import QMouseEvent

import classification


class PerceptronDemo(QMainWindow):

	def __init__(self):
		super().__init__()

		self.graph = pg.PlotWidget()
		self.graph.plot(np.random.random(10))
		# self.graph.mousePressEvent.connect()

		self.current_class = +1
		self.labels = np.zeros(0)

		self.x_data = np.zeros(0, dtype=float)
		self.y_data = np.zeros(0, dtype=float)

		self.setCentralWidget(self.graph)
		self.show()

	def mousePressEvent(self, a0: QMouseEvent) -> None:
		self.x_data = np.append(self.x_data, a0.x())
		self.y_data = np.append(self.y_data, a0.y())
		self.labels = np.append(self.labels, self.current_class)
		self.update_plot()

	def update_plot(self):
		self.graph.clear()

		for i in range(self.x_data.shape[0]):
			if self.labels[i] == -1:
				self.graph.plot((self.x_data[i], self.y_data[i]), pen=None, symbol="o")
			elif self.labels[i] == +1:
				self.graph.plot((self.x_data[i], self.y_data[i]), pen=None, symbol="x")


def parabola_fitting():
	x = np.linspace(-5, 5, 22).reshape((22, 1))

	data = np.concatenate((x, x**2, x ** 2 + 1), axis=1)

	data = np.concatenate((data, np.concatenate((x, x**2, x ** 2 - 1), axis=1)), axis=0)

	labels = np.append(np.ones(22), np.zeros(22)).reshape((44, 1))

	perceptron = classification.PerceptronClassifier(data, labels)
	perceptron.train()

	print("Accuracy = {}".format(perceptron.test_set_accuracy()))

	# print(perceptron.classify(np.array([3, 3])))
	# print(perceptron.classify(np.array([-3, -3])))
	# print(perceptron.classify(np.array([0.5, 0.5])))
	# print(perceptron.classify(np.array([-0.5, 0])))

	t = np.linspace(-20, 20, 100)

	plt.plot(t, -(perceptron.weights[0] + perceptron.weights[1] * t + perceptron.weights[2] * t**2) / perceptron.weights[3])

	plt.scatter(x, x**2 + 1, c="b")
	plt.scatter(x, x**2 - 1, c="r")

	plt.show()


def circle_fitting():

	x = np.array([
		[2, 2],
		[2, -2],
		[-2, 2],
		[-2, -2],
		[3, 3],
		[3, -3],
		[-3, 3],
		[-3, -3],
		[0, 0],
		[1, 1],
		[1, -1],
		[-1, 1],
		[-1, -1],
		[1.5, 1.5],
		[-1.5, 1.5],
		[1.5, -1.5],
		[-1.5, -1.5]
	])

	labels = np.append(np.ones((8, 1)), np.zeros((9, 1))).reshape((17, 1))

	data = np.concatenate((x, x[:, 0].reshape(17, 1)**2), axis=1)

	perceptron = classification.PerceptronClassifier(data, labels)
	perceptron.train()

	print("Accuracy = {}".format(perceptron.test_set_accuracy()))

	t = np.linspace(-20, 20, 100)

	plt.plot(t, -(perceptron.weights[0] + perceptron.weights[1] * t + perceptron.weights[3] * t**2) / perceptron.weights[2])

	plt.scatter(x[:8, 0], x[:8, 1], c="b")
	plt.scatter(x[8:, 0], x[8:, 1], c="r")

	plt.show()


if __name__ == "__main__":

	# app = QApplication([])

	# window = PerceptronDemo()

	# app.exec()

	# data = np.array([
	# 	[-1, -1],
	# 	[-1, 0],
	# 	[0, -1],
	# 	[-2, -2],
	# 	[-2, -1],
	# 	[-1, -2],
	# 	[2, 2],
	# 	[1, 2],
	# 	[2, 1],
	# 	[1, 1],
	# 	[1, 0],
	# 	[0, 1]
	# ])
	#
	# labels = np.array([
	# 	[-1],
	# 	[-1],
	# 	[-1],
	# 	[-1],
	# 	[-1],
	# 	[-1],
	# 	[+1],
	# 	[+1],
	# 	[+1],
	# 	[+1],
	# 	[+1],
	# 	[+1]
	# ])

	parabola_fitting()

	# circle_fitting()
