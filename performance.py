import numpy as np

class ClassifierPerformanceMeasure:

	def __init__(self, train_accuracy: float, crsv_accuracy: float, test_accuracy: float):
		self.train_accuracy = train_accuracy
		self.crsv_accuracy = crsv_accuracy
		self.test_accuracy = test_accuracy

	def as_row_array(self) -> np.array:
		return np.array([
			[self.train_accuracy, self.crsv_accuracy, self.test_accuracy]
		])
