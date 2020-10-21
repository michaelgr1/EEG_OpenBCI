import classification
import numpy as np

X = np.array([
	[1, 1],
	[0, 2],
	[2, 0],
	[-1, -1],
	[0, -2],
	[-2, 0]
])

y = np.array([
	1, 1, 1, 0, 0, 0
])

svm = classification.SvmClassifier(X, y)
svm.train()

print(svm.classify(np.array([[3, 3]])))

print(svm.classify(np.array([[-3, -3]])))

print(svm.training_set_accuracy())
print(svm.test_set_accuracy())