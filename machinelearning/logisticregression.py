import math
import numpy

class logisticregression:
	def __init__(self):
		False
	def sigmoid(self, data):
		if isinstance(data, (int, float)):
			if isinstance(data, int):
				data = float(data)
			return 1/(1 + math.exp(-data))
		elif isinstance(data, list):
			for element in range(len(data)):
				data[element] = self.sigmoid(data[element])
			return data
		elif isinstance(data, numpy.matrixlib.defmatrix.matrix):
			for row in range(data.shape[0]):
				for col in range(data.shape[1]):
					data[row,col] = self.sigmoid(data.item(row, col))
			return data
	def hypothesis(self, theta, X):
		theta = numpy.mat(theta, dtype=float)	# Recast to a numpy matrix of floats
		X = numpy.mat(X, dtype=float)			# Recast to a numpy matrix of floats
		return self.sigmoid(theta.T * X)
		