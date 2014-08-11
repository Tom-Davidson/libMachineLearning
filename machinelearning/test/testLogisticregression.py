import numpy
import unittest

class logisticregressionTests(unittest.TestCase):
	lr = False
	def setUp(self):
		from machinelearning import mlgraph, examples, logisticregression
		self.lr = logisticregression.logisticregression()
	def testSigmoid2D(self):
		answers = [0.7310585786300049, 0.8807970779778823, 0.9525741268224334]
		sigs = self.lr.sigmoid([1,2,3])
		self.assertEqual(sigs, answers)
	def testSigmoid3D(self):
		answers = [[0.7310585786300049, 0.8807970779778823, 0.9525741268224334]]
		sigs = self.lr.sigmoid([[1,2,3]])
		self.assertEqual(sigs, answers)
	def testHypothesisA(self):
		answers = [[0.95257413, 0.99330715, 0.99908895]]
		theta = [[0], [1], [0]] 			# [0 1 0]'
		X = [[8,1,6], [3,5,7], [4,9,2]]		# magic(3)
		y = self.lr.hypothesis(theta, X)
		self.assertTrue(numpy.allclose(y, answers))

