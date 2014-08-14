import numpy
import unittest


class logisticregressionTests(unittest.TestCase):
    lr = False

    def setUp(self):
        from machinelearning import mlgraph, examples, logisticregression

        self.lr = logisticregression.logisticregression()

    def testSigmoid2D(self):
        answers = [0.7310585786300049, 0.8807970779778823, 0.9525741268224334]
        sigs = self.lr.sigmoid([1, 2, 3])
        self.assertEqual(sigs, answers)

    def testSigmoid3D(self):
        answers = [[0.7310585786300049, 0.8807970779778823, 0.9525741268224334]]
        sigs = self.lr.sigmoid([[1, 2, 3]])
        self.assertEqual(sigs, answers)

    def testHypothesisA(self):
        answers = [[0.95257413, 0.99330715, 0.99908895]]
        theta = [[0], [1], [0]]  # [0 1 0]'
        X = [[8, 1, 6], [3, 5, 7], [4, 9, 2]]  # magic(3)
        y = self.lr.hypothesis(theta, X)
        self.assertTrue(numpy.allclose(y, answers))

    def testInitialParameters(self):
        answer = [0.0, 0.0, 0.0, 0.0]
        X = [[8, 1, 6], [3, 5, 7], [4, 9, 2]]  # magic(3)
        initialParameters = self.lr.initialParameters(X)
        self.assertTrue(numpy.allclose(initialParameters, answer))

    def testCost(self):
        answer = 0.04314969835997326
        theta = [[1], [0], [1], [0]] 			# [1 0 1 0]'
        X = [[8,1,6], [3,5,7], [4,9,2]]		    # magic(3)
        y = [[1], [1], [1]]                     # [1 1 1]'
        cost = self.lr.cost(theta, X, y)
        self.assertEqual(cost, answer)

#    def testGradient(self):
#        answer = [ [-0.04057365], [-0.32040761], [-0.04399154], [-0.24420556] ]
#        theta = [[1], [0], [1], [0]] 			# [1 0 1 0]'
#        X = [[8,1,6], [3,5,7], [4,9,2]]		    # magic(3)
#        y = [[1], [1], [1]]                     # [1 1 1]'
#        grad = self.lr.gradient(theta, X, y)
#        self.assertTrue(numpy.allclose(grad, answer))
