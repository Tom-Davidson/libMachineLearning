import math
import numpy
import scipy.optimize as op

class logisticregression:
    def __init__(self):
        False

    def sigmoid(self, data):
        if isinstance(data, (int, float)):
            if isinstance(data, int):
                data = float(data)
            return 1 / (1 + math.exp(-data))
        elif isinstance(data, list):
            for element in range(len(data)):
                data[element] = self.sigmoid(data[element])
            return data
        elif isinstance(data, numpy.matrixlib.defmatrix.matrix) or isinstance(data, numpy.ndarray):
            for row in range(data.shape[0]):
                if data.ndim > 1:
                    for col in range(data.shape[1]):
                       data[row, col] = self.sigmoid(data.item(row, col))
                else:
                    data[row] = self.sigmoid(data.item(row))
            return data
        else:
            print "Unknown type '"+str(type(data))+"' in logisticregression.sigmoid()"
            return False

    def hypothesis(self, theta, X, round=True):
        theta = numpy.array(theta, dtype=float) # Recast to a numpy matrix of floats
        X = numpy.array(X, dtype=float) # Recast to a numpy matrix of floats
        if theta.shape[0] == 1: # Check which orientation theta is in
            theta = theta.T
        hox = self.sigmoid(theta.T.dot(X)).T
        if round:
            hox.round(0)
        return hox

    def initialParameters(self, X):
        X = numpy.array(X, dtype=float) # Recast to a numpy matrix of floats
        n = X.shape[1] # Number of features (n) = width of training set (X)
        if self.hasBias(X):
            n = n - 1
        return numpy.zeros(n + 1) # Include the bias parameter

    def hasBias(self, X):
        X = numpy.array(X, dtype=float) # Recast to a numpy matrix of floats
        m = X.shape[0] # Number of examples (m) = height of training set (X)
        if X[:, 0].max() == 1 and X[:, 0].min() == 1:
            # 1st col is already 1 so likely already is has the bias unit
            return True
        else:
            biasFeatures = numpy.ones((m,1))
            return False

    def addBias(self, X):
        X = numpy.array(X, dtype=float) # Recast to a numpy matrix of floats
        m = X.shape[0] # Number of examples (m) = height of training set (X)
        if self.hasBias(X):
            # 1st col is already 1 so likely already is has the bias unit
            return X
        else:
            biasFeatures = numpy.ones((m,1))
            return numpy.concatenate((biasFeatures, X), axis=1) # Add the bias units

    def cost(self, theta, X, y):
        theta = numpy.array(theta, dtype=float) # Recast to a numpy matrix of floats
        X = numpy.array(X, dtype=float) # Recast to a numpy matrix of floats
        y = numpy.array(y, dtype=float) # Recast to a numpy matrix of floats
        m = X.shape[0] # Number of examples (m) = height of training set (X)
        X = self.addBias(X)
        sum = 0.0
        for i in range(m):
            # pull out this example and turn into a vector
            trainingExample = X[i, :] # 1d numpy.ndarray
            trainingExample = numpy.expand_dims(trainingExample, axis=0) # 2d numpy.ndarray single row
            trainingExample = trainingExample.T # 2d numpy.ndarray vector
            # pull out answer into a numpy.float64 rather than 1x1 matrix
            answer = float(y[(i,0)])
            hox = self.hypothesis(theta, trainingExample)[(0,0)]
            exampleCost = ( -answer * math.log(hox) ) - ( (1-answer) * math.log(1-hox) )
            sum = sum + exampleCost
        return (1/float(m)) * sum

    def gradient(self, theta, X, y):
        theta = numpy.array(theta, dtype=float) # Recast to a numpy matrix of floats
        X = numpy.array(X, dtype=float) # Recast to a numpy matrix of floats
        y = numpy.array(y, dtype=float) # Recast to a numpy matrix of floats
        m = X.shape[0] # Number of examples (m) = height of training set (X)
        X = self.addBias(X)
        grad = numpy.zeros((theta.shape[0], 1))
        for j in range(theta.shape[0]):
            sum = 0.0
            for i in range(m):
                # pull out this example and turn into a vector
                trainingExample = X[i, :] # 1d numpy.ndarray
                trainingExample = numpy.expand_dims(trainingExample, axis=0) # 2d numpy.ndarray single row
                trainingExample = trainingExample.T # 2d numpy.ndarray vector
                # pull out answer into a numpy.float64 rather than 1x1 matrix
                answer = float(y[(i,0)])
                hox = self.hypothesis(theta, trainingExample)[(0,0)]
                sum = sum + ( (hox - answer) * X[i, j] )
            grad[j] = (1/float(m)) * sum
        return grad

    def optimiseParameters(self, X, y):
        X = numpy.array(X, dtype=float) # Recast to a numpy array of floats
        y = numpy.array(y, dtype=float) # Recast to a numpy array of floats
        #X = numpy.array([[1,2,3],[1,3,4]]);
        #y = numpy.array([[1],[0]]);
        X = self.addBias(X)
        initialTheta = self.initialParameters(X)
        result = op.minimize(
            fun = self.cost,
            x0 = initialTheta,
            args = (X, y),
            method = 'TNC',
            jac = self.gradient
        )
        return result.x
