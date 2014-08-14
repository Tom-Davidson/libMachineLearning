import numpy as np
import scipy.optimize as op

class lrCopyPaste:

    def Sigmoid(self, z):
        return 1/(1 + np.exp(-z));


    def op(self):
        # intialize X and y
        X = np.array([[1,2,3],[1,3,4]]);
        y = np.array([[1],[0]]);

        m , n = X.shape;
        initial_theta = np.zeros(n)
        Result = op.minimize(fun = self.CostFunc, x0 = initial_theta, args = (X, y), method = 'TNC',
                                         jac = self.Gradient);
        optimal_theta = Result.x

        print optimal_theta



    def Gradient(self, theta,x,y):
        m , n = x.shape
        theta = theta.reshape((n,1));
        y = y.reshape((m,1))
        sigmoid_x_theta = self.Sigmoid(x.dot(theta));
        grad = ((x.T).dot(sigmoid_x_theta-y))/m;
        return grad.flatten();


    def CostFunc(self, theta,x,y):
        m,n = x.shape;
        theta = theta.reshape((n,1));
        y = y.reshape((m,1));
        term1 = np.log(self.Sigmoid(x.dot(theta)));
        term2 = np.log(1-self.Sigmoid(x.dot(theta)));
        term1 = term1.reshape((m,1))
        term2 = term2.reshape((m,1))
        term = y * term1 + (1 - y) * term2;
        J = -((np.sum(term))/m);
        return J;


lr = lrCopyPaste()
lr.op()