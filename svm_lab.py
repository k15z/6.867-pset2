"""
This is my implementation of a QP-powered SVM. It hasn't been tested but seems
to be mathematically sound.
"""
import cvxopt
import numpy as np

class quadSVM:
    def fit(self, x, y, C=0):
        num_samples, input_dims = x.shape
        assert y.shape[0] == num_samples
        
        P = cvxopt.matrix(np.outer(y, y) * self._gram_matrix(x))
        q = cvxopt.matrix(-np.ones(num_samples))
        G = cvxopt.matrix(np.vstack((-np.eye(num_samples), np.eye(num_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(num_samples), np.ones(num_samples) * C)))
        A = cvxopt.matrix(y, (1, num_samples))
        b = cvxopt.matrix(0.0)
        
        alpha = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])
        weight = np.sum(x * (np.array(alpha) * y), axis=0)
        return (alpha, weight)

    def predict(self, x):
        raise NotImplementedError()

    def _kernel(self, x1, x2):
        return np.dot(x1, x2)
    
    def _gram_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        return K

svm = quadSVM()
x = np.array([
    [0.0, 1.0], # SV1
    [1.0, 1.0], # SV2
    [1.0, 2.0], # irrelevant
    [1.0, 3.0]  # irrelevant
])
y = np.array([
    [-1.0],
    [1.0],
    [1.0],
    [1.0]
])

alpha = svm.fit(x, y, 0)
print(alpha)
