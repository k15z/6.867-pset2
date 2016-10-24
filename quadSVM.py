import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp
from cvxopt.solvers import options
options['show_progress'] = False

def linear_kernel(x_a, x_b):
    return np.dot(x_a, x_b)

def make_gaussian_kernel(gamma):
    def gaussian_kernel(x_a, x_b):
        return np.exp(-np.linalg.norm(x_a - x_b)*gamma)
    return gaussian_kernel

def make_polynomial_kernel(degree):
    def polynomial_kernel(x_a, x_b):
        return np.power(x_a.dot(x_b),degree)
    return polynomial_kernel

def gram_matrix(x, kernel):
    num_samples, num_features = x.shape
    gram = np.zeros((num_samples, num_samples))
    for i, x_i in enumerate(x):
        for j, x_j in enumerate(x):
            gram[i,j] = kernel(x_i, x_j)
    return gram

class QuadSVM:
    def __init__(self, C=1.0/1e-8, kernel=linear_kernel):
        self._C = C
        self._kernel = kernel

    def fit(self, x, y):
        assert x.shape[0] == y.shape[0]
        num_samples, num_features = x.shape

        C = self._C
        gram = gram_matrix(x, self._kernel)

        P = matrix(np.outer(y, y) * gram)
        q = matrix(-np.ones(num_samples))
        G = matrix(np.vstack((-np.eye(num_samples), np.eye(num_samples))))
        h = matrix(np.hstack((np.zeros(num_samples), np.ones(num_samples) * C)))
        A = matrix(y, (1, num_samples))
        b = matrix(0.0)
        alpha = np.array(qp(P, q, G, h, A, b)['x'])

        sv_x = []
        sv_y = []
        sv_a = []
        bias = 0.0
        for i in range(len(alpha)):
            if max(alpha)*1e-6 < alpha[i] < self._C:
                sv_x += [x[i]]
                sv_y += [y[i]]
                sv_a += [alpha[i]]

                bias += y[i]
                for j in range(len(alpha)):
                    bias -= y[j] * alpha[j,0] * gram[i,j]
        self.sv_x = sv_x = np.array(sv_x)
        self.sv_y = sv_y = np.array(sv_y)
        self.sv_a = sv_a = np.array(sv_a)
        self.bias = bias / len(sv_a)
        self.n_support_ = [len(sv_x)]

    def score(self, x, y):
        return sum(self.predict(x) * y > 0) / float(x.shape[0])

    def predict(self, x):
        num_samples, num_features = x.shape
        y = np.zeros((num_samples,1))
        for i, x_i in enumerate(x):
            for j, sv_x_i in enumerate(self.sv_x):
                y[i] += self.sv_a[j] * self.sv_y[j] * self._kernel(x_i, sv_x_i)
        return (y + self.bias).flatten()

    def predictOne(self, x):
        return self.predict(np.array([x]))
