# -*- coding: utf-8 -*-
"""
Homework 2: Question 3, Part 1
===============================================================================
Implement a kernelized version of the Pegasos algorithm. It should take in a 
Gram matrix and should should output the support vector values, α, or a function
that makes a prediction for a new input. In this version, you do not need to 
add a bias term.
"""

import numpy as np
import pylab as pl
from sklearn.svm import SVC
from plotBoundary import plotDecisionBoundary

def linear_kernel(x_a, x_b):
    return np.dot(x_a, x_b)

def make_gaussian_kernel(sigma):
    def gaussian_kernel(x_a, x_b):
        return np.exp(-np.linalg.norm(x_a - x_b)/(2.0 * sigma**2))
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

class pegasosSVM:
    def __init__(self, C=1.0/1e-8, kernel=linear_kernel):
        self._C = C
        self._kernel = kernel

    def fit(self, x, y):
        assert x.shape[0] == y.shape[0]
        num_samples, num_features = x.shape

        C = self._C
        gram = gram_matrix(x, self._kernel)
        
        t = 0.0
        alpha = np.zeros(num_samples)
        for epoch in range(1000000):
            for i in range(num_samples):
                t += 1.0
                lr = C / t
                if y[i,0] * alpha.dot(gram[:,i]) < 1.0:
                    alpha[i] = (1.0 - lr / C) * alpha[i] + lr * y[i,0]
                else:
                    alpha[i] = (1.0 - lr / C) * alpha[i]
        print(alpha)
        # [[  1.53846150e-01][  3.16686148e-09][  1.53846153e-01][  2.94624292e-10]]

        sv_x = []
        sv_y = []
        sv_a = []
        bias = 0.0
        for i in range(len(alpha)):
            if True or max(alpha)*1e-6 < alpha[i] < self._C:
                sv_x += [x[i]]
                sv_y += [y[i]]
                sv_a += [alpha[i]]

                bias += y[i,0]
                for j in range(len(alpha)):
                    bias -= y[j,0] * alpha[j] * gram[i,j]
        self.sv_x = sv_x = np.array(sv_x)
        self.sv_y = sv_y = np.array(sv_y)
        self.sv_a = sv_a = np.array(sv_a)
        self.bias = bias / len(sv_a)
        print(len(sv_x))

    def score(self, x, y):
        return sum(self.predict(x) * y > 0) / float(x.shape[0])

    def predict(self, x):
        num_samples, num_features = x.shape
        y = np.zeros((num_samples,1))
        for i, x_i in enumerate(x):
            for j, sv_x_i in enumerate(self.sv_x):
                y[i] += self.sv_a[j] * self.sv_y[j] * self._kernel(x_i, sv_x_i)
        return y + self.bias

    def predictOne(self, x):
        return self.predict(np.array([x]))

x = np.array([
    (2, 2),
    (2, 3),
    (0, -1),
    (-3, -2)
])
y = np.array([
    [1.0],
    [1.0],
    [-1.0],
    [-1.0]
])
C = 1.0
svm = pegasosSVM(C=C)
svm.fit(x, y)
print("b", svm.predict(x))
plotDecisionBoundary(x, y, svm.predictOne, [0.0], title = 'quadSVM')

clf = SVC(C=C, kernel='linear')
clf.fit(x, y.flatten())
def predictOne(x_i):
    return clf.decision_function(np.array([x_i]))
plotDecisionBoundary(x, y, predictOne, [0.0], title = 'sklearnSVM')
pl.show()
