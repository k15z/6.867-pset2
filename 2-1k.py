# -*- coding: utf-8 -*-
"""
Homework 2: Question 2, Part 1
===============================================================================
Implement the dual form of linear SVMs with slack variables. Please do not use 
the built-in SVM implementation in Matlab or sklearn. Instead, write a program 
that takes data as input, converts it to the appropriate objective function and 
constraints, and then calls a quadratic programming package to solve it.

Show in your report the constraints and objective that you generate for the 2D 
problem with positive examples (2, 2), (2, 3) and negative examples (0, -1), 
(-3, -2). Which examples are support vectors?
"""

import cvxopt
import numpy as np
from plotBoundary import plotDecisionBoundary

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
        
        weight = np.sum(x * (alpha * y), axis=0)
        self.weight = weight
        
        # TODO: Fix the bias term...
        bias = y - self._gram_matrix(x).dot(alpha * y)
        bias = np.sum(bias) / (num_samples)
        self.bias = bias
        
        return (alpha, weight)

    def predict(self, x):
        return x.dot(weight) + self.bias

    def predictOne(self, x):
        return self.predict(np.array([x]))

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
    [2.0, 2.0], # SV1
    [2.0, 3.0], # SV2
    [0.0, -1.0], # irrelevant
    [-3.0, -2.0]  # irrelevant
])
y = np.array([
    [1.0],
    [1.0],
    [-1.0],
    [-1.0]
])
L = 1e-10
alpha, weight = svm.fit(x, y, 1/L)

print(svm.predict(x))
plotDecisionBoundary(x, y, svm.predictOne, [0], title = 'quadSVM')
