# -*- coding: utf-8 -*-
"""
Homework 2: Question 2, Part 2
===============================================================================
Test your implementation on the 2D datasets. Set C=1 and report/explain your 
decision boundary and classification error rate on the training and validation 
sets. We provide some skeleton code in svm test.py.
"""

import numpy as np
import pylab as pl
from cvxopt import matrix
from sklearn.svm import SVC
from cvxopt.solvers import qp
from cvxopt.solvers import options
from plotBoundary import plotDecisionBoundary
options['show_progress'] = False

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

class quadSVM:
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

                bias += y[i,0]
                for j in range(len(alpha)):
                    bias -= y[j,0] * alpha[j,0] * gram[i,j]
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

dataset_id = "1"
train = np.loadtxt('data/data' + dataset_id + '_train.csv')
x_train, y_train = train[:,0:2], train[:,2:3]
test = np.loadtxt('data/data' + dataset_id + '_test.csv')
x_test, y_test = test[:,0:2], test[:,2:3]
val = np.loadtxt('data/data' + dataset_id + '_validate.csv')
x_val, y_val = val[:,0:2], val[:,2:3]

svm = quadSVM(C=1.0, kernel=make_gaussian_kernel(1.0))
svm.fit(x_train, y_train)
print("quadSVM val", 1.0 - svm.score(x_val, y_val))
print("quadSVM test", 1.0 - svm.score(x_test, y_test))
plotDecisionBoundary(x_train, y_train, svm.predictOne, [0.0], title = 'quadSVM')

clf = SVC(C=1.0, kernel='rbf', gamma=1.0)
clf.fit(x_train, y_train)
def predictOne(x_i):
    return clf.decision_function(np.array([x_i]))
print("sklearnSVM val", 1.0 - svm.score(x_val, y_val))
print("sklearnSVM test", 1.0 - svm.score(x_test, y_test))
plotDecisionBoundary(x_train, y_train, predictOne, [0.0], title = 'sklearnSVM')
