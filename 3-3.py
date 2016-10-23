# -*- coding: utf-8 -*-
"""
Homework 2: Question 3, Part 3
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

class pegasosSVM:
    def __init__(self, L=1.0, kernel=linear_kernel):
        self._L = L
        self._kernel = kernel

    def fit(self, x, y):
        assert x.shape[0] == y.shape[0]
        num_samples, num_features = x.shape

        L = self._L
        gram = gram_matrix(x, self._kernel)
        
        t = 0.0
        alpha = np.zeros(num_samples)
        for epoch in range(1000):
            for i in range(num_samples):
                t += 1.0
                lr = 1.0 / (t * L)
                if y[i,0] * alpha.dot(gram[:,i]) < 1.0:
                    alpha[i] = (1.0 - lr * L) * alpha[i] + lr * y[i,0]
                else:
                    alpha[i] = (1.0 - lr * L) * alpha[i]
        self.sv_x = x
        self.sv_y = y
        self.sv_a = alpha
        
        print("sv", sum(abs(alpha) > max(alpha)*1e-6))

    def score(self, x, y):
        return sum(self.predict(x) * y > 0) / float(x.shape[0])

    def predict(self, x):
        num_samples, num_features = x.shape
        y = np.zeros((num_samples,1))
        for i, x_i in enumerate(x):
            for j, sv_x_i in enumerate(self.sv_x):
                y[i] += self.sv_a[j] * self._kernel(x_i, sv_x_i)
        return y
        
    def predictOne(self, x):
        return self.predict(np.array([x]))

dataset_id = "1"
train = np.loadtxt('data/data' + dataset_id + '_train.csv')
x_train, y_train = train[:,0:2], train[:,2:3]
test = np.loadtxt('data/data' + dataset_id + '_test.csv')
x_test, y_test = test[:,0:2], test[:,2:3]
val = np.loadtxt('data/data' + dataset_id + '_validate.csv')
x_val, y_val = val[:,0:2], val[:,2:3]

x_axis = [2**i for i in range(-5,5+1)]
y_axis = []
for gamma in x_axis:
#    svm = pegasosSVM(L=0.02, kernel=make_gaussian_kernel(gamma))
    svm = SVC(C=0.1,gamma=gamma)
    svm.fit(x_train, y_train.flatten())
    print(sum(svm.n_support_))
    print("pegasosSVM val", 1.0 - svm.score(x_val, y_val))
    print("pegasosSVM test", 1.0 - svm.score(x_test, y_test))
    print("")
#    plotDecisionBoundary(x_train, y_train, svm.predictOne, [-1,0,1], title = 'pegasosSVM - ' + str(gamma))

pl.show()
