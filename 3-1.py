# -*- coding: utf-8 -*-
"""
Homework 2: Question 3, Part 1
"""

import numpy as np
import pylab as plt
from sklearn.svm import SVC
from plotBoundary import plotDecisionBoundary

class pLSVM:
    MAX_EPOCH = 100
    
    def __init__(self, L):
        self._L = L

    def fit(self, x, y):
        assert x.shape[0] == y.shape[0]
        x = np.hstack((np.ones((x.shape[0],1)),x.copy()))
        num_samples, num_features = x.shape

        t = 0.0
        w = np.zeros(num_features)
        
        L = self._L
        for epoch in range(pLSVM.MAX_EPOCH):
            for i in range(num_samples):
                t = t + 1.0
                n = 1.0 / (t * L)
                
                fake_w = w.copy()
                fake_w[0] = 0.0
                if y[i,0] * w.dot(x[i]) < 1.0:
                    w = w - n*L*fake_w + n*y[i,0]*x[i]
                else:
                    w = w - n*L*fake_w
        self.w = w
        print(w)
        return 1.0 / np.linalg.norm(self.w)

    def score(self, x, y):
        return sum(self.predict(x) * y > 0) / float(x.shape[0])

    def predict(self, x):
        x = np.hstack((np.ones((x.shape[0],1)),x.copy()))
        num_samples, num_features = x.shape
        y = np.zeros((num_samples,1))
        for i, x_i in enumerate(x):
            y[i] = self.w.dot(x_i)
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

x_axis = [2 * 10**-i for i in range(1,11)]
y_axis = []
for L in x_axis:
    svm = pLSVM(L=L)
    y_axis += [svm.fit(x_train, y_train)]
    print("pLSVM train", 1.0 - svm.score(x_train, y_train))
    print("pLSVM val", 1.0 - svm.score(x_val, y_val))
    print("pLSVM test", 1.0 - svm.score(x_test, y_test))
    print("")
    plotDecisionBoundary(x_train, y_train, svm.predictOne, [-1,0,1], title = 'pLSVM')

plt.figure()
plt.plot(x_axis, y_axis)
plt.xlabel(r"$\lambda$")
plt.ylabel('geometric margin')
plt.xlim(0,0.2)
plt.tight_layout()
plt.show()
