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
from quadSVM import QuadSVM, make_gaussian_kernel
options['show_progress'] = False

dataset_id = "2"
train = np.loadtxt('data/data' + dataset_id + '_train.csv')
x_train, y_train = train[:,0:2], train[:,2:3]
test = np.loadtxt('data/data' + dataset_id + '_test.csv')
x_test, y_test = test[:,0:2], test[:,2:3]
val = np.loadtxt('data/data' + dataset_id + '_validate.csv')
x_val, y_val = val[:,0:2], val[:,2:3]

C = 1.0
svm = QuadSVM(C=C, kernel=make_gaussian_kernel(1.0))
svm.fit(x_train, y_train.flatten())
print("quadSVM train", 1.0 - svm.score(x_train, y_train.flatten()))
print("quadSVM val", 1.0 - svm.score(x_val, y_val.flatten()))
print("quadSVM test", 1.0 - svm.score(x_test, y_test.flatten()))
plotDecisionBoundary(x_train, y_train, svm.predictOne, [-1, 0, 1], title = 'quadSVM')

clf = SVC(C=C, kernel='rbf', gamma=1.0)
clf.fit(x_train, y_train.flatten())
def predictOne(x_i):
    return clf.decision_function(np.array([x_i]))
print("sklearnSVM train", 1.0 - clf.score(x_train, y_train.flatten()))
print("sklearnSVM val", 1.0 - clf.score(x_val, y_val.flatten()))
print("sklearnSVM test", 1.0 - clf.score(x_test, y_test.flatten()))
plotDecisionBoundary(x_train, y_train, predictOne, [-1, 0, 1], title = 'sklearnSVM')

pl.show()
