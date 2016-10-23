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

import numpy as np
import pylab as pl
from cvxopt import matrix
from quadSVM import QuadSVM
from sklearn.svm import SVC
from cvxopt.solvers import qp
from cvxopt.solvers import options
from plotBoundary import plotDecisionBoundary
options['show_progress'] = False

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
C = 1000.0
svm = QuadSVM(C=C)
svm.fit(x, y.flatten())
print("a", svm.predict(x))
plotDecisionBoundary(x, y, svm.predictOne, [0.0], title = 'quadSVM')

clf = SVC(C=C, kernel='linear')
clf.fit(x, y.flatten())
def predictOne(x_i):
    return clf.decision_function(np.array([x_i]))
print("b", clf.decision_function(x))
plotDecisionBoundary(x, y, predictOne, [0.0], title = 'sklearnSVM')
pl.show()
