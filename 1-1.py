# -*- coding: utf-8 -*-
"""
Homework 2: Question 1, Part 1
===============================================================================
Use a gradient descent method to optimize the logistic regression objective, 
with L2 regularization on the weight vector. Run your code on data1_train.csv 
with λ = 0. What happens to the weight vector as a function of the number of 
iteration of gradient descent? What happens when λ = 1? Explain
"""

from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# load data from csv files
dataset_id = "1"
train = np.loadtxt('data/data' + dataset_id + '_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Train with regularization for fixed iterations
def train_model(my_lambda, my_iter):
    lr = LogisticRegression(penalty='l2',C=1.0/my_lambda, solver='sag', max_iter=my_iter)
    lr.fit(X, Y.flatten())
    assert lr.n_iter_ <= my_iter
    return lr.coef_

# Plot norm of weight vector vs number of iterations
x = []
y1 = []
y2 = []
for x_i in range(1, 50):
    x += [x_i]
    y1 += [np.linalg.norm(train_model(1e-50, x_i))]
    y2 += [np.linalg.norm(train_model(1.0, x_i))]
plt.figure()
plt.plot(x, y1, label=r"$\lambda = 0$")
plt.plot(x, y2, label=r"$\lambda = 1$")
plt.xlabel('# iteration')
plt.ylabel('norm of weights')
plt.legend(loc="center right")
plt.tight_layout()
plt.grid(True)
plt.show()
