# -*- coding: utf-8 -*-
"""
Homework 2: Question 1, Part 2
===============================================================================
Evaluate the effect of the choice of regularizer (L1 vs L2) and the value of λ 
on (a) the weights, (b) the decision boundary and (c) the classification error 
rate in each of the training data sets.
"""

from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from plotBoundary import plotDecisionBoundary
from sklearn.linear_model import LogisticRegression

# load data from csv files
dataset_id = "3"

train = np.loadtxt('data/data' + dataset_id + '_train.csv')
x_train, y_train = train[:,0:2], train[:,2:3]

test = np.loadtxt('data/data' + dataset_id + '_test.csv')
x_test, y_test = test[:,0:2], test[:,2:3]

val = np.loadtxt('data/data' + dataset_id + '_validate.csv')
x_val, y_val = val[:,0:2], val[:,2:3]

# Train with regularization for fixed iterations
def evaluate_model(penalty, my_lambda):
    lr = LogisticRegression(penalty=penalty, C=1.0/my_lambda, intercept_scaling=1e3)
    lr.fit(x_train, y_train.flatten())
    err_train = 1.0 - lr.score(x_train, y_train)
    err_test = 1.0 - lr.score(x_test, y_test)
    err_val = 1.0 - lr.score(x_val, y_val)
    def predictLR(x):
        return lr.predict(np.array([x]))
    return (lr.coef_, predictLR, err_train, err_test, err_val)

# Plot norm of weight vector vs number of iterations
x = []
norm_weights_l1=[]
err_trains_l1 = []
err_tests_l1 = []
err_vals_l1 = []
norm_weights_l2=[]
err_trains_l2 = []
err_tests_l2 = []
err_vals_l2 = []
for my_lambda in np.linspace(1e-50, 20.0, num=41):
    x += [my_lambda]
    
    weights, predictLR, err_train, err_test, err_val = evaluate_model('l1', my_lambda)
    norm_weights_l1 += [np.linalg.norm(weights)]
    err_trains_l1 += [err_train]
    err_tests_l1 += [err_test]
    err_vals_l1 += [err_val]
    print("L1", my_lambda, err_train, err_test, err_val)
#    plotDecisionBoundary(x_train, y_train, predictLR, [0.5], title = 'L1, $\lambda = ' + str(my_lambda) + '$')
    
    weights, predictLR, err_train, err_test, err_val = evaluate_model('l2', my_lambda)
    norm_weights_l2 += [np.linalg.norm(weights)]
    err_trains_l2 += [err_train]
    err_tests_l2 += [err_test]
    err_vals_l2 += [err_val]
    print("L2", my_lambda, err_train, err_test, err_val)
#    plotDecisionBoundary(x_train, y_train, predictLR, [0.5], title = 'L2, $\lambda = ' + str(my_lambda) + '$')

plt.figure()
plt.title("Data Set " + dataset_id)
plt.plot(x, norm_weights_l1, label="L1")
plt.plot(x, norm_weights_l2, label="L2")
plt.xlabel('lambda')
plt.ylabel('norm of weights')
plt.legend(loc="bottom right")
plt.tight_layout()
plt.grid(True)

plt.figure()
plt.title("Data Set " + dataset_id)
#plt.plot(x, err_trains_l1, label="Train (L1)")
plt.plot(x, err_tests_l1, label="Testing (L1)")
plt.plot(x, err_vals_l1, label="Validation (L1)")
#plt.plot(x, err_trains_l2, label="Train (L2)")
plt.plot(x, err_tests_l2, label="Testing (L2)")
plt.plot(x, err_vals_l2, label="Validation (L2)")
plt.xlabel('lambda')
plt.ylabel('error')
plt.legend(loc="bottom right")
plt.tight_layout()
plt.grid(True)

plt.show()
