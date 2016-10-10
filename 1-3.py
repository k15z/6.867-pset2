"""
Homework 2: Question 1, Part 3
===============================================================================
Use the training and validation sets to pick the best regularizer and value of 
λ for each data set: data1, data2, data3, data4. Report the performance on the 
test sets.
"""

from __future__ import unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# load data from csv files
dataset_id = "1"

train = np.loadtxt('data/data' + dataset_id + '_train.csv')
x_train, y_train = train[:,0:2], train[:,2:3]

test = np.loadtxt('data/data' + dataset_id + '_test.csv')
x_test, y_test = test[:,0:2], test[:,2:3]

val = np.loadtxt('data/data' + dataset_id + '_validate.csv')
x_val, y_val = val[:,0:2], val[:,2:3]

# Train with regularization for fixed iterations
def evaluate_model(penalty, my_lambda):
    lr = LogisticRegression(penalty=penalty, C=1.0/my_lambda)
    lr.fit(x_train, y_train.flatten())
    err_train = 1.0 - lr.score(x_train, y_train)
    err_test = 1.0 - lr.score(x_test, y_test)
    err_val = 1.0 - lr.score(x_val, y_val)
    return (err_train, err_test, err_val)

# Plot norm of weight vector vs number of iterations
x = []
err_trains_l1 = []
err_tests_l1 = []
err_vals_l1 = []
err_trains_l2 = []
err_tests_l2 = []
err_vals_l2 = []
for my_lambda in np.linspace(1e-50, 20.0, num=100):
    x += [my_lambda]
    err_train, err_test, err_val = evaluate_model('l1', my_lambda)
    err_trains_l1 += [err_train]
    err_tests_l1 += [err_test]
    err_vals_l1 += [err_val]
    err_train, err_test, err_val = evaluate_model('l2', my_lambda)
    err_trains_l2 += [err_train]
    err_tests_l2 += [err_test]
    err_vals_l2 += [err_val]

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
#plt.ylim([-0.05, 0.6])
plt.show()
