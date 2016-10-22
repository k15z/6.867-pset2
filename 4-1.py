# -*- coding: utf-8 -*-
import numpy as np
from quadSVM import QuadSVM
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

digit_1 = np.loadtxt('data/mnist_digit_1.csv')
digit_7 = np.loadtxt('data/mnist_digit_7.csv')

x_train = np.vstack((digit_1[0:200,:], digit_7[0:200,:]))
y_train = np.hstack((np.ones(200), -np.ones(200)))

x_val = np.vstack((digit_1[200:350,:], digit_7[200:350,:]))
y_val = np.hstack((np.ones(150), -np.ones(150)))

x_test = np.vstack((digit_1[350:500,:], digit_7[350:500,:]))
y_test = np.hstack((np.ones(150), -np.ones(150)))

# sklearn.LR
lr = LogisticRegression(C=1.0)
lr.fit(x_train, y_train)
print("LR Train", lr.score(x_train, y_train))
print("LR Test", lr.score(x_test, y_test))
print("LR Val", lr.score(x_val, y_val))
print()

# sklearn.SVM
svm = SVC(C=1.0, kernel='linear')
svm.fit(x_train, y_train)
def predictOne(x_i):
    return svm.decision_function(np.array([x_i]))
print("SVM Train", svm.score(x_train, y_train))
print("SVM Test", svm.score(x_test, y_test))
print("SVM Val", svm.score(x_val, y_val))
print()

# kf.QuadSVM
svm = QuadSVM(C=0.001)
svm.fit(x_train, y_train)
print("SVM Train", svm.score(x_train, y_train))
print("SVM Test", svm.score(x_test, y_test))
print("SVM Val", svm.score(x_val, y_val))
print()
