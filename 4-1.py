# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from quadSVM import QuadSVM, make_gaussian_kernel
from sklearn.linear_model import LogisticRegression

def load_data(pos, neg, normalize=False):
    pos_x = [np.loadtxt('data/mnist_digit_' + i + '.csv') for i in pos]
    neg_x = [np.loadtxt('data/mnist_digit_' + i + '.csv') for i in neg]
    
    pos_x_train = [arr[0:200,:] for arr in pos_x]
    pos_x_val = [arr[200:350,:] for arr in pos_x]
    pos_x_test = [arr[350:500,:] for arr in pos_x]
    neg_x_train = [arr[0:200,:] for arr in neg_x]
    neg_x_val = [arr[200:350,:] for arr in neg_x]
    neg_x_test = [arr[350:500,:] for arr in neg_x]
    
    x_train = np.vstack(pos_x_train + neg_x_train)
    y_train = np.hstack((
        np.ones(200*len(pos_x_train)), -np.ones(200*len(neg_x_train))
    ))
    
    x_val = np.vstack(pos_x_val + neg_x_val)
    y_val = np.hstack((
        np.ones(150*len(pos_x_val)), -np.ones(150*len(neg_x_val))
    ))
    
    x_test = np.vstack(pos_x_test + neg_x_test)
    y_test = np.hstack((
        np.ones(150*len(pos_x_test)), -np.ones(150*len(neg_x_test))
    ))
    
    if normalize:
        x_train = x_train * 2.0 / 255.0 - 1.0
        x_val = x_val * 2.0 / 255.0 - 1.0
        x_test = x_test * 2.0 / 255.0 - 1.0
    
    return (x_train, y_train, x_val, y_val, x_test, y_test)

x_train, y_train, x_val, y_val, x_test, y_test = load_data("02468","13579")

# sklearn.LR
lr = LogisticRegression(C=1.0)
lr.fit(x_train, y_train)
print("LR Train", lr.score(x_train, y_train))
print("LR Test", lr.score(x_test, y_test))
print("LR Val", lr.score(x_val, y_val))
print("")

# sklearn.SVM
svm = SVC(C=1.0, kernel='linear')
svm.fit(x_train, y_train)
def predictOne(x_i):
    return svm.decision_function(np.array([x_i]))
print("SVM Train", svm.score(x_train, y_train))
print("SVM Test", svm.score(x_test, y_test))
print("SVM Val", svm.score(x_val, y_val))
print("")

# kf.QuadSVM
svm = QuadSVM(C=1.0, kernel=make_gaussian_kernel(100.0))
svm.fit(x_train, y_train)
print("SVM Train", svm.score(x_train, y_train))
print("SVM Test", svm.score(x_test, y_test))
print("SVM Val", svm.score(x_val, y_val))
print("")

"""
result = svm.predict(x_test) * y_test > 0
for i in range(len(result)):
    if not result[i]:
        image = x_test[i,:].reshape(28,28)
        plt.figure(1, figsize=(3, 3))
        if y_test[i] > 0:
            plt.title("Positive")
        else:
            plt.title("Negative")
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.show()
"""
