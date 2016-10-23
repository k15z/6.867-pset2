# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from quadSVM import QuadSVM, make_gaussian_kernel
from sklearn.linear_model import LogisticRegression

digit_4 = np.loadtxt('data/mnist_digit_4.csv')
digit_9 = np.loadtxt('data/mnist_digit_9.csv')

x_train = np.vstack((digit_4[0:200,:], digit_9[0:200,:]))
y_train = np.hstack((np.ones(200), -np.ones(200)))

x_val = np.vstack((digit_4[200:350,:], digit_9[200:350,:]))
y_val = np.hstack((np.ones(150), -np.ones(150)))

x_test = np.vstack((digit_4[350:500,:], digit_9[350:500,:]))
y_test = np.hstack((np.ones(150), -np.ones(150)))

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


result = svm.predict(x_test) * y_test > 0
for i in range(len(result)):
    if not result[i]:
        image = x_test[i,:].reshape(28,28)
        plt.figure(1, figsize=(3, 3))
        if y_test[i] > 0:
            plt.title("4")
        else:
            plt.title("9")
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.show()
