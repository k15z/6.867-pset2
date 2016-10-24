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

x_train, y_train, x_val, y_val, x_test, y_test = load_data("1","7", True)

dx = 10
dy = 10
cs = np.linspace(1e-8, 20.0, num=dx)
gammas = np.linspace(1e-8, 20.0, num=dy)
train_acc = np.zeros((dy, dx))
test_acc = np.zeros((dy, dx))
val_acc = np.zeros((dy, dx))

for i, C in enumerate(cs):
    for j, gamma in enumerate(gammas):
        svm = QuadSVM(C=C, kernel=make_gaussian_kernel(gamma))
        svm.fit(x_train, y_train)

        train_acc[j][i] = svm.score(x_train, y_train)
        test_acc[j][i] = svm.score(x_test, y_test)
        val_acc[j][i] = svm.score(x_val, y_val)

        print(i, j, C, gamma)
        print("SVM Train", svm.score(x_train, y_train))
        print("SVM Test", svm.score(x_test, y_test))
        print("SVM Val", svm.score(x_val, y_val))
        print("")

plt.figure()
cs, gammas = np.meshgrid(cs, gammas)
cs = plt.contourf(cs, gammas, test_acc)
plt.xlabel("C")
plt.ylabel("gamma")
plt.colorbar(cs)
plt.show()

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
