# -*- coding: utf-8 -*-
import time
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from quadSVM import QuadSVM
from pegasosSVM import PegasosSVM
from pegasosSVM import make_gaussian_kernel

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

x = []
y_quad = []
y_pegasos = []
for pos, neg in [("1","7"),("02","13"),("024","135"),("0246","1357"),("02468","13579"),("012468","123579")]:
    gamma = 1.0
    
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(pos, neg, True)
    x += [x_train.shape[0]//2]

    start = time.time()
    svm = QuadSVM(C=1.0, kernel=make_gaussian_kernel(gamma))
    svm.fit(x_train, y_train)
    y_quad += [time.time() - start]
    print(svm.score(x_train, y_train), svm.score(x_val, y_val), svm.score(x_test, y_test))

    start = time.time()
    svm = PegasosSVM(L=1.0, kernel=make_gaussian_kernel(gamma))
    svm.fit(x_train, y_train)
    y_pegasos += [time.time() - start]
    print(svm.score(x_train, y_train), svm.score(x_val, y_val), svm.score(x_test, y_test))
    
    print("")

plt.figure()
plt.plot(x, y_quad, label="QP (cvxopt)")
plt.plot(x, y_pegasos, label="Pegasos")
plt.ylabel("time (s)")
plt.xlabel("# data points")
plt.legend(loc="top right")
plt.show()
