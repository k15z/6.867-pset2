import numpy as np

def linear_kernel(x_a, x_b):
    return np.dot(x_a, x_b)

def make_gaussian_kernel(gamma):
    def gaussian_kernel(x_a, x_b):
        return np.exp(-np.linalg.norm(x_a - x_b)*gamma)
    return gaussian_kernel

def make_polynomial_kernel(degree):
    def polynomial_kernel(x_a, x_b):
        return np.power(x_a.dot(x_b),degree)
    return polynomial_kernel

def gram_matrix(x, kernel):
    num_samples, num_features = x.shape
    gram = np.zeros((num_samples, num_samples))
    for i, x_i in enumerate(x):
        for j, x_j in enumerate(x):
            gram[i,j] = kernel(x_i, x_j)
    return gram

class PegasosSVM:
    def __init__(self, L=1.0, kernel=linear_kernel):
        self._L = L
        self._kernel = kernel

    def fit(self, x, y):
        assert x.shape[0] == y.shape[0]
        num_samples, num_features = x.shape

        L = self._L
        gram = gram_matrix(x, self._kernel)
        
        t = 0.0
        alpha = np.zeros(num_samples)
        for epoch in range(20):
            for i in range(num_samples):
                t += 1.0
                lr = 1.0 / (t * L)
                if y[i] * alpha.dot(gram[:,i]) < 1.0:
                    alpha[i] = (1.0 - lr * L) * alpha[i] + lr * y[i]
                else:
                    alpha[i] = (1.0 - lr * L) * alpha[i]
        self.sv_x = x
        self.sv_y = y
        self.sv_a = alpha
        self.n_support_ = [sum(abs(alpha) > max(alpha)*1e-6)]

    def score(self, x, y):
        return sum(self.predict(x) * y > 0) / float(x.shape[0])

    def predict(self, x):
        num_samples, num_features = x.shape
        y = np.zeros(num_samples)
        for i, x_i in enumerate(x):
            for j, sv_x_i in enumerate(self.sv_x):
                y[i] += self.sv_a[j] * self._kernel(x_i, sv_x_i)
        return y
        
    def predictOne(self, x):
        return self.predict(np.array([x]))
