import numpy as np


class Activation:
    def __call__(self, X):
        pass

    def derivative(self, X):
        pass


class SigmoidActivation(Activation):
    def __call__(self, X):
        return 1 / (1 + np.exp(-X))

    def derivative(self, X):
        sigmoid = self.__call__(X)
        return sigmoid * (1 - sigmoid)


class ReLUActivation(Activation):
    def __call__(self, X):
        return np.maximum(0, X)

    def derivative(self, X):
        return np.where(X > 0, 1, 0)


class TanhActivation(Activation):
    def __call__(self, X):
        return np.tanh(X)

    def derivative(self, X):
        return 1 - np.tanh(X) ** 2


class SoftmaxActivation(Activation):
    def __call__(self, X):
        exp_vals = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def derivative(self, X):
        softmax_output = self.__call__(X)
        return softmax_output * (1 - softmax_output)
