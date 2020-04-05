import numpy as np


def relu(X: np.ndarray, derivative: bool = False):
    if derivative:
        return (X > 0).astype(int)
    X[X < 0] = 0
    return X


def linear(X: np.ndarray, derivative: bool = False):
    if derivative:
        return np.ones(X.shape)
    return X


def sigmoid(X: np.ndarray, derivative: bool = False):
    if derivative:
        return sigmoid(X) * (1 - sigmoid(X))

    return 1 / (1 + np.exp(-X))
