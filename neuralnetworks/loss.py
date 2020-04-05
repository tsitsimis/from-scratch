from functools import partial

import numpy as np

from . import nn


def softmax(X: np.ndarray) -> np.ndarray:
    a = X - np.max(X, axis=1)[:, None]  # to avoid overflow in exp
    a = np.exp(a)
    p = a / a.sum(axis=1)[:, None]
    return p


def crossentropy(y1: np.ndarray, y2: np.ndarray, derivative: bool = False, y2_normalized: bool = False) -> float:
    """
    Cross-entropy loss. It normalizes predictions with the softmax function if necessary

    Parameters
    ----------
    y1 : numpy.ndarray
       2D array with true targets/labels

    y2 : numpy.ndarray
       2D array with predictions

    derivative : bool
        If True it returns the derivative of the loss

    y2_normalized : bool
        Indicates if the predictions are normalized (sum to one)
    """

    if y2_normalized:
        p = y2
    else:
        p = softmax(y2)
    if derivative:
        return p - y1
    return np.mean(np.sum(y1*np.log(1e-15 + p), axis=1))


def mse(y: np.ndarray, a: np.ndarray, derivative: bool = False):
    """
    Mean Square Error

    Parameters
    ----------
    y : numpy.ndarray
        2D array with true targets/labels

    a : numpy.ndarray
        2D array with predictions

    derivative : bool
        If True it returns the derivative of the loss
    """

    if len(y.shape) == 1:
        y = y[:, None]

    if len(a.shape) == 1:
        a = a[:, None]

    if derivative:
        return a - y

    return (1 / 2) * (a - y).T * (a - y)


class Loss:
    """
    Encapsulates a neural network and a loss function in order to
    calculate gradients and perform back-propagation

    Methods
    -------
    backward(X, y)
        Calls the `backward` method of each layer of the network
        by passing the accumulated gradient from the last layer
        to the first
    """

    def __init__(self, net: nn.NN, cost: callable):
        self.cost = cost
        self.cost_d = partial(self.cost, derivative=True)
        self.net = net

    def backward(self, X: np.ndarray, y: np.ndarray):
        """
        Calls the `backward` method of each layer of the network
        by passing the accumulated gradient from the last layer
        to the first

        Parameters
        ----------
        X : numpy.ndarray
            2D array containing input samples

        y : numpy.ndarray
            1D or 2D array in case of multiple outputs/labels
        """

        net = self.net
        L = net.n_layers - 1

        cost_grad = self.cost_d(y, net.forward(X))
        net.layers[L].backward(cost_grad, net.forward(X, L - 1))

        # Propagate errors backwards to the network
        for layer in range(L - 1, -1, -1):
            accum_grad = net.layers[layer + 1].accum_grad
            net.layers[layer].backward(accum_grad, net.forward(X, layer - 1))
