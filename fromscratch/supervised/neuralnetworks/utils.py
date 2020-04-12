import numpy as np


def augmented(X: np.ndarray):
    """
    Augments input data array by adding an extra dimension
    with value 1
    """
    return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
