import numpy as np


class PCA:
    """
    Principal Components Analysis

    Parameters
    ----------
    n_components : int
        Number of dimensions to keep after PCA projection

    Methods
    -------
    fit
        Calculates eigenvectors of X's covariance matrix with columns
        sorted in descending eigenvalue order

    transform
        Maps a new data array according to the linear transformation calculated during fit.
        Also keeps only the dimensions of the mapped samples corresponding to the
        biggest n_components eigenvalues calculated in fit.
    """

    def __init__(self, n_components: int = None):
        self.n_components = n_components

        self.A = None
        self.eigenvalues = None
        self.dim = None

    def fit(self, X: np.ndarray):
        """
        Calculates eigenvectors of X's covariance matrix with columns
        sorted in descending eigenvalue order

        Parameters
        ----------
        X : numpy.ndarray
            Data array (N, D) where N is the number of samples and d the number of
            dimensions/features
        """

        self.dim = X.shape[1]

        X_scaled = X - X.mean(axis=0)

        Rx = np.cov(X_scaled.T)
        eigenvalues, eigenvectors = np.linalg.eig(Rx)

        idx = np.argsort(-eigenvalues)  # sort in descending order
        self.eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.A = eigenvectors

    def transform(self, X: np.ndarray):
        """
        Maps a new data array according to the linear transformation calculated during fit.
        Also keeps only the dimensions of the mapped samples corresponding to the
        biggest n_components eigenvalues calculated in fit.

        Parameters
        ----------
        X : numpy.ndarray
            Data array (N, D) where N is the number of samples and d the number of
            dimensions/features
        """

        if self.A is None:
            raise ValueError("Call fit() method first")

        if X.shape[1] != self.dim:
            raise ValueError("Give input array X with same number of dimensions as the array called in fit")

        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        return X.dot(self.A)[:, 0:n_components]
