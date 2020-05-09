from __future__ import annotations

import numpy as np


class KMeans:
    """
    Iterative K-means implementation

    Parameters
    ----------
    K : int
        Number of clusters to produce

    Methods
    -------
    fit(X)
        Repeats assignment and update steps until convergence (no change in estimated centers)
    """

    def __init__(self, K):
        self.K = K

        self.means = None
        self.assigned_means = None

    def fit(self, X: np.ndarray):
        """
        Repeats assignment and update steps until convergence (no change in estimated centers)

        Parameters
        ----------
        X : numpy.ndarray
            Array with samples to cluster
        """

        # Randomly sample K samples as initial centers
        means = X[np.random.randint(X.shape[0], size=self.K), :]

        while True:
            assigned_means = self.assign(X, means)  # Assignment Step
            means_new = self.update(X, assigned_means)  # Update step
            if np.array_equal(means_new, means):
                break
            means = means_new

        self.means = means
        self.assigned_means = assigned_means

    def assign(self, X: np.ndarray, means: np.ndarray) -> np.ndarray:
        dist = np.stack([
            np.sum((X - means[[k], :]) ** 2, axis=1) for k in range(self.K)
        ]).T

        assigned_means = np.argmin(dist, axis=1)
        return assigned_means

    def update(self, X: np.ndarray, assigned_means: np.ndarray) -> np.ndarray:
        means = np.stack([
            X[assigned_means == k].mean(axis=0) for k in range(self.K)
        ])
        return means
