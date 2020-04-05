from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from cvxopt import matrix
from cvxopt.solvers import qp


class SVC(BaseEstimator, ClassifierMixin):
    def __init__(self, C: float = 1.0, kernel: callable = None):
        self.C = C
        if kernel is None:
            self.kernel = lambda x1, x2: x1.dot(x2)
        else:
            self.kernel = kernel

        self.lagrange_mult = None
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> SCV:
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError("This has to be a binary classification problem")

        n_samples, n_features = X.shape
        X_aug = np.column_stack((np.ones(n_samples), X))

        # Build convex optimization matrices
        X = X.astype(float)
        y = y.astype(float)

        kernel_matrix = np.empty((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X_aug[i, :], X_aug[j, :])

        P = np.outer(y, y) * kernel_matrix
        P = matrix(P)
        q = matrix(-np.ones((n_samples, 1)))
        G = matrix(np.concatenate((np.eye(n_samples), -np.eye(n_samples))))
        h = matrix(np.concatenate((C * np.ones(n_samples), np.zeros(n_samples))))
        A = matrix(y[None, :])
        b = matrix(0.0)

        solution = qp(P, q, G, h, A, b, options={"show_progress": False})
        self.lagrange_mult = np.array(solution["x"]).T[0]

        self.X_train = X_aug
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        n1 = self.X_train.shape[0]
        n2 = X.shape[0]

        X_aug = np.column_stack((np.ones(n2), X))

        kernel_matrix = np.empty((n1, n2))
        for i in range(n1):
            for j in range(n2):
                kernel_matrix[i, j] = self.kernel(self.X_train[i, :], X_aug[j, :])

        y_pred = kernel_matrix * self.lagrange_mult[:, None] * self.y_train[:, None]
        y_pred = np.sum(y_pred, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
