from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from .classification_tree import ClassificationTree


class AdaBoost(BaseEstimator, ClassifierMixin):
    """
    Adaptive Boosting learner based on classification trees

    Methods
    -------
    fit(X, y)
        Iteratively adds and fits weak learners on data by updating sample weights
        according to classification error of previous model

    predict(X)
        Returns predictions for the input samples
    """

    def __init__(self):
        self.alphas = None  # weights used to combine weak learners
        self.learners = None
        self.p_list = None

    def fit(self, X: np.ndarray, y: np.ndarray, max_iters: int = 10) -> AdaBoost:
        """
        Iteratively adds and fits weak learners on data by updating sample weights
        according to classification error of previous model

        Parameters
        ----------
        X : numpy.ndarray
            Array of training samples with shape (n_samples, n_features)

        y : numpy.ndarray
            Array of training targets with shape (n_samples,)

        max_iters : int
            Number of boosting iterations
        """

        weights = np.ones(X.shape[0]) / X.shape[0]
        m = 0
        self.learners = [None] * max_iters  # arrayholding all learners
        self.alphas = [None] * max_iters  # array holding weights
        self.p_list = [None] * max_iters  # array holding classification error in each iteration

        while True:
            clf = ClassificationTree(max_depth=4, min_leaf_samples=1, min_delta_impurity=0.0)
            clf = clf.fit(X, y, sample_weights=weights)
            self.learners[m] = clf

            y_pred = clf.predict(X)
            P_m = (((1 - y * y_pred) > 0).astype(int) * weights).sum()
            self.p_list[m] = P_m

            a_m = (1 / 2) * np.log((1 - P_m) / P_m)
            self.alphas[m] = a_m

            weights = weights * np.exp(-y * a_m * y_pred)
            weights = weights / weights.sum()

            m += 1
            if m == max_iters:
                break

        return self

    def predict(self, X: np.ndarray):
        """
        Returns predictions for the input samples

        Parameters
        ----------
        X : numpy.ndarray
            Array of testing samples with shape (n_samples, n_features)
        """

        if self.alphas is None:
            raise ValueError("Model not fitted. Call fit() method first")

        return np.sign(np.array([a * clf.predict(X) for a, clf in zip(self.alphas, self.learners)]).T.sum(axis=1))

    def score(self, X, y, **kwargs):
        return accuracy_score(y, self.predict(X))
