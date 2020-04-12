from __future__ import annotations

import numpy as np


class LinearRegression:
    """
    Fit line (weighted linear function) using Least Squares Estimation (matrix inversion)

    Attributes
    ----------
    betas : numpy.ndarray
        Array of shape (n_features + 1) containing fit coefficients (including intercept)
    
    rsquared : float
        Coefficient of determination. Denotes the proportion of the variance in the 
        dependent variable that is predictable from the independent variable(s)

    Methods
    -------
    fit(X, y)
        Estimates the linear model betas (weights/coefficients)
        
    predict(X)
        Applies fit model on new data array and returns predictions
    """

    def __init__(self):
        self.betas = None
        self.rsquared = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Estimates the linear model betas (weights/coefficients)
        
        Parameters
        ----------
        X : numpy.ndarray
            Array of training samples with shape (n_samples, n_features)
        
        y : numpy.ndarray
            Array of training targets with shape (n_samples,)
        """

        n_samples, n_features = X.shape

        X_aug = np.column_stack((np.ones(n_samples), X))  # to account for intercept

        k = 1e-6 * X.min()  # to avoid inverting a singular matrix
        self.betas = np.linalg.inv(X_aug.T.dot(X_aug) + k * np.eye(n_features + 1)).dot(X_aug.T).dot(y)

        # R squared
        self.rsquared = 1 - ((y - self.predict(X)) ** 2).sum() / ((y - np.mean(y)) ** 2).sum()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Applies fit model on new data array and returns predictions
        
        Parameters
        ----------
        X : numpy.ndarray
            Array of testing samples with shape (n_samples, n_features)
            
        Raises
        ------
        ValueError
            The fit() method has to be called first so that the betas
            are estimated
        """

        if self.betas is None:
            raise ValueError("Betas not fitted. Call fit() method first")

        n_samples = X.shape[0]

        X_aug = np.column_stack((np.ones(n_samples), X))
        return X_aug.dot(self.betas)
