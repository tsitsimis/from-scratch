import numpy as np


class Bandit:
    """
    Simulates a n-armed bandit

    Parameters
    ----------
    n : int
        Number of arms of the bandit

    alpha : float
        Step-size parameter of exponential (recency-weighted) average. Takes value from 0 to 1.

    observation_noise : float
        noise to add to the observed reward


    Methods
    -------
    get_reward(action)
        Returns the reward of the selected action

    update_estimate(action)
        Given a selected action, it returns the reward and updates the estimate of the action value
    """

    def __init__(self, n: int, alpha: float = 0.1, action_values=None, observation_noise: float = 1):
        self.alpha = alpha
        self.observation_noise = observation_noise

        if action_values is None:
            self.action_values = np.random.multivariate_normal(np.ones(n), np.eye(n))
        else:
            if len(action_values) != n:
                raise ValueError("Provide as many action values as bandit arms (n)")

            self.action_values = action_values

        self.action_values_est = np.zeros(n)

    def get_reward(self, action: int) -> float:
        """
        Returns the reward of the selected action

        Parameters
        ----------
        action : int
            Action index
        """
        return self.action_values[action] + np.random.normal(0, 1)

    def update_estimate(self, action: int) -> float:
        """
        Given a selected action, it returns the reward and updates the estimate of the action value

        Parameters
        ----------
        action : int
            Action index
        """
        reward = self.get_reward(action)
        self.action_values_est[action] += self.alpha * (reward - self.action_values_est[action])
        return reward


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, action_values):
        if np.random.random() < self.epsilon:
            return np.random.randint(action_values.shape[0])

        return np.argmax(action_values)
