import numpy as np


class SARSA:
    """
    Implementation of the SARSA on-policy TD control

    Parameters
    ----------
    mdp
        MDP the policy is applied on

    policy
        Policy for which the Q function is estimated

    lr : float
        Learning rate of the SARSA update rule

    Methods
    -------
    play_episodes(n)
        Iterates n episodes of the SARSA TD control. In each episode the state-action value
        function Q of the policy is updated
    """

    def __init__(self, mdp, policy, lr=0.1):
        self.mdp = mdp
        self.policy = policy
        self.lr = lr

        self.episode_reward = None

    def play_episodes(self, n=100):
        """
        Iterates n episodes of the SARSA TD control. In each episode the state-action value
        function Q of the policy is updated

        Parameters
        ----------
        n : int
            Number of episodes to play
        """

        if self.episode_reward is None:
            self.episode_reward = np.empty(n)
            i0 = 0
        else:
            i0 = self.episode_reward.shape[0]
            self.episode_reward = np.concatenate((self.episode_reward, np.empty(n)))

        for i in range(n):
            self.mdp.set_state(np.random.choice(self.mdp.states))

            state = self.mdp.state
            if state in self.mdp.goal:
                self.episode_reward[i + i0] = None
                continue

            action = self.policy.select_action()

            total_reward = 0
            while True:
                state_next, reward = self.mdp.step(action, transition=True)
                total_reward += reward
                if state_next in self.mdp.goal:
                    break

                action_next = self.policy.select_action()

                self.policy.Q[state][action] = self.policy.Q[state][action] + self.lr * (
                        reward + self.policy.gamma * self.policy.Q[state_next][action_next] -
                        self.policy.Q[state][action])

                state = state_next
                action = action_next

            self.episode_reward[i + i0] = total_reward


class QLearning:
    """
    Implementation of the Q-Learning off-policy TD control

    Parameters
    ----------
    mdp
        MDP the policy is applied on

    policy
        Policy for which the Q function is estimated

    lr : float
        Learning rate of the Q-Learning update rule

    Methods
    -------
    play_episodes(n)
        Iterates n episodes of the Q-Learning TD control. In each episode the state-action value
        function Q of the policy is updated
    """

    def __init__(self, mdp, policy, lr=0.1):
        self.mdp = mdp
        self.policy = policy
        self.lr = lr

        self.episode_reward = None

    def play_episodes(self, n=100):
        """
        Iterates n episodes of the Q-Learning TD control. In each episode the state-action value
        function Q of the policy is updated

        Parameters
        ----------
        n : int
            Number of episodes to play
        """

        if self.episode_reward is None:
            self.episode_reward = np.empty(n)
            i0 = 0
        else:
            i0 = self.episode_reward.shape[0]
            self.episode_reward = np.concatenate((self.episode_reward, np.empty(n)))

        for i in range(n):
            self.mdp.set_state(np.random.choice(self.mdp.states))

            state = self.mdp.state
            if state in self.mdp.goal:
                self.episode_reward[i + i0] = None
                continue

            total_reward = 0
            while True:
                action = self.policy.select_action()
                state_next, reward = self.mdp.step(action, transition=True)
                total_reward += reward
                if state_next in self.mdp.goal:
                    break

                self.policy.Q[state][action] = self.policy.Q[state][action] + self.lr * (
                        reward +
                        self.policy.gamma * np.max([self.policy.Q[state_next][a] for a in state.allowed_actions]) -
                        self.policy.Q[state][action])

                state = state_next

            self.episode_reward[i + i0] = total_reward
