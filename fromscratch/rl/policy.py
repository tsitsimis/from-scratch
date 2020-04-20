import numpy as np
from fromscratch.rl.mdp import State


class EpsilonGreedy:
    """
    Implements the ε-greedy policy. Actions are selected based on the
    value function of each state or state-action at each moment.

    Parameters
    ----------
    mdp
        An instance of a class implementing a Markov Decision Process

    epsilon : float
        Between 0 and 1 probability of exploration

    gamma : float
        Between 0 and 1 discount rate

    value_function : str
        Can be 'V' or 'Q' depending on which value function is used for action selection
        For 'V' (state value) the model must be known

    Methods
    -------
    get_action_proba(state)
        Returns action probabilities

    select_action(state)
        Returns an action based on action probabilities of the state

    reset()
        Sets all state and state-action values to 0
    """

    def __init__(self, mdp, epsilon, gamma, value_function="V"):
        self.mdp = mdp
        self.epsilon = epsilon
        self.gamma = gamma
        self.value_function = value_function

        self.V = None
        self.Q = None

        self.reset()

    def get_action_proba(self, state: State) -> dict:
        n_actions = len(state.allowed_actions)
        action_proba = np.ones(n_actions) * self.epsilon / n_actions

        if self.value_function == "V":
            self.mdp.set_state(state)
            values = [self.V[self.mdp.step(a, transition=False)[0]] for a in state.allowed_actions]
        else:
            values = [self.Q[state][a] for a in state.allowed_actions]

        # argmax_ind = np.argwhere(values == np.amax(values)).flatten().tolist()
        # action_proba[np.random.choice(argmax_ind)] = 1 - self.epsilon + self.epsilon / n_actions
        action_proba[np.argmax(values)] = 1 - self.epsilon + self.epsilon / n_actions
        return {a: p for a, p in zip(state.allowed_actions, action_proba)}

    def select_action(self):
        """
        Returns an action based on action probabilities of the state
        """

        state = self.mdp.state

        if len(state.allowed_actions) == 0:
            return None

        action_proba = self.get_action_proba(state)
        return np.random.choice(state.allowed_actions, p=list(action_proba.values()))

    def reset(self):
        """
        Sets value functions to 0
        """

        self.V = {state: 0 for state in self.mdp.states}
        self.Q = {state: {a: 0 for a in state.allowed_actions} for state in self.mdp.states}


class GenericPolicy:
    """
    Implements a generic policy. Actions are selected based on the
    action probabilities of each state at each moment. Initially actions
    probabilities are equal.

    Parameters
    ----------
    mdp
        An instance of a class implementing a Markov Decision Process

    Methods
    -------
    select_action(state)
        Returns an action (Action instance) based on action probabilities of the state

    update_action_proba(state, proba)
        Updates the action probabilities of a state

    reset()
        Makes all action probabilities equal in each state and sets all state Value to 0
    """

    def __init__(self, mdp, gamma):
        self.mdp = mdp
        self.gamma = gamma

        self.action_proba = None
        self.V = None
        self.reset()

    def get_action_proba(self, state):
        return self.action_proba[state]

    def select_action(self):
        """
        Returns an action (Action instance) based on action probabilities of the state
        """

        state = self.mdp.state

        if len(state.allowed_actions) == 0:
            return None
        return np.random.choice(state.allowed_actions, p=list(self.action_proba[state].values()))

    def update_action_proba(self, state, proba: dict):
        """
        Updates the action probabilities of a state

        Parameters
        ----------
        state
            State of the MDP

        proba : dict
            Dictionary mapping actions to probabilities
        """

        self.action_proba[state] = proba

    def reset(self):
        """
        Makes all action probabilities equal in each state and sets all state Value to 0
        """

        self.action_proba = {state: {action: 1 / len(state.allowed_actions) for action in state.allowed_actions} for
                             state in self.mdp.states}
        self.V = {state: 0 for state in self.mdp.states}
