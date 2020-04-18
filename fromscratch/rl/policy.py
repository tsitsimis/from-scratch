import numpy as np


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

    def __init__(self, mdp):
        self.mdp = mdp

        self.action_proba = None
        self.value = None
        self.reset()

    def select_action(self, state):
        """
        Returns an action (Action instance) based on action probabilities of the state

        Parameters
        ----------
        state
            State of the MDP
        """
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

        self.action_proba[state] = {**self.action_proba[state], **proba}

    def reset(self):
        """
        Makes all action probabilities equal in each state and sets all state Value to 0
        """

        self.action_proba = {state: {action: 1 / len(state.allowed_actions) for action in state.allowed_actions} for
                             state in self.mdp.states}
        self.value = {state: 0 for state in self.mdp.states}
