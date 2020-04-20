"""
Dynamic Programming (DP) algorithms
"""


import numpy as np


class PolicyEvaluation:
    """
    Performs Policy Evaluation on given MDP and policy

    Parameters
    ----------
    mdp
        An instance of a class implementing a Markov Decision Process

    policy
        Policy to evaluate

    gamma : float
        Discount factor. between 0 and 1

    Methods
    -------
    evaluate_once()
        Runs one iteration of the Policy Iteration algorithm. Returns
        the max difference of the value function in the previous and
        current step
    """

    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

    def evaluate_once(self) -> float:
        """
        Runs one iteration of the Policy iteration algorithm. Returns
        the max difference of the value function in the previous and
        current step
        """

        delta = 0
        for s in self.mdp.states:
            self.mdp.set_state(s)

            state_value_prev = self.policy.V[s]

            state_value = 0
            for a in s.allowed_actions:
                s_next, r = self.mdp.step(a, transition=False)
                state_value += self.policy.get_action_proba(s)[a] * 1 * (r + self.policy.gamma * self.policy.V[s_next])

            self.policy.V[s] = state_value
            delta = np.max([delta, np.abs(state_value_prev - state_value)])
        return delta


class PolicyImprovement:
    """
    Performs Policy Improvement on given MDP and policy

    Parameters
    ----------
    mdp
        An instance of a class implementing a Markov Decision Process

    policy
        Policy to evaluate

    gamma : float
        Discount factor. between 0 and 1

    Methods
    -------
    improve_once()
        Runs one iteration of the Policy Improvement algorithm. Returns
        if the policy is stable which means than no changes (improvements)
        have been made
    """

    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

        self.policy_stable = True

    def improve_once(self) -> bool:
        """
        Runs one iteration of the Policy Improvement algorithm. Returns
        if the policy is stable which means than no changes (improvements)
        have been made
        """

        for s in self.mdp.states:
            if len(s.allowed_actions) == 0:
                continue

            self.mdp.set_state(s)

            a = self.policy.select_action()

            values = np.empty(len(s.allowed_actions))
            for i, a in enumerate(s.allowed_actions):
                s_next, r = self.mdp.step(a, transition=False)
                values[i] = r + self.policy.gamma * self.policy.V[s_next]

            action_proba = np.zeros(len(s.allowed_actions))
            action_proba[np.argmax(values)] = 1

            self.policy.update_action_proba(s, {a: p for a, p in zip(s.allowed_actions, action_proba)})

            if a != self.policy.select_action():
                self.policy_stable = False

        return self.policy_stable


class PolicyIteration:
    """
        Performs Policy Iteration on given MDP and policy by running
        Policy Evaluation and Policy Improvement sequentially

        Parameters
        ----------
        mdp
            An instance of a class implementing a Markov Decision Process

        policy
            Policy to evaluate

        gamma : float
            Discount factor. between 0 and 1

        Methods
        -------
        iterate()
            Run evaluation and improvement steps until the policy is
            stable or maximum number of iterations is reached
        """

    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

        self.evaluator = PolicyEvaluation(mdp, policy)
        self.improver = PolicyImprovement(mdp, policy)

    def iterate(self, iterations: int = 10):
        """
        Run evaluation and improvement steps until the policy is
        stable or maximum number of iterations is reached

        Parameters
        ----------
        iterations : int
            Maximum number of evaluation and improvement steps
            to perform
        """

        i = 0
        while not self.improver.policy_stable and i <= iterations:
            self.evaluator.evaluate_once()
            self.improver.improve_once()
            i += 1
