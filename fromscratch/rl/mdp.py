import itertools

import numpy as np
import matplotlib.pyplot as plt


class State:
    """
    Class implementing a state. Defines allowed actions and feature vector.
    Tho States are equal if their vectors have all their components equal.

    Parameters
    ----------
    vector
        List / numpy array with feature representation of the state

    Methods
    -------
    set_allowed_actions(actions)
        Sets which actions can be performed in the state
    """

    def __init__(self, vector):
        if isinstance(vector, list):
            vector = np.array(vector)
        self.vector = vector

        self.allowed_actions = []

    def set_allowed_actions(self, actions: list):
        """
        Sets which actions can be performed in the state

        Parameters
        ----------
        actions : list
            List of actions
        """

        self.allowed_actions = actions
        return self

    def __repr__(self):
        return str(self.vector)

    def __eq__(self, other):
        return np.all(self.vector == other.vector)

    def __hash__(self):
        return hash(tuple(self.vector.tolist()))


class ActionSpace:
    """
    Class for an action space representation. Defines which are the possible actions
    and their names (aliases) and vectors.

    Parameters
    ----------
    allowed_actions : dict
        Dictionary of the form name: vector defining the allowed actions and their
        aliases of the action space

    Methods
    -------
    action(vector_or_name)
        Returns an action based on the name of feature vector given
    """

    def __init__(self, allowed_actions):
        self.allowed_actions = allowed_actions
        for k in self.allowed_actions:
            if not isinstance(self.allowed_actions[k], np.ndarray):
                self.allowed_actions[k] = np.array(self.allowed_actions[k])

    def action(self, vector_or_name):
        """
        Returns an action based on the name of feature vector given

        Parameters
        ----------
        vector_or_name
            Vector or string. If string, it represents the name or alias of the action and is mapped
            to the corresponding vector.
        """

        if isinstance(vector_or_name, str):
            if vector_or_name not in self.allowed_actions.keys():
                raise ValueError("Not allowed action name")
            return self.Action(vector=self.allowed_actions[vector_or_name], name=vector_or_name)

        if isinstance(vector_or_name, list):
            vector_or_name = np.array(vector_or_name)

        if isinstance(vector_or_name, np.ndarray):
            if vector_or_name not in self.allowed_actions.values():
                raise ValueError("Not allowed action vector")
            return self.Action(vector=vector_or_name)

    class Action:
        def __init__(self, vector, name=None):
            self.vector = vector
            self.name = name

        def __eq__(self, other):
            return np.all(self.vector == other.vector)

        def __hash__(self):
            return hash(tuple(self.vector.tolist()))

        def __repr__(self):
            return self.name if self.name is not None else self.vector


class GridMDP:
    """
    Generic MDP on grid state-space where allowed actions are going nort, south,
    west or east.

    Parameters
    ----------
    rows : int
        Rows of the grid

    cols : int
        Columns of the grid

    goal : list
        Goal (terminal) states
    """

    def __init__(self, rows, cols, goal):
        self.rows = rows
        self.cols = cols
        self.goal = goal

        self.state = None

        self.states = list(map(lambda x: State(list(x)), itertools.product(range(rows), range(cols))))

        self.action_space = ActionSpace({
            "north": np.array([-1, 0]),
            "south": np.array([1, 0]),
            "east": np.array([0, 1]),
            "west": np.array([0, -1])
        })

        for s in self.states:
            s.set_allowed_actions([
                self.action_space.action("north"),
                self.action_space.action("south"),
                self.action_space.action("east"),
                self.action_space.action("west")
            ])

        self.state = None

    def find_state(self, state):
        """
        Returns a reference to the state of the MDP with the same feature vector
        as the one given as input
        """

        if isinstance(state, np.ndarray):
            state = State(state)
        filtered_states = list(filter(lambda x: x == state, self.states))
        if len(filtered_states) > 0:
            return filtered_states[0]
        return None

    def set_state(self, state):
        """
        Sets current MDP state. Returns self for method chaining
        """

        state = self.find_state(state)
        if state is not None:
            self.state = state
        else:
            raise ValueError("State not found")
        return self

    def step(self, action, transition: bool = True):
        raise NotImplementedError("step method must be implemented by inherited class")

    def get_reward(self, action):
        """
        Implements the reward function base on the action performed
        """

        if self.state in self.goal:
            return 0
        else:
            return -1


class GridWorld(GridMDP):
    """
    Implements the Grid World environment

    Parameters
    ----------
    rows : int
        Rows of the grid

    cols : int
        Columns of the grid

    goal : list
        Goal (terminal) states

    wind : numpy.ndarray
        vector of vertical force (wind) applied to the agent in each state
    """

    def __init__(self, rows: int, cols: int, goal: list, wind: np.ndarray = None):
        super().__init__(rows, cols, goal)
        self.wind = wind
        if wind is None:
            self.wind = np.zeros(cols)

    def step(self, action, transition: bool = True):
        """
        The most important part of the MDP. Defines how the MDP state changes when
        an action is selected. Returns the reward of the performed action

        Parameters
        ----------
        action
            Action to perform

        transition : bool
            If True, the MDP transitions to the next state. If False, the next state
            is returned
        """

        if isinstance(action, str):
            action = self.action_space.action(action)

        reward = self.get_reward(action)

        new_vector = self.state.vector + action.vector

        wind_vertical = self.wind[self.state.vector[1]]
        new_vector = new_vector + np.array([wind_vertical, 0])

        new_state = State(np.array([
            np.clip(new_vector[0], 0, self.rows - 1),
            np.clip(new_vector[1], 0, self.cols - 1)
        ]))

        if transition:
            self.state = self.find_state(new_state)

        return self.find_state(new_state), reward

    def plot(self, policy):
        """
        Visualizes the state space of the MDP and colors them according to their Value.
        Also shows the policy actions as arrows on the grid
        """

        values = np.array([policy.V[s] for s in self.states]).reshape((self.rows, self.cols))
        plt.imshow(values, cmap="Reds")

        for g in self.goal:
            plt.scatter([g.vector[1]], [g.vector[0]], facecolor="blue")

        optimal_actions_inds = [
            np.random.choice(range(len(s.allowed_actions)), p=list(policy.get_action_proba(s).values()))
            for s in self.states
        ]

        optimal_actions = [
            s.allowed_actions[i].name for s, i in zip(self.states, optimal_actions_inds)
        ]

        dxdy = {
            "north": [0, -0.4],
            "south": [0, 0.4],
            "east": [0.4, 0],
            "west": [-0.4, 0],
        }
        [plt.arrow(s.vector[1], s.vector[0], dxdy[a][0], dxdy[a][1], head_width=0.1, head_length=0.1, fc='k', ec='k')
         for s, a in zip(self.states, optimal_actions) if s not in self.goal]

        plt.title("Grid World")
        plt.show()


class CliffWalking(GridMDP):
    def __init__(self, rows: int, cols: int, goal: list):
        super().__init__(rows, cols, goal)

    def step(self, action, transition: bool = True):
        if isinstance(action, str):
            action = self.action_space.action(action)

        reward = self.get_reward(action)

        new_vector = self.state.vector + action.vector

        if new_vector[0] == self.rows - 1 and 0 < new_vector[1] < self.cols - 1:
            new_vector = np.array([self.rows - 1, 0])

        new_state = State(np.array([
            np.clip(new_vector[0], 0, self.rows - 1),
            np.clip(new_vector[1], 0, self.cols - 1)
        ]))

        if transition:
            self.state = self.find_state(new_state)

        return self.find_state(new_state), reward

    def get_reward(self, action):
        """
        Implements the reward function base on the action performed
        """

        if self.state in self.goal:
            return 0
        elif (self.state.vector + action.vector)[0] == self.rows - 1 and 0 < (self.state.vector + action.vector)[1] < self.cols - 1:
            return -100
        else:
            return -1

    def plot(self, policy):
        """
        Visualizes the state space of the MDP and colors them according to their Value.
        Also shows the policy actions as arrows on the grid
        """

        values = np.array([policy.V[s] for s in self.states]).reshape((self.rows, self.cols))
        plt.imshow(values, cmap="Reds")

        plt.scatter(range(1, self.cols), [self.rows - 1] * (self.cols - 1), marker="x", c="grey")

        for g in self.goal:
            plt.scatter([g.vector[1]], [g.vector[0]], facecolor="blue")

        optimal_actions_inds = [
            np.random.choice(range(len(s.allowed_actions)), p=list(policy.get_action_proba(s).values()))
            for s in self.states
        ]

        optimal_actions = [
            s.allowed_actions[i].name for s, i in zip(self.states, optimal_actions_inds)
        ]

        dxdy = {
            "north": [0, -0.4],
            "south": [0, 0.4],
            "east": [0.4, 0],
            "west": [-0.4, 0],
        }
        [plt.arrow(s.vector[1], s.vector[0], dxdy[a][0], dxdy[a][1], head_width=0.1, head_length=0.1, fc='k', ec='k')
         for s, a in zip(self.states, optimal_actions) if s not in self.goal and not (s.vector[0] == self.rows-1 and 0 < s.vector[1] < self.cols - 1)]

        plt.title("Grid World")
        plt.show()
