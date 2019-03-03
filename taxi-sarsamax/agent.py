import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=0.1, gamma=1.0): #1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.gamma = gamma
        self.alpha = alpha
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state, epsilon):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy = np.ones(self.nA)
        # If state has been visited before, select greedy action
        # with highest probability
        if state in self.Q:
            policy_random = epsilon / self.nA
            policy *= policy_random
            policy_greedy = 1. - epsilon + (epsilon / self.nA)
            policy[np.argmax(self.Q[state])] = policy_greedy
        # If state has not been seen before, select randomly
        else:
            policy *= 1. / self.nA
        return np.random.choice(np.arange(self.nA), p=policy)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += self.alpha * (reward + self.gamma * \
                          np.max(self.Q[next_state]) - self.Q[state][action])
