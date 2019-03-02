#!/usr/bin/env python
import sys
import gym
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


def get_epsilon_greedy_policy(Q, nA, state, epsilon):
    """Given a Q-table, the dimension of the action space, the current state
    and a probability for non-greedy action epsilon, this function returns
    the policy for the different actions (in terms of probabilities for
    each action)

    Args:
        Q (:obj:`dict` of :obj:)
        nA (int): Dimension of the action space.
        state (:obj:`state`): Current state.
        epsilon (float): Probability for non-greedy action (b/w 0. and 1.)

    Returns:
        policy (:obj:`np.array`): Epsilon greedy policy.
    """
    policy = np.ones(nA)
    # If state has been visited before, select greedy action
    # with highest probability
    if state in Q:
        policy_random = epsilon / nA
        policy *= policy_random
        policy_greedy = 1. - epsilon + (epsilon / nA)
        policy[np.argmax(Q[state])] = policy_greedy
    # If state has not been seen before, select randomly
    else:
        policy *= 1. / nA
    return policy


def get_epsilon_greedy_action(Q, nA, state, epsilon):
    """Given a Q-table, the dimension of the action space, the current state
    and a probability for non-greedy action epsilon, this function returns
    the epsilon greedy action

    Args:
        Q (:obj:`dict` of :obj:)
        nA (int): Dimension of the action space.
        state (:obj:`state`): Current state.
        epsilon (float): Probability for non-greedy action (b/w 0. and 1.)

    Returns:
        action (int): Epsilon greedy action.

    """
    policy = get_epsilon_greedy_policy(Q, nA, state, epsilon)
    return np.random.choice(np.arange(nA), p=policy)


def temporal_difference_learning(env, num_episodes, alpha, method='sarsa', gamma=1.0):
    """Function that implements TD-learning using different solution strategies.

    Args:
    Args:
        env (:obj: `gym.env`): An OpenAI Black Jack env instance.
        num_episodes (int): Number of episodes to generate.
        alpha (float): Fraction by which the Q-table shall be updated at every
            new visit to a state-action pair and a new final reward.
        method (str): The TD-learning method to apply, either 'sarsa', 'sarsamax' or
            'expected_sarsa'.
        gamma (float): The discount factor gramma.


    Returns:
        Q (:obj:`defaultdict`): Dictionary of the form Q[state][action].
        returns (:obj:`np.array`): Numpy array of total returns for each
            episode, for plotting.
    """

    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    returns = [] # List of returns

    # initialize performance monitor

    nA = env.action_space.n

    # loop over episodes
    for i_episode in range(1, num_episodes+1):

        # monitor progress
        print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
        sys.stdout.flush()

        # reset environment
        state = env.reset()

        # epsilon is reduced with every episode, so
        # the algorithm becomes more and more greedy
        epsilon = 1.0 / i_episode

        # get the first epsilon-greedy action
        action = get_epsilon_greedy_action(Q, nA, state, epsilon)

        # initialize return
        total_return = 0

        # inner loop
        while True:

            next_state, reward, done, info = env.step(action)
            total_return += reward

            if not done:

                next_action = get_epsilon_greedy_action(Q, nA, next_state, epsilon)

                if method=='sarsa':

                    # update the Q-table using the next state and next action
                    Q[state][action] += alpha * (reward + gamma * \
                                        Q[next_state][next_action] - Q[state][action])

                if method=='q_learning' or method=='sarsamax':

                    # update the Q-table using the action that maximizes Q of the next state
                    Q[state][action] += alpha * (reward + gamma * \
                                        np.max(Q[next_state]) - Q[state][action])

                if method=='expected_sarsa':

                    # for expected sarsa we need the entire policy for this action
                    policy = get_epsilon_greedy_policy(Q, nA, next_state, epsilon)

                    # update the Q table using the dot product of policy and the Q table and next state
                    Q[state][action] += alpha * (reward + gamma * \
                                        np.dot(policy, Q[next_state]) - Q[state][action])
            else:
                Q[state][action] += alpha * (reward - Q[state][action])
                break

            state = next_state
            action = next_action

        # for plotting
        returns.append(total_return)

    return Q, np.array(returns)



if __name__ == '__main__':
    """Solves the Cliff Walking environment using TD-learning.

    Usage:
        python td_cliff_walking.py
    """

    # Generate an instance of the Gym CliffWalking environment
    env = gym.make('CliffWalking-v0')

    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsa, G_sarsa = temporal_difference_learning(env, 5000, .01, 'sarsa')
    Q_sarsamax, G_sarsamax = temporal_difference_learning(env, 5000, .01, 'sarsamax')
    Q_expsarsa, G_expsarsa = temporal_difference_learning(env, 5000, .01, 'expected_sarsa')

    df = pd.DataFrame(
        {
            'sarsamax': G_sarsamax,
            'expsarsa': G_expsarsa,
            'sarsa': G_sarsa,
        }
    )

    window_size = 100

    plt.figure()
    plt.plot(df.sarsa.rolling(window=window_size, center=False).mean())
    plt.plot(df.sarsamax.rolling(window=window_size, center=False).mean())
    plt.plot(df.expsarsa.rolling(window=window_size, center=False).mean())
    plt.ylabel("Average return at episode")
    plt.xlabel("Episode")
    plt.show()
