#!/usr/bin/env python
import sys
import gym
import numpy as np
from collections import defaultdict

"""This contains methods to solve the OpenAI Gym
Black Jack environment using Monte Carlo Control.
"""

def generate_black_jack_episode_limit_stochastic(env):
    """Implements an apriori policy where the agent draws with 0.8 probability
    when the some of his cards is less or equal than 18, and 0.2 when the sum
    is higher than 18.

    Args:
        env (:obj: `gym.env`): An OpenAI Black Jack env instance.

    Returns:
        episode (:obj:`list` of :obj:`tuple`): List of state-action-reward
            tuples within the generated episode.
    Raises:
        None
    """
    episode = []
    state = env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0, first_visit=False):
    """Implements first- and every-visit Monte-Carlo prediction of the Action
    Value Function Q, in the form of a Q-table.

    Args:
        env (:obj: `gym.env`): An OpenAI Black Jack env instance.
        num_episodes (int): Number of episodes to generate.
        generate_episode (:obj:`func`): Function that returns a list of episodes
            to be used to update the Q-table. E.g.
        gamma (float): The discount factor gramma.
        first_visit (bool): True, if first-visit MC prediction should be ran,
            False, if every-visit method should be used.

    Returns:
        Q (:obj:`defaultdict`): A dictionary ala Q[state][action] = reward.

    Raises:
        None
    """

    # Initialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Loop over episodes
    for i_episode in range(1, num_episodes+1):

        # Generate episode
        episode = generate_episode(env)

        # For Black Jack, the reward -1 or +1 is always given at the last episode.
        # To be more general, we sum all the reward for every step in this episode.
        # Furthermore we multiply with the discount factor. Note that el[2] is the
        # reward for the current state.
        final_reward = sum([el[2]*(gamma**i) for i, el in enumerate(episode)])

        # Loop over state-action-reward tubples in episode
        for sar_tuple in episode:

            state  = sar_tuple[0]
            action = sar_tuple[1]

            # Must be zero in case state-action pair was not visited before
            if first_visit:
                if not N[state][action]:
                    N[state][action] += 1
                    returns_sum[state][action] += final_reward
            else:
                N[state][action] += 1
                returns_sum[state][action] += final_reward

            # Update Q value
            Q[state][action] = returns_sum[state][action] / N[state][action]

        # Monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

    return Q

def generate_episode_using_policy(env, epsilon, Q):
    """Generates an episode using by evaluating an the current best policy using
    a provided Q-table.

    Args:
        env (:obj: `gym.env`): An OpenAI Black Jack env instance.
        epsilon (float): Epsilon factor in epsilon-greedy policy
            evaliation.
        Q (:obj:`defaultdict`): The Q-table which is a dictionary
            ala Q[state][action] = reward.

    Returns:
        episode (:obj:`list` of :obj:`tuple`): List of state-action-reward
            tuples within the generated episode.

    Raises:
        None
    """
    episode = []

    state = env.reset() # this gives a random starting state

    nA = env.action_space.n

    while True:

        probs = [epsilon / nA, 1 - epsilon + (epsilon / nA)]
        np.random.choice(np.arange(nA), p=probs)

        # Chose between exploration and exploitation
        choice = np.random.choice(np.arange(nA), p=probs)

        # Exploitation case, we use the currently best policy
        if choice == 1 and state in Q:
            action = np.argmax(Q[state]) # <- eval policy based on Q table

        # Exploration case
        else:
            # Use the other rule
            action = np.random.choice(np.arange(nA))

        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state

        if done:
            break

    return episode

def mc_control_glie(env, num_episodes, eps_start=1.0, alpha=0.02, gamma=1.0,
                    eps_decay=.99999, eps_min=0.05):
    """Algorithm to solve the mc control problem using an epsilon-greedy GLIE
    (greedy in the limit of infinite  exploration) strategy and a every-visit
    constant-alpha approach.

    Args
        env (:obj: `gym.env`): An OpenAI Black Jack env instance.
        num_episodes (int): Number of episodes to generate.
        eps_start (float): Starting value for epsilon factor in epsilon-greedy
            policy evaluation.
        alpha (float): Fraction by which the Q-table shall be updated at every
            new visit to a state-action pair and a new final reward.
        gamma (float): The discount factor gramma.
        eps_decay (float): Amount by which epsilon shall be reduced after each
            episode.
        eps_min (float): The minimum value for epsilon.

    Returns:
        policy (:obj:`dict`): Policy dictionary of the form policy[state]
        Q (:obj:`defaultdict`): The Q-table which is a dictionary ala
            Q[state][action] = reward.

    Raises:
        None

    """
    nA = env.action_space.n

    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    policy = {}

    epsilon = eps_start

    # Loop over episodes
    for i_episode in range(1, num_episodes+1):

        # Generate episode according to current policy
        episode = generate_episode_using_policy(env, epsilon, Q)

        G = sum([el[2]*(gamma**i) for i, el in enumerate(episode)])

        # Loop over state-action-reward tubples in episode
        for sar_tuple in episode:

            state  = sar_tuple[0]
            action = sar_tuple[1]

            # Update the Q-table using an every-visit
            # constant alpha GLIE method
            Q[state][action] += alpha*(G-Q[state][action])

        # Monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        if epsilon > eps_min:
            epsilon -= eps_decay

    # Output the final policy
    for s, v in Q.items():
        policy[s] = np.argmax(v)

    return policy, Q


if __name__ == '__main__':
    """Plays Black Jack.

    Usage:
        python monte_carlo_black_jack.py
    """

    # Generate an instance of the Gym Blackjack environment
    env = gym.make('Blackjack-v0')

    # Obtain the action-value function
    Q = mc_prediction_q(env, 500000, generate_black_jack_episode_limit_stochastic)

    # Obtain the estimated optimal policy and action-value function
    policy, Q = mc_control_glie(env, 1000000, 1.0, 0.02)
