import numpy as np


class ValueIteration:
    def __init__(self, env):
        """

        :param env: gym environment
        """
        self._env = env
        self._N_STATES = env.env.nS
        self._N_ACTIONS = env.env.nA
        self._trans_probs = env.env.P
        self._final_policy = None

    def value_iteration(self, gamma, theta):
        """
        value iteration train
        :param gamma: discount factor
        :param theta: threshold
        :return: optimal policy (dict where key is state, value is action)
        """

        V = np.zeros(self._N_STATES)  # empty array with value functions

        while True:
            delta = 0

            for state in range(self._N_STATES):
                prev_value = V[state]

                V[state] = self._bellman_optimization_equality(state, V, gamma)
                delta = max(delta, abs(prev_value - V[state]))  # difference between value function of prev and current

            if delta < theta:
                break

        self._final_policy = self._generate_final_policy(V, gamma)

        return self._final_policy

    '''
    private methods
    '''

    def _bellman_optimization_equality(self, state, V, gamma):
        """
        :param state:
        :param V: array of value functions
        :param gamma: discount factor
        :return: best action
        """

        action_values = list()

        for action in range(self._N_ACTIONS):
            trans_prob = self._trans_probs[state][action]
            action_value = 0

            for i in trans_prob:
                s_ = i[1]
                p = i[0]
                r = i[2]
                action_value += p * (r + gamma * V[s_])
            action_values.append(action_value)

        return max(action_values)

    def _generate_final_policy(self, V, gamma):
        """

        :param V: array of value functions
        :param gamma: discount factor
        :return: optimal policy
        """
        policy = {}
        for state in range(self._N_STATES):
            action_values = list()
            for action in range(self._N_ACTIONS):
                trans_prob = self._trans_probs[state][action]
                action_value = 0

                for i in trans_prob:
                    s_ = i[1]
                    p = i[0]
                    r = i[2]
                    action_value += p * (r + gamma * V[s_])
                action_values.append(action_value)
            best_action = np.argmax(action_values)
            policy[state] = best_action
        return policy
