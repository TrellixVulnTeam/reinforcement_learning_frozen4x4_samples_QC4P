import gym
import numpy as np


class ValueIteration:
    def __init__(self, env):
        self.env = env
        self.N_STATES = env.env.nS
        self.N_ACTIONS = env.env.nA
        self.trans_probs = env.env.P
        self._final_policy = None

    def value_iteration(self, gamma, theta):
        """

        :param gamma: discount factor
        :param theta: threshold
        :return: optimal policy
        """

        V = np.zeros(self.N_STATES)  # empty array with value functions

        while True:
            delta = 0

            for state in range(self.N_STATES):
                prev_value = V[state]

                V[state] = self._bellman_optimization_equality(state, V, gamma)
                delta = max(delta, abs(prev_value - V[state]))  # difference between value function of prev and current

            if delta < theta:
                break

        final_policy = self._generate_final_policy(V, gamma)

        self._final_policy = final_policy
        return final_policy

    def _bellman_optimization_equality(self, state, V, gamma):
        """

        :param state:
        :param V: array of value functions
        :param gamma: disocunt factor
        :return: best action
        """

        action_values = list()

        for action in range(self.N_ACTIONS):
            trans_prob = self.trans_probs[state][action]
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
        policy = np.zeros(self.N_STATES)
        for state in range(self.N_STATES):
            action_values = list()
            for action in range(self.N_ACTIONS):
                trans_prob = self.trans_probs[state][action]
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

    def _simulate_value_iteration(self, episodes):
        if self._final_policy is None:
            raise Exception("You need to train first")
        else:
            wins = 0
            for ep in range(1, episodes + 1):
                obs = self.env.reset()
                print("episode: {}".format(ep))

                for turn in range(100):
                    obs, reward, done, info = self.env.step(int(self._final_policy[obs]))

                    if done:
                        if reward == 1:
                            wins += 1
                        self.env.render()
                        break

            print(" agent succeeded to reach goal {} out of {} Episodes using this policy ".format(wins, episodes))
