import gym
import numpy as np


class MonteCarlo:
    def __init__(self, env):
        self.N_STATES = env.env.nS
        self.N_ACTIONS = env.env.nA
        self.trans_probs = env.env.P
        self.env = env

    def monte_carlo(self, episodes_count=10000, gamma=0.8, epsilon=0.2):
        """

        :param episodes_count: count of episodes
        :param gamma: discount factor
        :param epsilon:
        :return: optimal policy
        """
        self._init_monte_carlo()

        # iterate for each episode
        for _ in range(episodes_count):

            # generate an episode policy
            ep_policy = self._generate_policy()  # [[s, a, r]...]
            print(ep_policy)
            G = 0  # initialize G ()

            # iterate for each step of episode
            for t in reversed(range(len(ep_policy))):
                G = self._calculate_new_G(G, gamma, ep_policy[t][2])

                # check is S[t] and A[t] wasn't in previously policys in current episode
                if (ep_policy[t][0], ep_policy[t][1]) not in [(prev_policy[0], prev_policy[1]) for prev_policy in
                                                              ep_policy[:t]]:
                    self._add_G_to_returns(G, ep_policy[t][0], ep_policy[t][1])
                    self.Q[ep_policy[t][0]][ep_policy[t][1]] = sum(self.returns[(ep_policy[t][0], ep_policy[t][1])]) / \
                                                               len(self.returns[(ep_policy[t][0], ep_policy[t][1])])
                    A_ = np.argmax(self.Q[ep_policy[t][0]])

                    self.pi[ep_policy[t][0]] = epsilon / self.N_ACTIONS
                    self.pi[ep_policy[t][0]][A_] += 1 - epsilon

        for s in range(self.N_STATES):
            print("S{}: {}".format(s, np.argmax(self.pi[s])))

        return self.pi

    def _generate_policy(self):
        """
        generate random policy
        :return: 2D array with [obs, action, reward, dont]
        """
        episode_policy = list()
        self.env.reset()
        while True:
            action = self.env.action_space.sample()
            obs, reward, done, _ = self.env.step(action)  # obs, reward, done, info
            episode_policy.append([obs, action, reward, done])
            self.env.render()
            if done:
                return episode_policy

    def _init_monte_carlo(self):
        self.Q = np.zeros([self.N_STATES, self.N_ACTIONS])
        self.returns = dict()  # with key tuple of (state, action)
        self.pi = np.full((self.N_STATES, self.N_ACTIONS), 0.25)

    def _calculate_new_G(self, G, gamma, r):
        return gamma * G + r

    def _add_G_to_returns(self, G, state, action):
        if (state, action) in self.returns.keys():
            self.returns[(state, action)].append(G)
        else:
            self.returns[(state, action)] = [G]


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env.reset()
    monte_carlo = MonteCarlo(env)
    monte_carlo.monte_carlo()
