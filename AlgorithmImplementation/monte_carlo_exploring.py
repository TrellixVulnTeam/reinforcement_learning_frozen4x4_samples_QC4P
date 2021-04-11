import numpy as np
from itertools import product

'''
Monte carlo first visit prediction
'''

class FirstVisitMonteCarloPrediction:
    def __init__(self, env, gamma=0.9, epsilon=0.2, explore_first=1000):
        self._env = env
        self._gamma = gamma
        self._epsilon = epsilon
        self._N_STATES = env.env.nS
        self._N_ACTIONS = env.env.nA
        self.episode_logs = []
        self._explore_first = explore_first

    def train(self, episodes=10000):
        """

        :param episodes: count of episodes
        :return: optimal policy
        """
        self._initialize_params()

        self._ep_counter = 0

        for ep_counter in range(episodes):
            self._train_episode()
            self._ep_counter += 1

        return self._policy


    def _initialize_params(self):
        """
        initialize paramas
        """
        # arbitrary policy
        self._policy = {state: np.random.choice(self._env.action_space.n) for state in range(self._N_STATES)}

        # Q(s, a) arbitrary state action dict
        self._Q = {state: {action: 0 for action in range(self._N_ACTIONS)}
                   for state in range(self._N_STATES)}
        self._returns = {(s, a): [] for s, a in product(range(self._N_STATES), range(self._N_ACTIONS))}

    def _train_episode(self):
        # array of visited state
        visited_steps = set()

        # generate episode
        episode = self._generate_episode()

        # transform episode [state, action, reward] to [state, action, g]
        # g is cumulative discounted reward
        episode_sag = self._episode_sag(episode)

        for t in episode_sag:
            s_t, a_t, g_t = t  # step values

            state_action = (s_t, a_t)  # state action tuple

            if state_action not in visited_steps:  # check is step was visited
                # if not then

                self._returns[state_action].append(g_t)  # add cum. reward for state
                self._Q[s_t][a_t] = np.mean(self._returns[state_action])  # add to q table avg. rew. for state_action
                visited_steps.add(state_action)  # write that step is visited

        # evaluate new policy
        for state in self._policy.keys():
            self._policy[state] = max(self._Q[state], key=self._Q[state].get)



    def _episode_sag(self, episode):

        G = 0  # cumulative discounted reward
        state_action_gain = []
        for s_t, a_t, r_t in reversed(episode):
            G = r_t + self._gamma * G
            state_action_gain.append([s_t, a_t, G])
        return reversed(state_action_gain)

    def _generate_episode(self):
        self._env.reset()
        is_done = False

        episode = []
        # for logging
        episode_rewards = []
        while not is_done:
            state = self._env.env.s
            action = self._propose_action(self._policy[state])
            new_state, reward, is_done, _ = self._env.step(action)
            episode.append([state, action, reward])

            # for logging
            episode_rewards.append(reward)

        self.episode_logs.append(episode_rewards)

        return episode


    def _propose_action(self, action):
        # for exploration start
        epsilon = self._epsilon if self._ep_counter > self._explore_first else 1

        if np.random.random() > epsilon:
            return action
        else:
            return self._env.action_space.sample()

