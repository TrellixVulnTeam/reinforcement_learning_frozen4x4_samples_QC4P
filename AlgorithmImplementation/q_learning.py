import random
import numpy as np

'''
Q-decay learning
'''
class QLearning:
    def __init__(self, env, gamma=0.8, epsilon=1, learning_rate=0.1,
                 epsilon_min=0.01, epsilon_decay=0.99995):
        """

        :param env: gym _env
        :param gamma: discount factor
        :param epsilon: e-policy threshold
        :param learning_rate:
        :param epsilon_min: minimum _epsilon value
        :param epsilon_decay:
        """
        self._env = env
        self._N_STATES = self._env.env.nS
        self._N_ACTIONS = self._env.env.nA
        self._gamma = gamma
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

        self.episode_logs = []  # 2d array with episodes reward by step

    def train(self, episodes=10000):
        """

        :param episodes: count of episodes
        :return: best action for state dict
        """
        self._create_state_action_dictionary()

        for ep_counter in range(episodes):
            self._train_episode()

        return self._transform_q_table_to_dict()


    '''
    private methods
    '''
    def _train_episode(self):
        """
        train on episode
        """
        state = self._env.reset()
        is_done = False

        # param for logs
        episode_rewards = []

        while not is_done:
            action = self._propose_action(state)  # get action
            new_state, reward, is_done, _ = self._env.step(action)

            self._learn(state, new_state, action, reward)
            state = new_state

            # logging
            episode_rewards.append(reward)

        # add logs of episode to logs
        self.episode_logs.append(episode_rewards)

        # reduce _epsilon
        if self._epsilon >= self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def _create_state_action_dictionary(self):
        self.actions = ['N', 'E', 'S', 'W']

        self.q_table = np.zeros(
            (self._N_STATES, self._N_ACTIONS))

    def _learn(self, old_state, new_state, action, reward):
        old_value = self.q_table[old_state][action]
        next_value = np.max(self.q_table[new_state])
        new_q_value = self._compute_new_q_value(old_value, reward, next_value)
        self.q_table[old_state][action] = new_q_value

    def _compute_new_q_value(self, old_val, reward, next_value):
        return (old_val + self._learning_rate * (reward + self._gamma * next_value - old_val))

    def _propose_action(self, state):
        """

        :param state: state of _env
        :return: action for state
        """
        row = self.q_table[state]
        max_Q = np.max(row)

        # if random less than _epsilon => random action
        if np.random.random() < self._epsilon:
            return random.randrange(0, self._N_ACTIONS, 1)

        # else return best action for state from q_table
        indexes = np.where(row == max_Q)[0]

        if len(indexes) > 1:
            action = np.random.choice(indexes)
            return action
        else:
            return indexes[0]

    def _transform_q_table_to_dict(self):
        q_dict = {}
        for state in range(self._N_STATES):
            q_dict[state] = np.argmax(self.q_table[state])
        return q_dict
