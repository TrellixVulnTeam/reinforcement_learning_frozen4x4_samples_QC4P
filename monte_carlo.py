import random
import numpy as np
import gym

'''
Monte carlo on policy first visit
'''
class MonteCarloOnPolicyFirstVisit:
    def __init__(self, env, gamma=0.9, epsilon=0.2):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.N_STATES = env.env.nS
        self.N_ACTIONS = env.env.nA

    def train(self, episodes=10000):
        """

        :param episodes: count of episodes
        :return: optimal policy
        """

        self._initialize_params()

        for ep_counter in range(episodes):  # repeat for each episode
            if ep_counter % 1000 == 0:
                print(ep_counter)

            episode = self._generate_episode()  # generate policy

            self._train_on_episode(episode)  # train on episode
        return self._create_dictionary_state_best_action()

    '''
    private methods
    '''
    def _create_dictionary_state_best_action(self):
        dict = {}
        for key in self.policy:
            v = []
            for act_key in self.policy[key]:
                v.append(self.policy[key][act_key])

            dict[key] = np.argmax(v)
        return dict

    def _train_on_episode(self, episode):
        G = 0  # initialize cumulative discounted reward
        episode.reverse()
        length = len(episode)

        for t in range(length-1):
            s_t, a_t, r_t = episode[t]  # get state, action and reward of t episode
            _, _, r_t_plus = episode[t+1]
            G += self.gamma*G + r_t_plus   # update G

            state_action = (s_t, a_t)

            # check is (state, action) was prev in episode
            if state_action not in [(x[0], x[1]) for x in episode[0:t]]:  # if not then

                # append G to returns
                if self.returns.get(state_action):
                    self.returns[state_action].append(G)
                else:
                    self.returns[state_action] = [G]

                # Q(s, a) <- average returns for (state, action)
                self.Q[s_t][a_t] = sum(self.returns[state_action]) \
                                   / len(self.returns[state_action])

                # get A_star (max action value for state)
                Q_list = list(map(lambda x: x[1], self.Q[s_t].items()))
                indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
                A_star = random.choice(indices)

                # calculate new policy
                for a in self.policy[s_t].items():
                    if a[0] == A_star:
                        self.policy[s_t][a[0]] = 1 - self.epsilon + (self.epsilon / abs(sum(self.policy[s_t].values())))
                    else:
                        self.policy[s_t][a[0]] = (self.epsilon / abs(sum(self.policy[s_t].values())))

    def _generate_episode(self):
        """
        generate episode based on current policy
        :param policy: current policy
        :return: 2d array ([[state, action, reward]])
        """
        episode = []
        game_over = False

        self.env.reset()

        while not game_over:
            state = self.env.env.s

            timestep = []
            timestep.append(state)

            n = random.uniform(0, sum(self.policy[state].values()))
            top_range = 0
            for prob in self.policy[state].items():
                top_range += prob[1]
                if n < top_range:
                    action = prob[0]
                    break


            _, reward, game_over, _ = self.env.step(action)
            # self.env.render()

            timestep.append(action)
            timestep.append(reward)
            episode.append(timestep)

        return episode

    def _create_random_policy(self):
        """
        :return: arbitrary policy
        """
        policy = {}

        for key in range(self.N_STATES):
            p = {}
            for action in range(self.N_ACTIONS):
                p[action] = 1 / self.N_ACTIONS
            policy[key] = p
        return policy

    def _create_state_action_dictionary(self, policy):
        """
        :param policy: arbitrary policy
        :return: state action dictionary
        """
        Q = {}
        for key in policy.keys():
            Q[key] = {a: 0.0 for a in range(self.N_ACTIONS)}
        return Q

    def _initialize_params(self):
        """
        initialize returns, policy and state action table (as dictionary)
        """
        self.policy = self._create_random_policy()
        self.Q = self._create_state_action_dictionary(self.policy)
        self.returns = {}


