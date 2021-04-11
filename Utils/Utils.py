import time
import gym
import csv
import matplotlib.pyplot as plt

'''
Utils for frozen lake
'''

class Utils:
    @staticmethod
    def test_game(env, policy, sleep=False):
        """
        start one game with visualization
        :param env: gym _env
        :param policy: policy in dict format {state: best action}
        :param sleep: turn on/off visualization
        :return: game reward
        """
        state = env.reset()
        if sleep:
            env.render()
            time.sleep(1)

        is_done = False
        reward = 0

        while not is_done:
            action = policy[state]
            state, reward, is_done, _ = env.step(int(action))

            if sleep:
                env.render()
                time.sleep(1)

        return reward

    @staticmethod
    def test_on_eps(env, policy, episodes):
        """
        test algorithm on count of eps, and print results
        :param env: gym _env
        :param policy: policy in dict format {state: best action}
        :param episodes: the number of episodes on which the algorithm is testing
        :returns wins, episodes
        """
        wins = 0

        for _ in range(episodes):
            reward = Utils().test_game(env, policy)

            if reward == 1:
                wins += 1

        print("Won {} out of {} episodes".format(wins, episodes))
        return wins, episodes

    @staticmethod
    def create_csv(policy, name):
        """
        create csv with policy
        :param policy: policy in dict format {state: best action}
        :param name: name of file
        """
        with open(name, 'w', newline='') as file:
            writer = csv.writer(file)
            for key in policy.keys():
                row = [key]
                row.append(policy[key])
                writer.writerow(row)

    @staticmethod
    def read(name):
        """

        :param name: name of file
        :return: policy in dict format {state: best action}
        """
        policy_dict = {}
        with open(name) as file:
            csv_reader = csv.reader(file, delimiter=',')
            for row in csv_reader:
                policy_dict[row[0]] = row[1:]
        return policy_dict

    @staticmethod
    def build_average_reward_length_plots(logs):
        avgs = []
        lens = []
        for index, reward in enumerate(logs):
            length = len(reward)
            avg = sum(reward) / length
            avgs.append(avg)
            lens.append(length)

        fig, axs = plt.subplots(1, 2)
        axs[0].plot(avgs)
        axs[0].set_title('Average Reward per Episode')
        axs[1].plot(lens, 'tab:orange')
        axs[1].set_title('Episode_Length')
        plt.show()

