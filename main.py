from AlgorithmImplementation.value_iteration import ValueIteration
from AlgorithmImplementation.monte_carlo_exploring import FirstVisitMonteCarloPrediction
from AlgorithmImplementation.q_learning import QLearning
import Utils.Utils
import gym






def value_iteration_experiment(env, utils):
    # gamma_vals = [0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.75, 0.85, 0.9, 0.9, 0.9]
    # theta_vals = [0.1, 0.05, 0.01, 0.005, 0.001, 0.1, 0.1, 0.1, 0.1, 0.0001]
    # count_of_exp = 10
    # for expirement_number in range(count_of_exp):
    # best_res = 0
    # best_agent_policy = dict()
    # for _ in range(20):
    test_on_ep = 1000
    value_iteration = ValueIteration(env)
    gamma = 0.9
    theta = 0.001
    vl_policy = value_iteration.value_iteration(gamma, theta)
    vl_win, _ = utils.test_on_eps(env, vl_policy, test_on_ep)

    print_res(vl_win, test_on_ep)
        #     if vl_win / test_on_ep > best_res:
        #         best_res = vl_win / test_on_ep
        #         best_agent_policy = vl_policy
        # print(best_agent_policy)
        # print(best_res)
        # utils.create_csv(best_agent_policy, f"./experiments/k/experiment_{expirement_number}.csv")


def monte_carlo_experiments(env, utils):
    # best_res = -1
    # all_res = []
    # best_agent_policy = dict()
    # for _ in range(10):
    gamma = 0.97
    epsilon = 0.2
    explore_first = 500
    episode_count = 10000
    monte_carlo = FirstVisitMonteCarloPrediction(env,
                                                 gamma=gamma,
                                                 epsilon=epsilon,
                                                 explore_first=explore_first)
    mc_policy = monte_carlo.train(episodes=episode_count)

    test_on_ep = 1000
    mc_win, _ = utils.test_on_eps(env, mc_policy, test_on_ep)

    # print_res(mc_win, test_on_ep)

        # if mc_win / test_on_ep >= best_res:
        #     best_res = mc_win / test_on_ep
        #     best_agent_policy = mc_policy
        # all_res.append(mc_win / test_on_ep)

    # utils.test_game(env, best_agent_policy, sleep=False)
    # print("policy is {}".format(best_agent_policy))
    # print(best_res)
    # print("average {}".format(sum(all_res) / len(all_res)))
    # utils.build_average_reward_length_plots(monte_carlo.episode_logs)
    # utils.create_csv(best_agent_policy, "./experiments/monte_carlo/experiment_10.csv")


def q_learning_experiment(env, utils):
    # best_res = -1
    # all_res = []
    # best_agent_policy = dict()
    # for _ in range(1):
    gamma = 0.98
    epsilon = 1
    lerning_rate = 0.1
    epsilon_min = 0.01
    epsilon_decay = 0.99995
    q_learning = QLearning(env,
                           gamma=gamma,
                           epsilon=epsilon,
                           learning_rate=lerning_rate,
                           epsilon_min=epsilon_min,
                           epsilon_decay=epsilon_decay)

    episode_count = 10000
    q_policy = q_learning.train(episodes=episode_count)

    test_on_ep = 1000
    q_win, _ = utils.test_on_eps(env, q_policy, test_on_ep)

    # print_res(q_win, test_on_ep)

    # if q_win / test_on_ep >= best_res:
    #     best_res = q_win / test_on_ep
    #     best_agent_policy = q_policy
    # all_res.append(q_win / test_on_ep)

    # utils.test_game(env, best_agent_policy, sleep=False)
    # print("policy is {}".format(best_agent_policy))
    # print(best_res)
    # print("average {}".format(sum(all_res) / len(all_res)))
    # utils.build_average_reward_length_plots(q_learning.episode_logs)
    # utils.create_csv(best_agent_policy, "./experiments/q_learning/experiment_11.csv")


def print_res(win, episodes):
    print("Won {} out of {} episodes".format(win, episodes))




def main():
    env = gym.make('FrozenLake-v0')

    utils = Utils.Utils.Utils()
    # value_iteration_experiment(env, utils)
    # monte_carlo_experiments(env, utils)
    # q_learning = AlgorithmImplementation.q_learning.QLearning(env)
    # mc = FirstVisitMonteCarloPrediction(env)
    # pi = mc.train(30000)
    # q_learning_experiment(env, utils)
    # test_on_ep = 1000
    # mc_win, _ = utils.test_on_eps(env, pi, test_on_ep)

    # q_policy = q_learning.train(30000)
    # print(q_policy)
    # vl_policy = value_iteration.value_iteration(0.9, 0.0005)
    # utils.build_average_reward_length_plots(q_learning.episode_logs)
    # utils.build_average_reward_length_plots(mc.episode_logs)

    # test_on_ep = 1000
    # q_win, _ = utils.test_on_eps(env, q_policy, test_on_ep)
    # print_res("q_learning", q_win, test_on_ep)
    # print_res("monte carlo", mc_win, test_on_ep)

    '''calculate time '''
    import time

    tic = time.perf_counter()
    monte_carlo_experiments(env, utils)
    # q_learning = AlgorithmImplementation.q_learning.QLearning(env)
    # value_iteration_experiment(env, utils)

    # q_learning_experiment(env, utils)

    toc = time.perf_counter()
    print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")

    ''' read policy '''
    # name = ""
    # policy = utils.read(name)




if __name__ == '__main__':
    main()
