import AlgorithmImplementation.q_learning
import AlgorithmImplementation.monte_carlo
import AlgorithmImplementation.value_iteration
import Utils.Utils
import gym




def main():
    def print_res(name, win, ep_count):
        print("{} ||| algorithm won {} out of {} episodes ".format(name, win, ep_count))
    env = gym.make('FrozenLake-v0')

    utils = Utils.Utils.Utils()

    q_learning = AlgorithmImplementation.q_learning.QLearning(env)
    monte_carlo = AlgorithmImplementation.monte_carlo.MonteCarloOnPolicyFirstVisit(env)
    # value_iteration = AlgorithmImplementation.value_iteration.ValueIteration(env)


    q_policy = q_learning.train(30000)
    # print(q_policy)
    mc_policy = monte_carlo.train(30000)
    # vl_policy = value_iteration.value_iteration(0.9, 0.0005)
    # utils.build_average_reward_length_plots(q_learning.episode_logs)
    utils.build_average_reward_length_plots(monte_carlo.episode_logs)

    test_on_ep = 1000
    # q_win, _ = utils.test_on_eps(env, q_policy, test_on_ep)
    # mc_win, _ = utils.test_on_eps(env, q_policy, test_on_ep)
    # vl_win, _ = utils.test_on_eps(env, vl_policy, test_on_ep)
    # print_res("q_learning", q_win, test_on_ep)
    # print_res("monte carlo", mc_win, test_on_ep)
    # print_res("value iteration", vl_win, test_on_ep)




if __name__ == '__main__':
    main()