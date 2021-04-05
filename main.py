import value_iteration as vi
import policy_iteration as pi
import gym


class Utils:
    def save(self, policy, path=""):
        pass

    def load(self, path):
        pass

    def test_policy(self, policy):
        wins = 0
        r = 1000
        for i in range(r):
            w = self.run_game(env, policy)[-1][-1]
            if w == 1:
                wins += 1

        print("win {} of {}".format(wins, r))
        return wins / r



env = gym.make('FrozenLake-v0')
gamma = 0.9
theta = 0.001


policy_iter = pi.PolicyIteration(env)
policy_iter.policy_iteration(gamma, theta)
print(policy_iter)
# env.reset()
# env.render()

# value_iter = vi.ValueIteration(env)
# value_iter.value_iteration(gamma, theta)
# value_iter.simulate_value_iteration(10000)

