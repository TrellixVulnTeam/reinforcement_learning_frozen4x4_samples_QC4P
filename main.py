import value_iteration as vi
import policy_iteration as pi
import gym

env = gym.make('FrozenLake-v0')
gamma = 0.9
theta = 0.001


policy_iter = pi.PolicyIteration(env)
policy_iter.policy_iteration(gamma, theta)

# env.reset()
# env.render()

# value_iter = vi.ValueIteration(env)
# value_iter.value_iteration(gamma, theta)
# value_iter.simulate_value_iteration(10000)

