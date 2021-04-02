import value_iteration as vi
import gym

env = gym.make('FrozenLake-v0')
gamma = 0.99
theta = 0.000001

value_iter = vi.ValueIteration(env)
value_iter.value_iteration(gamma, theta)
value_iter.simulate_value_iteration(100)
