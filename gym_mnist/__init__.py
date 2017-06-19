from gym.envs.registration import register

register(
    id='mnist-v0',
    entry_point='gym_mnist.envs:MNISTEnv',
)
register(
    id='mnist-barebones-v0',
    entry_point='gym_mnist.envs:BarebonesMNISTEnv',
)
register(
    id='mnist-9game-v0',
    entry_point='gym_mnist.envs:MNIST9GameEnv',
)
register(
    id='blockwalker-v0',
    entry_point='gym_mnist.envs:BlockWalkerEnv',
)
register(
    id='blockwalker-multicolored-v0',
    entry_point='gym_mnist.envs:MulticoloredBlockWalkerEnv',
)
register(
    id='blockwalker-random-v0',
    entry_point='gym_mnist.envs:MulticoloredRandomBlockWalkerEnv',
    max_episode_steps=300,
)
