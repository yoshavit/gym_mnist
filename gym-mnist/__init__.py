from gym.envs.registration import register

register(
    id='mnist-v0',
    entry_point='gym_mnist.envs:MNISTEnv',
)
register(
    id='mnist-multiaction-v0',
    entry_point='gym_mnist.envs:MNISTMultiactionEnv',
)
