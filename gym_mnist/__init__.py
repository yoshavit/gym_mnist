from gym.envs.registration import register

register(
    id='mnist-v0',
    entry_point='gym_mnist.envs:MNISTEnv',
)
register(
    id='mnist-full-v0',
    entry_point='gym_mnist.envs:MNISTEnv',
    kwargs={'full_mnist':True},
)
register(
    id='mnist-multiaction-v0',
    entry_point='gym_mnist.envs:MNISTMultiactionEnv',
)
