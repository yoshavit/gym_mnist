from gym.envs.registration import register
import numpy as np

envnames = ['mnist-v0',
            'mnist-linear-v0',
            'mnist-multigoal-v0',
            'mnist-9game-v0',
            'flipgame-v0',
            'rotgame-v0',
            'mnist-9game-simple-v0',
            # 'blockwalker-v0',
            # 'blockwalker-multicolored-v0',
            # 'blockwalker-random-v0',
           ]


register(
    id='mnist-v0',
    entry_point='gym_mnist.envs:MNISTEnv',
)
register(
    id='mnist-linear-v0',
    entry_point='gym_mnist.envs:MNISTEnv',
    kwargs={'actions_type': "linear"}
)
register(
    id='mnist-multigoal-v0',
    entry_point='gym_mnist.envs:MNISTEnv',
    kwargs={'target_digits': [0, 4, 8]}
)
register(
    id='mnist-9game-v0',
    entry_point='gym_mnist.envs:MNIST9GameEnv',
    max_episode_steps=300,
)
register(
    id='flipgame-v0',
    entry_point='gym_mnist.envs:FlipgameEnv',
    max_episode_steps=100,
)
register(
    id='rotgame-v0',
    entry_point='gym_mnist.envs:RotgameEnv',
    max_episode_steps=100,
)
simple_init_fn = lambda: np.random.permutation([0]+[1]*4+[2]*4).reshape([3, 3])
# simple_init_fn = lambda: np.random.permutation([0]+[1] + [2]*2).reshape([2, 2])
register(
    id='mnist-9game-simple-v0',
    max_episode_steps=100,
    entry_point='gym_mnist.envs:MNIST9GameEnv',
    kwargs={"init_fn":simple_init_fn}
)
# register(
    # id='blockwalker-v0',
    # entry_point='gym_mnist.envs:BlockWalkerEnv',
# )
# register(
    # id='blockwalker-multicolored-v0',
    # entry_point='gym_mnist.envs:MulticoloredBlockWalkerEnv',
# )
# register(
    # id='blockwalker-random-v0',
    # entry_point='gym_mnist.envs:MulticoloredRandomBlockWalkerEnv',
    # max_episode_steps=300,
# )
