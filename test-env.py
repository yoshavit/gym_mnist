import numpy as np
import gym
import gym_mnist
from gym_mnist.resources.getch import getch

env = gym.make('blockwalker-multicolored-v0')

wasd = ['w','a','s','d', 'c']
intkeys = [0,1,2,3]

input_keys = wasd

obs = env.reset()
total_reward = 0
for i in range(20):
    obs = env.reset()
    done = False
    total_reward = 0
    print ("Starting new game.")
    env.render()
    while not done:
        action = getch()
        while action not in input_keys:
            if action == 'q':
                exit()
            print ("Action must be one of {}. To quit, press 'q'.".format(
                input_keys))
            action = getch()
        action = next(i for i in range(len(input_keys))
                      if input_keys[i] == action)
        obs, reward, done,  _ = env.step(action)
        env.render()
        total_reward += reward
    print ("Episode reward: %d"%total_reward)
