import tensorflow as tf
import numpy as np
import gym
import gym_mnist
from gym_mnist.resources.getch import getch

env = gym.make('mnist-v0')

observation = env.reset()
with tf.Session() as sess:
    sess.run(initializer)
    obs = env.reset()
    total_reward = 0 
    for i in range(20):
        obs = env.reset()
        done = False
        total_reward = 0
        print ("Starting new game.")
        while not done:
            action = getch()
            while action not in [0,1,2]:
                print ("Action must be 0 (+1), 1 (-1), or 2 (x2)")
                action = getch()
            obs, reward, done,  _ = env.step(action)
            env.render()
            total_reward += reward
        print ("Episode reward: %d"%total_reward)
