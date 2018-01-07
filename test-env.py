import gym
import gym_mnist
from gym_mnist.resources.getch import getch

import argparse
parser = argparse.ArgumentParser("Interactively play one of the gym_mnist "
                                 "games, using keys 1 through 9\n")
parser.add_argument('game', choices=gym_mnist.envnames,
                    help="Which of the gym_mnist games you'd like to play.")
args = parser.parse_args()

env = gym.make('rotgame-v0')

# wasd = ['w','a','s','d','c']
flipgame_ctrls = list('123456789')
intkeys = [0,1,2,3,4]

input_keys = flipgame_ctrls

keys_to_action = lambda inp: next(i for i in range(len(input_keys)) if
                                  input_keys[i] == inp[0]) # single key input


obs = env.reset()
total_reward = 0
for i in range(20):
    obs = env.reset()
    done = False
    total_reward = 0
    print ("Starting new game.")
    env.render(mode='human')
    c = 1
    while not done:
        action = getch()
        while action not in input_keys:
            if action == 'q':
                exit()
            print ("Action must be one of {}. To quit, press 'q'.".format(
                input_keys))
            action = getch()
        action = keys_to_action(action)
        obs, reward, done,  _ = env.step(action)
        c += 1
        env.render(mode='human')
        total_reward += reward
    print ("Episode reward: %d"%total_reward)
