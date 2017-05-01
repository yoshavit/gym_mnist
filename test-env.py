import numpy as np
import gym
import time
import gym_mnist
from gym_mnist.resources.getch import getch
from pynput import keyboard

env = gym.make('mnist-v0')

wasd = ['w','a','s','d', 'c']
intkeys = [0,1,2,3, 4]

input_keys = wasd

keys_down = []
def on_press(key):
    print("HI!")
    if key.char in input_keys:
        if key.char not in keys_down:
            keys_down.append(key.char)
    if key.char == 'q':
        exit()

def on_release(key):
    if key.char in input_keys:
        if key.char in keys_down:
            keys_down.remove(key.char)

keys_to_action = lambda inp: next(i for i in range(len(input_keys)) if
                                input_keys[i] == inp[0]) # single key input
# keys_to_action_dict = env.env.get_keys_to_action()
# keys_to_action = lambda keys: keys_to_action_dict[tuple(sorted(keys))]


obs = env.reset()
total_reward = 0
with keyboard.Listener(on_press=on_press,
                       on_release=on_release) as listener:
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
            # while len(keys_down) != 1:
                # if len(keys_down) > 1:
                    # print("Only one at a time! To quit press q")
                # time.sleep(.3)
            # action = keys_to_action(keys_down)
            action = keys_to_action(action)
            obs, reward, done,  _ = env.step(action)
            c += 1
            env.render(mode='human')
            total_reward += reward
        print ("Episode reward: %d"%total_reward)
