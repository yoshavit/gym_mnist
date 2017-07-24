import gym
from gym import spaces
import numpy as np
import os
import random
from scipy.ndimage import imread
import cv2
# import matplotlib.pyplot as plt

IMG_SIDEWIDTH = 28
CURRENT_DIR_TO_MNIST_DIR = "/../resources/"
PARTIAL_DATASET_SIZE = 500

class MNISTEnv(gym.Env):
    '''
    This environment encodes a game with a reward for reaching
    a particular number (default 0) using only elementary operations
    (specified by actions_type as either "complex" or "linear") always mod 10.
    The reward for each action is 0,
    until we reach the goal, at which point it is 1.0
'''
    metadata = {'render.modes': ['human', 'training']}
    action_space = spaces.Discrete(4)
    observation_space = spaces.Box(low=0, high=255,
                                   shape=(IMG_SIDEWIDTH,IMG_SIDEWIDTH, 1))
    def __init__(self, target_digits=0, full_mnist=False,
                 actions_type="complex"):
        # first, load all the MNIST images into RAM
        self.filename_library = [[] for k
                              in range(10)]
        CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
        MNIST_DIR = CURRENT_DIR + CURRENT_DIR_TO_MNIST_DIR
        for i in range(10):
            DIGIT_DIR = MNIST_DIR + "digit%d/" % i
            digit_filenames = [DIGIT_DIR + name for name in
                               os.listdir(DIGIT_DIR)]
            self.filename_library[i] = digit_filenames
        # Now, decide on the target digit
        if target_digits is None:
            self.target_digits = list(range(10))
        elif isinstance(target_digits, list):
            self.target_digits = target_digits
        else:
            self.target_digits = [target_digits]
        self.target_digit = random.choice(self.target_digits)
        self.goal_image = self._get_image_from_digit(self.target_digit)
        self.actions_type = actions_type
        assert self.actions_type in ["linear", "complex"]


    def _step(self, action):
        old_digit = self.current_digit
        if self.actions_type == "complex":
            if action == 0:
                pass
            elif action == 1:
                self.current_digit = (self.current_digit + 1) % 10
            elif action == 2:
                self.current_digit = (self.current_digit * 2) % 10
            elif action == 3:
                self.current_digit = (self.current_digit * 3) % 10
            else:
                raise ValueError("Action must be encoded as an int between 0 and 3")
        elif self.actions_type == "linear":
            if action == 0 or action == 2:
                self.current_digit = (self.current_digit - 1) % 10
            elif action == 1 or action == 3:
                self.current_digit = (self.current_digit + 1) % 10
            else:
                raise ValueError("Action must be encoded as an int between 0 and 2")
        else:
            raise ValueError("Invalid actions_type; was {}".format(self.actions_type))
        self.current_digit_image = self._get_image_from_digit(self.current_digit)
        done = self.current_digit == self.target_digit
        reward = 1 if done else 0
        return (self.current_digit_image, reward, done, {'state': old_digit,
                                                         'next_state': self.current_digit,
                                                         'goal_state': self.goal_image
                                                        })

    def _reset(self):
        self.target_digit = random.choice(self.target_digits)
        self.goal_image = self._get_image_from_digit(self.target_digit)
        self.current_digit = np.random.choice(list(range(self.target_digit)) +\
                                              list(range(self.target_digit + 1, 10)))
        self.current_digit_image = self._get_image_from_digit(self.current_digit)
        return self.current_digit_image, self.goal_image

    def _render(self, mode='human', close=False):
        if mode != 'human': return
        if close:
            cv2.destroyWindow('game')
            return
        cv2.namedWindow('game', cv2.WINDOW_NORMAL)
        img = cv2.resize(self.current_digit_image, None, fx=5, fy=5)
        cv2.imshow('game', img)
        cv2.waitKey(300)

    def _get_image_from_digit(self, digit):
        # returns an MNIST digit corresponding to the inputted digit
        filename = random.choice(self.filename_library[digit])
        im = imread(filename)
        im = np.expand_dims(im, -1) # add a single color channel)
        return im

    def _get_random_obs(self, digit):
        digit = random.randrange(10)
        return self._get_image_from_digit(digit)

    def get_action_meanings(self):
        if self.actions_type == "linear":
            return ACTION_MEANING_LINEAR
        elif self.actions_type == "complex":
            return ACTION_MEANING_COMPLEX
        else:
            raise ValueError

class BarebonesMNISTEnv(MNISTEnv):
    observation_space = spaces.Box(low=0, high=1, shape=(10))
    def _get_image_from_digit(self, digit):
        arr = np.zeros(10)
        arr[digit] = 1
        return arr
    def _render(self):
        print (self.current_digit)



ACTION_MEANING_COMPLEX = {
    0 : "NOP",
    1 : "+1",
    2 : "x2",
    3 : "x3"
}

ACTION_MEANING_LINEAR = {
    0: "-1",
    1: "+1",
    2: "-1",
    3: "+1"
}
