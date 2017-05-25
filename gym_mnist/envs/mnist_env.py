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
    (+1, -1, *2) always mod 10. The reward for each action is -0.1,
    until we reach the goal, at which point it is 1.0

    Action definitions:
        '0' : do nothing
        '1' : increment digit by 1
        '2' : multiply digit by 2
        '3' : multiply digit by 3
'''
    metadata = {'render.modes': ['human', 'training']}
    action_space = spaces.Discrete(4)
    observation_space = spaces.Box(low=0, high=255,
                                   shape=(IMG_SIDEWIDTH,IMG_SIDEWIDTH, 1))
    def __init__(self, target_digit=0, full_mnist=False):
        # first, load all the MNIST images into RAM
        self.mnist_library = [np.empty((0, IMG_SIDEWIDTH, IMG_SIDEWIDTH)) for k
                              in range(10)]
        CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
        MNIST_DIR = CURRENT_DIR + CURRENT_DIR_TO_MNIST_DIR
        for i in range(10):
            DIGIT_DIR = MNIST_DIR + "digit%d/" % i
            digit_filenames = os.listdir(DIGIT_DIR)
            if not full_mnist:
                digit_filenames = np.random.choice(digit_filenames,
                                                   int(PARTIAL_DATASET_SIZE))
            self.mnist_library[i] = np.empty(
                (len(digit_filenames),IMG_SIDEWIDTH,IMG_SIDEWIDTH))
            for j in range(len(digit_filenames)):
                filename = DIGIT_DIR + digit_filenames[j]
                im = imread(filename)
                self.mnist_library[i][j,:,:] = im
        # Now, decide on the target digit
        self.target_digit = target_digit


    def _step(self, action):
        old_digit = self.current_digit
        if action == 0:
            pass
        elif action == 1:
            self.current_digit = (self.current_digit + 1) % 10
        elif action == 2:
            self.current_digit = (self.current_digit * 2) % 10
        elif action == 3:
            self.current_digit = (self.current_digit * 3) % 10
        else:
            raise ValueError("Action must be encoded as an int between 0 and 2")
        self.current_digit_image = self._get_image_from_digit(self.current_digit)
        done = self.current_digit == self.target_digit
        if done:
            reward = 1.0
        else:
            reward = -0.2
        return (self.current_digit_image, reward, done, {'state': old_digit,
                                                         'next_state': self.current_digit})

    def _reset(self):
        self.current_digit = np.random.randint(9) + 1
        self.current_digit_image = self._get_image_from_digit(self.current_digit)
        return self.current_digit_image

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
        idx = np.random.choice(self.mnist_library[digit].shape[0])
        return np.expand_dims(self.mnist_library[digit][idx,:,:], -1)

    def _get_full_mnist_dataset(self):
        images = []
        labels = []
        for digit in range(len(self.mnist_library)):
            for image in self.mnist_library[digit]:
                images.append(image)
                labels.append(digit)
        pairs = list(zip(images, labels))
        random.shuffle(pairs)
        images, labels = zip(*pairs)
        return images, labels

    def get_action_meanings(self):
        return ACTION_MEANING

ACTION_MEANING = {
    0 : "NOP",
    1 : "+1",
    2 : "x2",
    3 : "x3"
}

