import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import os
import random
from scipy.ndimage import imread
import matplotlib.pyplot as plt

IMG_SIDEWIDTH = 28
CURRENT_DIR_TO_MNIST_DIR = "/../resources/"
PARTIAL_DATASET_SIZE = 10000

class MNISTEnv(gym.Env):
    '''
    This environment encodes a game with a reward for reaching
    a particular number (default 0) using only elementary operations
    (+1, -1, *2) always mod 10. The reward for each action is -0.1,
    until we reach the goal, at which point it is 1.0

    Action definitions:
        '0' : do nothing
        '1' : increment digit by 1
        '2' : decrement digit by 1
        '3' : multiply digit by 2
'''
    metadata = {'render.modes': ['human', 'training']}
    action_space = spaces.Discrete(3)
    observation_space = spaces.Box(low=0, high=255, shape=(IMG_SIDEWIDTH,IMG_SIDEWIDTH))
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
                                                   int(PARTIAL_DATASET_SIZE/10))
            self.mnist_library[i] = np.empty(
                (len(digit_filenames),IMG_SIDEWIDTH,IMG_SIDEWIDTH))
            for j in range(len(digit_filenames)):
                filename = DIGIT_DIR + digit_filenames[j]
                im = imread(filename)
                self.mnist_library[i][j,:,:] = im
        # Now, decide on the target digit
        self.target_digit = target_digit
        # Lastly, a little bookkeeping
        self.figure = None

    def _step(self, action):
        if action == 0:
            pass
        elif action == 1:
            self.current_digit = (self.current_digit + 1) % 10
        elif action == 2:
            self.current_digit = (self.current_digit - 1) % 10
        elif action == 3:
            self.current_digit = (self.current_digit * 2) % 10
        else:
            raise ValueError("Action must be encoded as an int between 0 and 2")
        self.current_digit_image = self._get_image_from_digit(self.current_digit)
        done = self.current_digit == self.target_digit
        if done:
            reward = 1.0
        else:
            reward = -0.1
        return (self.current_digit_image, reward, done, None)

    def _reset(self):
        self.current_digit = np.random.randint(10)
        self.current_digit_image = self._get_image_from_digit(self.current_digit)

    def _render(self, mode='human', close=False):
        if mode != 'human': return
        if close:
            if self.figure:
                plt.figure(self.figure.number)
                plt.close()
            return
        if not self.figure:
            self.figure = plt.figure()
        plt.figure(self.figure.number)
        plt.imshow(self.current_digit_image, cmap='Greys')
        plt.pause(0.01)

    def _get_image_from_digit(self, digit):
        # returns an MNIST digit corresponding to the inputted digit
        idx = np.random.choice(self.mnist_library[digit].shape[0])
        return self.mnist_library[digit][idx,:,:]

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

ACTION_MEANING = {
    0 : "NOP",
    1 : "+1",
    2 : "-1",
    3 : "x2"
}

