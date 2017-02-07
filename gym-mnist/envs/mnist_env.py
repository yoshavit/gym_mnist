import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import os
from scipy.ndimage import imread
import matplotlib.pyplot as plt

IMG_SIDEWIDTH = 28
CURRENT_DIR_TO_MNIST_DIR = "../resources/"
PARTIAL_DATASET_SIZE = 1000

class MNISTEnv(gym.Env):
    '''
    This environment encodes a game with a reward for reaching
    a particular number (default 0) using only elementary operations
    (+1, -1, *2) always mod 10. The reward for each action is -0.1,
    until we reach the goal, at which point it is 1.0

    Action definitions:
        '0' : increment digit by 1
        '1' : decrement digit by 1
        '2' : multiply digit by 2
'''
    metadata = {'render.modes': ['human']}
    action_space = spaces.Discrete(3)
    def __init__(self, target_digit=0, full_mnist=False):
        # first, load all the MNIST images into RAM
        self.mnist_library = {k:np.empty((0, IMG_SIDEWIDTH, IMG_SIDEWIDTH)) for k
                              in range(10)}
        CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
        MNIST_DIR = CURRENT_DIR + CURRENT_DIR_TO_MNIST_DIR
        for i in range(10):
            DIGIT_DIR = MNIST_DIR + "digit%d/" % i
            digit_filenames = os.listdir(DIGIT_DIR)
            if not full_mnist:
                digit_filenames = np.random.choice(digit_filenames,
                                                   PARTIAL_DATASET_SIZE/10)
            self.mnist_library[i] = np.empty(
                (len(digit_filenames),IMG_SIDEWIDTH,IMG_SIDEWIDTH))
            for j in range(len(digit_filenames)):
                filename = digit_filenames[j]
                im = imread(filename)
                self.mnist_library[i][j,:,:] = im

        # Now, decide on the target digit
        self.target_digit = target_digit

        # Lastly, a little bookkeeping
        self.figure = None

    def _step(self, action):
        if action == 0:
            self.current_digit = (self.current_digit + 1) % 10
        elif action == 1:
            self.current_digit = (self.current_digit - 1) % 10
        elif action == 2:
            self.current_digit = (self.current_digit * 2) % 10
        else:
            raise ValueError("Action must be encoded as an int between 0 and 2")
        self.current_digit_image = _get_image_from_digit(self.current_digit)
        done = self.current_digit == self.target_digit
        if done:
            reward = 1.0
        else:
            reward = -0.1
        return (self.current_digit_image, reward, done, None)

    def _reset(self):
        self.current_digit = np.random.randint(10)
        self.current_digit_image = _get_image_from_digit(self.current_digit)

    def _render(self, mode='human', close=False):
        if self.figure == None:
            self.figure = plt.figure()
        plt.figure(self.figure)
        plt.imshow(self.current_digit_image)
        sleep(1)

    def _get_image_from_digit(digit):
        # returns an MNIST digit corresponding to the inputted digit
        idx = np.random.choice(self.mnist_dict[digit].shape[0])
        return self.mnist_dict[digit][idx,:,:]

ACTION_MEANING = {
    0 : "+1",
    1 : "-1",
    2 : "x2"
}

