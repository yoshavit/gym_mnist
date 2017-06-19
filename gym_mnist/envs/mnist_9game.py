import gym
from gym import spaces
import numpy as np
import os
import random
import cv2

SUBPANE_SIDEWIDTH = 14
CURRENT_DIR_TO_MNIST_DIR = "/../resources/"
d = 3 # panes per side

class MNIST9GameEnv(gym.Env):
    '''
'''
    metadata = {'render.modes': ['human', 'training']}
    action_space = spaces.Discrete(4)
    observation_space = spaces.Box(low=0, high=255,
                                   shape=(d*SUBPANE_SIDEWIDTH,
                                          d*SUBPANE_SIDEWIDTH,
                                          1))
    def __init__(self, target_ordering=None):
        # first, load all the MNIST image filenames
        self.filename_library = [[] for k
                              in range(10)]
        CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
        MNIST_DIR = CURRENT_DIR + CURRENT_DIR_TO_MNIST_DIR
        for i in range(10):
            DIGIT_DIR = MNIST_DIR + "digit%d/" % i
            digit_filenames = [DIGIT_DIR + name for name in
                               os.listdir(DIGIT_DIR)]
            self.filename_library[i] = digit_filenames
        # Now, decide on the target arrangement of digits
        self.target_ordering = target_ordering
        # and initialize the game engine
        self.tg = TileGame(d, target_ordering=target_ordering)

    def _step(self, action):
        old_order = np.copy(self.tg.state)
        reward = done = self.tg._step(action)
        self.current_image = self._get_image_from_order(self.tg.state)
        return (self.current_image, reward, done, {'state': old_order,
                                                         'next_state':
                                                         self.tg.state})

    def _reset(self):
        self.tg._reset()
        self.current_image = self._get_image_from_order(self.tg.state)
        return self.current_image

    def _render(self, mode='human', close=False):
        if mode != 'human': return
        print(self.tg.state)
        if close:
            cv2.destroyWindow('game')
            return
        cv2.namedWindow('game', cv2.WINDOW_NORMAL)
        img = cv2.resize(self.current_image, None, fx=5, fy=5)
        cv2.imshow('game', img)
        cv2.waitKey(300)

    def _get_image_from_order(self, order):
        image_slices = []
        for x in range(d):
            panes = [self._get_image_from_digit(order[x,y]) for y in
                                                 range(d)]
            image_slices.append(np.concatenate(panes, axis=0))
        output_image = np.concatenate(image_slices, axis=1)
        return output_image

    def _get_image_from_digit(self, digit):
        # returns an MNIST digit corresponding to the inputted digit
        filename = random.choice(self.filename_library[digit])
        im = cv2.imread(filename)
        im = cv2.resize(im, (SUBPANE_SIDEWIDTH, SUBPANE_SIDEWIDTH))
        return im

    def get_action_meanings(self):
        return ACTION_MEANING

ACTION_MEANING = {
    0 : "UP",
    1 : "LEFT",
    2 : "DOWN",
    3 : "RIGHT"
}

class TileGame:
    def __init__(self, sidewidth, initial_ordering=None, target_ordering=None):
        self.d = sidewidth
        self.initial_ordering = initial_ordering
        self._reset()
        if target_ordering is None:
            target_ordering = np.reshape(np.arange(self.d**2), [self.d,
                                                                self.d])
        self.target_ordering = target_ordering
        assert (np.sort(self.state.flatten()) ==\
                np.sort(target_ordering.flatten())).all()
        self.goal_state = target_ordering

    def _reset(self):
        if self.initial_ordering is None:
            # if initial_ordering isn't preset, keep generating random
            # initializations
            initial_ordering = np.reshape(np.random.permutation(self.d**2), [self.d,
                                                                    self.d])
        else:
            initial_ordering = self.initial_ordering
        assert initial_ordering.shape == (self.d, self.d)
        assert len(list(zip(*np.where(initial_ordering==0)))) == 1
        self.state = initial_ordering

    def _step(self, action):
        # 0 up, 1 left, 2 down, 3 right
        movable_tile = list(zip(*np.where(self.state == 0)))[0]
        x, y = movable_tile
        if action == 0:
            if y>0:
                tmp = self.state[x, y-1]
                self.state[x, y-1] = self.state[x,y]
                self.state[x, y] = tmp
        elif action == 1:
            if x>0:
                tmp = self.state[x-1, y]
                self.state[x-1, y] = self.state[x,y]
                self.state[x,y] = tmp
        elif action == 2:
            if y + 1 < self.d:
                tmp = self.state[x, y+1]
                self.state[x, y+1] = self.state[x, y]
                self.state[x, y] = tmp
        elif action == 3:
            if x + 1 < self.d:
                tmp = self.state[x+1, y]
                self.state[x+1, y] = self.state[x, y]
                self.state[x, y] = tmp
        else:
            raise RuntimeError("{} is not a valid action; must be 0 (up), 1 (left), 2 (down), or 3 (right)".format(action))
        goal = (self.state == self.goal_state).all()
        return goal
