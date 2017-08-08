import gym
from gym import spaces
import numpy as np
import os
import random
import cv2

SUBPANE_SIDEWIDTH = 28
CURRENT_DIR_TO_MNIST_DIR = "/../resources/"
dx = 2 # panes per side
dy = 3

class RotgameEnv(gym.Env):
    metadata = {'render.modes': ['human', 'training']}
    action_space = spaces.Discrete(dx + dy)
    observation_space = spaces.Box(low=0, high=255,
                                   shape=(dx*SUBPANE_SIDEWIDTH,
                                          dy*SUBPANE_SIDEWIDTH,
                                          1))
    def __init__(self, always_feasible=True, init_fn=None, goal_fn=None):
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
        # initialize the game engine
        self.rg = RotGame()

    def _step(self, action):
        old_order = np.copy(self.rg.state)
        reward = done = self.rg._step(action)
        self.current_image = self._get_image_from_order(self.rg.state)
        return (self.current_image, reward, done, {'state': old_order,
                                                   'next_state': self.rg.state,
                                                   'goal_state': self.goal_image
                                                })

    def _reset(self):
        self.rg._reset()
        self.current_image = self._get_image_from_order(self.rg.state)
        self.goal_image = self._get_image_from_order(self.rg.target)
        return self.current_image, self.goal_image

    def _render(self, mode='human', close=False):
        if mode != 'human': return
        if close:
            cv2.destroyWindow('game')
            return
        cv2.namedWindow('game', cv2.WINDOW_NORMAL)
        img = cv2.resize(self.current_image, None, fx=5, fy=5)
        cv2.imshow('game', img)
        cv2.waitKey(300)

    def _get_image_from_order(self, order):
        image_slices = []
        for y in range(dy):
            panes = [self._get_image_from_digit(order[x,y]) for x in
                                                 range(dx)]
            image_slices.append(np.concatenate(panes, axis=0))
        output_image = np.concatenate(image_slices, axis=1)
        return output_image

    def _get_random_obs(self):
        order = np.random.randint(dy, size=[dx, dy])
        return self._get_image_from_order(order), order

    def _get_image_from_digit(self, digit):
        # returns an MNIST digit corresponding to the inputted digit
        filename = random.choice(self.filename_library[digit])
        im = cv2.imread(filename, 0) # load as B/W
        im = cv2.resize(im, (SUBPANE_SIDEWIDTH, SUBPANE_SIDEWIDTH))
        im = np.expand_dims(im, -1) #add channel
        return im

    def get_action_meanings(self):
        return ACTION_MEANING

ACTION_MEANING = {}
for i in range(dx):
    ACTION_MEANING[i] = "ROW%d"%i
for i in range(dy):
    ACTION_MEANING[dx+i] = "COL%d"%i

class RotGame:
    def __init__(self):
        # x axis (major) is vertical, y axis (minor) is vertical
        target = np.asarray(list(range(dy))*dx)
        self.target = target.reshape([dx, dy])
        self.init_fn = self._init_from_goal
        # self.init_fn = lambda: np.random.permutation(target).reshape([dx, dy])
        self.goal_fn = lambda x: (x == self.target).all()

    def _reset(self):
        self.state = self.init_fn()

    def _init_from_goal(self):
        self.state = self.target.copy()
        for _ in range(dx + dy):
            self._step(random.randrange(dx + dy), rollright=False) # reverse of
            # normal actions to ensure state is reachable
        return self.state

    def _step(self, action, rollright=True):
        # 0 through d-1 are row-flips, d through 2d-1 are column flips, 2d is
        # diag
        assert action < dx + dy
        polarity = rollright*2 - 1
        flip_column = (action >= dx) # true if column, false if row
        if flip_column:
            idx = action - dx
            self.state[:, idx] = np.roll(self.state[:, idx], polarity)
        else:
            idx = action
            self.state[idx, :] = np.roll(self.state[idx, :], polarity)
        goal = self.goal_fn(self.state)
        return goal
