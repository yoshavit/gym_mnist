import gym
from gym import spaces
import numpy as np
import os
import random
import cv2

SUBPANE_SIDEWIDTH = 28
CURRENT_DIR_TO_MNIST_DIR = "/../resources/"
d = 3 # panes per side

class MNIST9GameEnv(gym.Env):
    metadata = {'render.modes': ['human', 'training']}
    action_space = spaces.Discrete(4)
    observation_space = spaces.Box(low=0, high=255,
                                   shape=(d*SUBPANE_SIDEWIDTH,
                                          d*SUBPANE_SIDEWIDTH,
                                          1))
    def __init__(self, init_fn=None, goal_fn=None):
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
        self.tg = TileGame(d, init_fn=init_fn,
                           goal_fn=goal_fn)

    def _step(self, action):
        old_order = np.copy(self.tg.state)
        reward = done = self.tg._step(action)
        self.current_image = self._get_image_from_order(self.tg.state)
        return (self.current_image, reward, done, {'state': old_order,
                                                   'next_state': self.tg.state,
                                                   'goal_state': self.goal_image
                                                })

    def _reset(self):
        self.tg._reset()
        self.current_image = self._get_image_from_order(self.tg.state)
        self.goal_image = self._get_image_from_order(self.tg.target)
        return self.current_image

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
        for y in range(d):
            panes = [self._get_image_from_digit(order[x,y]) for x in
                                                 range(d)]
            image_slices.append(np.concatenate(panes, axis=0))
        output_image = np.concatenate(image_slices, axis=1)
        return output_image

    def _get_image_from_digit(self, digit):
        # returns an MNIST digit corresponding to the inputted digit
        filename = random.choice(self.filename_library[digit])
        im = cv2.imread(filename, 0) # load as B/W
        im = cv2.resize(im, (SUBPANE_SIDEWIDTH, SUBPANE_SIDEWIDTH))
        im = np.expand_dims(im, -1) #add channel
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
    def __init__(self, sidewidth, init_fn=None, goal_fn=None, target=None):
        # x axis (major) is vertical, y axis (minor) is vertical
        self.d = sidewidth
        if init_fn is None:
            init_fn = lambda: np.reshape(np.random.permutation(self.d**2),
                                           [self.d, self.d])
        self.init_fn = init_fn
        self._reset()
        assert not (bool(goal_fn) and bool(target)), "Must specify at most one of goal_fn or target"
        if goal_fn is None:
            self.target = target if target else np.sort(self.state.flatten()).reshape([self.d, self.d])
            # assumes init_fn permutes the same set of inputs
            self.goal_fn = lambda x: (x.flatten() == self.target.flatten()).all()
        else:
            self.target = None
            self.goal_fn = goal_fn

    def _reset(self):
        self.state = self.init_fn()

    def _step(self, action):
        # 0 up, 1 left, 2 down, 3 right
        movable_tile = list(zip(*np.where(self.state == 0)))[0]
        x, y = movable_tile
        if action == 0:
            if x>0:
                tmp = self.state[x-1, y]
                self.state[x-1, y] = self.state[x,y]
                self.state[x, y] = tmp
        elif action == 1:
            if y>0:
                tmp = self.state[x, y-1]
                self.state[x, y-1] = self.state[x,y]
                self.state[x,y] = tmp
        elif action == 2:
            if x + 1 < self.d:
                tmp = self.state[x+1, y]
                self.state[x+1, y] = self.state[x, y]
                self.state[x, y] = tmp
        elif action == 3:
            if y + 1 < self.d:
                tmp = self.state[x, y+1]
                self.state[x, y+1] = self.state[x, y]
                self.state[x, y] = tmp
        else:
            raise RuntimeError("{} is not a valid action; must be 0 (up), 1 (left), 2 (down), or 3 (right)".format(action))
        goal = self.goal_fn(self.state)
        return goal
