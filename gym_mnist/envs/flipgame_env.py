import gym
from gym import spaces
import numpy as np
import os
import random
import cv2

SUBPANE_SIDEWIDTH = 28
CURRENT_DIR_TO_MNIST_DIR = "/../resources/"
d = 3 # panes per side
maxval = 2 # num values each tile can take (0 to vals-1)

class FlipgameEnv(gym.Env):
    metadata = {'render.modes': ['human', 'training']}
    action_space = spaces.Discrete(2*d+1)
    observation_space = spaces.Box(low=0, high=255,
                                   shape=(d*SUBPANE_SIDEWIDTH,
                                          d*SUBPANE_SIDEWIDTH,
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
        self.fg = FlipGame(d, maxval, always_feasible=always_feasible,
                           init_fn=init_fn,
                           goal_fn=goal_fn)

    def _step(self, action):
        old_order = np.copy(self.fg.state)
        reward = done = self.fg._step(action)
        self.current_image = self._get_image_from_order(self.fg.state)
        return (self.current_image, reward, done, {'state': old_order,
                                                   'next_state': self.fg.state,
                                                   'goal_state': self.goal_image
                                                })

    def _reset(self):
        self.fg._reset()
        self.current_image = self._get_image_from_order(self.fg.state)
        self.goal_image = self._get_image_from_order(self.fg.target)
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
        for y in range(d):
            panes = [self._get_image_from_digit(order[x,y]) for x in
                                                 range(d)]
            image_slices.append(np.concatenate(panes, axis=0))
        output_image = np.concatenate(image_slices, axis=1)
        return output_image

    def _get_random_obs(self):
        order = np.random.randint(maxval, size=[d, d])
        return self._get_image_from_order(order)

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
for i in range(d):
    ACTION_MEANING[i] = "ROW%d"%d
    ACTION_MEANING[d+i] = "COL%d"%d
ACTION_MEANING[2*d] = "DIAG"

class FlipGame:
    def __init__(self, sidewidth, maxval=2, always_feasible=True,
                 init_fn=None, goal_fn=None, target=None):
        # x axis (major) is vertical, y axis (minor) is vertical
        self.d = sidewidth
        self.maxval = maxval
        if init_fn is not None:
            self.init_fn = init_fn
        elif always_feasible:
            self.init_fn = self._init_from_goal
        else:
            self.init_fn = lambda: np.random.randint(self.maxval,
                                                     size=[self.d, self.d])
        assert not (bool(goal_fn) and bool(target)), "Must specify at most one of goal_fn or target"
        if goal_fn is None:
            self.target = target if target else np.zeros([self.d, self.d],
                                                         dtype='int64')
            # assumes init_fn permutes the same set of inputs
            self.goal_fn = lambda x: (x == self.target).all()
        else:
            self.target = target
            self.goal_fn = goal_fn

    def _reset(self):
        self.state = self.init_fn()

    def _init_from_goal(self):
        self.state = self.target.copy()
        for _ in range(2*d+2):
            self._step(random.randrange(2*d+1))
        return self.state

    def _step(self, action):
        # 0 through d-1 are row-flips, d through 2d-1 are column flips, 2d is
        # diag
        assert action <= 2*d
        diag = action == 2*d
        flip_column = (action >= d) and not diag # true if column, false if row
        idx = action % d # lowest-bits denote which column to be flipped
        if diag:
            for i in range(d):
                self.state[i,(d-1)-i] = (self.state[i,(d-1) - i] + 1) % self.maxval
        elif flip_column:
            for row in range(d):
                self.state[row, idx] = (self.state[row, idx] + 1) % self.maxval
        else:
            for col in range(d):
                self.state[idx, col] = (self.state[idx, col] + 1) % self.maxval
        goal = self.goal_fn(self.state)
        return goal
