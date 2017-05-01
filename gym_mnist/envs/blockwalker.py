import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageColor
#------------------------------------------------------
# A hacky solution for @yo-shavit to run OpenCV in conda without bugs
# Others should remove
import os
if os.environ['HOME'] == '/Users/yonadav':
    import sys;
    sys.path.append("/Users/yonadav/anaconda/lib/python3.5/site-packages")
#------------------------------------------------------
import cv2

xmax = 8; ymax = 8
PIXELS_PER_INDEX = 4
noise=True

class BlockWalkerEnv(gym.Env):

    metadata = {'render.modes': ['human', 'training']}
    action_space = spaces.Discrete(4)
    observation_space = spaces.Box(low=0, high=255,
                                   shape=(xmax*PIXELS_PER_INDEX,
                                          ymax*PIXELS_PER_INDEX,
                                          3))
    def __init__(self):
        self.figure = None

    def _step(self, action):
        pos_before = np.copy(self.board.playerpos)
        self.board.step(ACTION_MEANING[action])
        self.obs = self.board.visualize()
        done = np.array_equal(self.board.playerpos, self.board.goalpos)
        pos_after = np.copy(self.board.playerpos)
        reward = 0
        if ACTION_MEANING[action] == "CSWAP":
            reward = -.3
        if done: reward = 1
        info = {'state': pos_before, 'next_state': pos_after}
        return self.obs, reward, done, info

    def _reset(self):
        self.board = BlockWalkerBoard()
        self.obs = self.board.visualize()
        return self.obs

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
        plt.imshow(self.obs)
        plt.pause(.01)

    def get_action_meanings(self):
        return ACTION_MEANING

class MulticoloredBlockWalkerEnv(BlockWalkerEnv):
    action_space = spaces.Discrete(5)
    def _reset(self):
        self.board = MulticoloredBlockWalkerBoard()
        self.obs = self.board.visualize()
        return self.obs

class MulticoloredRandomBlockWalkerEnv(MulticoloredBlockWalkerEnv):
    def _reset(self):
        self.board = MulticoloredBlockWalkerBoard(boardarr=random_board())
        self.obs = self.board.visualize()
        return self.obs

ACTION_MEANING = {
    0 : "UP",
    1 : "LEFT",
    2 : "DOWN",
    3 : "RIGHT",
    4 : "CSWAP",
}


RED = ImageColor.getrgb("#ff0000")
BLUE = ImageColor.getrgb("#00ff00")
GREEN = ImageColor.getrgb("#0000ff")
MAGENTA = ImageColor.getrgb("#ff00ff")
YELLOW = ImageColor.getrgb("#ffff00")
WHITE = ImageColor.getrgb("#ffffff")
BLACK = ImageColor.getrgb("#000000")


class BlockWalkerBoard:
    def __init__(self, boardarr=None):
        if boardarr is not None:
            self.boardarr = boardarr
        else:
            self.boardarr = BlockWalkerBoard.default_board()
        self.goalpos = np.array([random.randrange(0,xmax),
                                 random.randrange(0,ymax)])
        self.playerpos = None
        while self.playerpos is None or np.array_equal(self.playerpos,
                                                       self.goalpos):
            self.playerpos = np.array([random.randrange(0,xmax),
                                       random.randrange(0,ymax)])

    def step(self, action):
        if action == "UP":
            if self.playerpos[1] - 1 >= 0:
                self.playerpos[1] -=1
        elif action == "LEFT":
            if self.playerpos[0] - 1 >= 0:
                self.playerpos[0] -=1
        elif action == "DOWN":
            if self.playerpos[1] + 1 < ymax:
                self.playerpos[1] += 1
        elif action == "RIGHT":
            if self.playerpos[0] + 1 < xmax:
                self.playerpos[0] += 1
        else:
            raise RuntimeError("BlockWalker action must be 'UP', 'DOWN', 'LEFT', or 'RIGHT'")

    def default_board():
        boardarr = -1*np.ones((xmax, ymax))
        return boardarr

    def visualize(self):
        im = Image.new("RGB", (xmax*PIXELS_PER_INDEX, ymax*PIXELS_PER_INDEX),
                       color=WHITE)
        draw = ImageDraw.Draw(im)
        draw.rectangle(idx_to_rectanglecoords(self.goalpos), fill=MAGENTA)
        draw.rectangle(idx_to_rectanglecoords(self.playerpos),
                       fill=RED, outline=BLACK)
        im = np.array(im)
        if noise == True:
            im += cv2.randn(np.empty_like(im), (0, 0, 0), 0.25*
                           np.array([[1, 0.5, 0.5],
                                     [0.5, 1, 0.5],
                                     [0.5, 0.5, 1]]))
        return im

COLORS = [RED, GREEN, BLUE]

class MulticoloredBlockWalkerBoard(BlockWalkerBoard):
    """
    Board integer meanings:
        -1 is empty
        0 is red
        1 is green
        2 is blue
    """
    def __init__(self, boardarr=None):
        if boardarr is None:
            boardarr = MulticoloredBlockWalkerBoard.default_board()
        BlockWalkerBoard.__init__(self, boardarr)
        self.player_cid = random.randrange(len(COLORS))
        self.boardarr[self.goalpos[0], self.goalpos[1]] = -1

    def step(self, action):
        if action == "CSWAP":
            self.player_cid = (self.player_cid + 1)%len(COLORS)
        else:
            original = np.copy(self.playerpos)
            BlockWalkerBoard.step(self, action)
            if self._at_arridx(self.playerpos) != self.player_cid and\
               self._at_arridx(self.playerpos) != -1:
                self.playerpos = original

    def _at_arridx(self, arridx):
        return self.boardarr[arridx[0], arridx[1]]

    def default_board():
        arr = np.array(
            [[ 1, 1, 1, 1,-1,-1,-1,-1],
             [ 1,-1,-1, 1,-1,-1,-1,-1],
             [ 0,-1,-1, 1,-1,-1,-1,-1],
             [ 0,-1,-1, 1,-1,-1, 2,-1],
             [ 0,-1,-1, 1,-1, 2, 2, 2],
             [ 0, 0, 0, 0,-1, 2,-1, 2],
             [-1,-1,-1,-1,-1, 2,-1, 2],
             [ 2, 2, 2, 2, 2, 2, 2, 2],]).transpose()
        return arr

    def visualize(self):
        im = Image.new("RGB", (xmax*PIXELS_PER_INDEX, ymax*PIXELS_PER_INDEX),
                       color=WHITE)
        draw = ImageDraw.Draw(im)
        for i in range(xmax):
            for j in range(ymax):
                if self.boardarr[i,j] != -1:
                    draw.rectangle(
                        idx_to_rectanglecoords(np.array([i,j])),
                        fill=COLORS[self.boardarr[i,j]])
        draw.rectangle(idx_to_rectanglecoords(self.goalpos), fill=MAGENTA)
        draw.rectangle(idx_to_rectanglecoords(self.playerpos),
                       fill=COLORS[self.player_cid], outline=BLACK)
        im = np.array(im)
        if noise == True:
            im += cv2.randn(np.empty_like(im), (0, 0, 0), 0.25*
                           np.array([[1, 0.5, 0.5],
                                     [0.5, 1, 0.5],
                                     [0.5, 0.5, 1]]))
        # im = Image.fromarray(im, mode="RGB").convert(mode="L")
        return np.array(im)

def random_board():
    board = np.reshape(
        np.random.choice([-1,0,1,2], size=(xmax, ymax), p=[.4, .2, .2, .2]),
        (xmax, ymax))
    return board



def idx_to_rectanglecoords(idx):
    pidx = idx*PIXELS_PER_INDEX
    rc = np.concatenate([pidx, pidx + np.array([PIXELS_PER_INDEX,
                                               PIXELS_PER_INDEX])]).tolist()
    return rc

