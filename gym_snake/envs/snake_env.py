import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def addFood():
        empty_spaces = np.argwhere(field==0)
        if np.size(empty_spaces) > 0:
            food_pos = np.random.choice(empty_spaces)
            field[food_pos] = -1
        else:
            pass

    def head_pos():
        return argmax(field)

    def __init__(self):
        field = np.zeros((11,20))
        field[10, 0:9] = np.array(range(1,10))
        addFood()
    def _step(self, action):
        field -= (field>0)
        field[new_head_pos] = np.amax(field) + 1
    def _reset(self):
    ...
    def _render(self, mode='human', close=False):
    ...
    