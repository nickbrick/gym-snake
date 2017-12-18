import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time
import itertools

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}
    step_directions = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
    def __init__(self):
        self.W = 20
        self.H = 11
        self.every_space_flat = np.array(range(self.W * self.H))
        self.action_space = spaces.Discrete(4)
        lows = -np.ones(self.W * self.H)
        highs = np.ones(self.W * self.H)
        self.observation_space = spaces.Box(lows, highs)
        self.viewer = None
        self.head_new = None
        self.food_new = None
        self.body_new = None
        self.empty_spaces_new = None
        self.state = None
        self.field = np.zeros((self.H, self.W))
        self.field[10, 0:9] = np.array(range(1, 10))


    # def head_pos(self):
    #     return np.argwhere(self.field==np.amax(self.field))[0]
    #     # return np.argmax(self.field)
    
    def addFood(self):
        snake = np.append(self.head_new, self.body_new, 0)
        snake_ravel = np.array([[np.ravel_multi_index(x, (self.H, self.W))] for x in snake])
        empty_spaces = np.setdiff1d(self.every_space_flat, snake_ravel, True)

        if np.size(empty_spaces) > 0:
            space_ravel = np.random.choice(empty_spaces)
            space = np.unravel_index(space_ravel, (self.H, self.W))
            self.food_new = np.array([space])
        else:
            pass

        
    def _step(self, action):
        # 0 1 2 3 = Down Left Up Right
        direction = SnakeEnv.step_directions[action]
        added = np.add(self.head_new, direction)
        # print(self.head_new[0])
        self.body_new = np.insert(self.body_new, 0, self.head_new, 0)  # add old head to body
        self.head_new = np.array(tuple(added))
        body_ravel = np.array([np.ravel_multi_index(x, (self.H, self.W)) for x in self.body_new])
        head_ravel = np.array([np.ravel_multi_index(self.head_new[0], (self.H, self.W))])
        print(body_ravel)
        print(head_ravel)
        if -1 in self.head_new[0] or self.head_new[0][0] == self.H or self.head_new[0][1] == self.W:  # hit a wall
            reward = -1
            done = True
        elif np.isin(head_ravel[0], body_ravel, True):  # hit body
            print('hit body')
            if np.array_equal(self.head_new[0], self.body_new[0]):  # try to hit neck, illegal move
                reward = 0
                done = False
            else:  # hit rest of body, lose
                reward = -1
                done = True
        else:  # move
            if np.array_equal(self.head_new, self.food_new):  # eat a food
                self.addFood()
                reward = 20
            else:
                np.delete(self.body_new, np.size(self.body_new,0) - 1, 0)  # remove tail from body
                reward = -1
            done = False
        return self.field, reward, done, {}
    # def _step(self, action):
        # # 0 1 2 3 = Down Left Up Right
        # head = self.head_pos()
        # direction = SnakeEnv.step_directions[action]
        # added = np.add(head, direction)
        # new_head_pos = tuple(added)
        # if -1 in new_head_pos or new_head_pos[0] == self.H or new_head_pos[1] == self.W:
        #     reward = -1
        #     done = True
        # elif self.field[new_head_pos] > 0:
        #     if self.field[new_head_pos] == self.field[tuple(head)] - 1:
        #         reward = 0
        #         done = False
        #     else:
        #         reward = -1
        #         done = True
        # else:
        #     self.field[new_head_pos] = np.amax(self.field) + 1
            # if new_head_pos == self.food_pos:
            #     self.addFood()
            #     reward = 20
        #     else:
        #         self.field -= (self.field>0)
        #         reward = -1
        #     done = False
        # return self.field, reward, done, {}

    def _reset(self):
        self.head_new = np.array([[10,8]])
        self.body_new =np.array([[10,x] for x in range(7,-1,-1)])
        self.field = np.zeros((self.H, self.W))
        self.field[10, 0:9] = np.array(range(1,10))
        self.addFood()

    def _render(self, mode='human', close=False):
        print('head: {}'.format(self.head_new))
        print('body: {}'.format(self.body_new))
        print('food: {}'.format(self.food_new))
        # if close:
        #     if self.viewer is not None:
        #         self.viewer.close()
        #         self.viewer = None
        #     return
        
        # CELL_SIZE = 30
        # PAD = 2
        # screen_width = CELL_SIZE * self.W + 2
        # screen_height = CELL_SIZE * self.H + 2

        # from gym.envs.classic_control import rendering
        # if self.viewer is None:
        #     self.viewer = rendering.Viewer(screen_width, screen_height)

        
        # back = rendering.FilledPolygon([(0,0), (0,screen_height),(screen_width,screen_height),(screen_width,0)])
        # back.set_color(1,1,1)
        # self.viewer.add_geom(back)

    
        # food_idx = np.argwhere(self.field==-1)[0]
        # l, r, t, b = food_idx[1] * CELL_SIZE, (food_idx[1] + 1) * CELL_SIZE, (self.H - food_idx[0]) * CELL_SIZE, (self.H - food_idx[0] - 1) * CELL_SIZE
        # l, r, t, b = l + PAD, r - PAD, t - PAD, b + PAD
        # food = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        # food.set_color(0, 1, 0)
        # self.viewer.add_geom(food)

        # snake_idxs = np.argwhere(self.field>0)
        # snake_lifes = [self.field[tuple(idx)] for idx in snake_idxs]
        # snake_idxs = [(idx, self.field[tuple(idx)]) for idx in snake_idxs]
        # snake_idxs.sort(key=lambda x: x[1])
        # idx0 = snake_idxs[0][0]
        # l0, r0, t0, b0 = idx0[1] * CELL_SIZE, (idx0[1] + 1) * CELL_SIZE, (self.H - idx0[0]) * CELL_SIZE, (self.H - idx0[0] - 1) * CELL_SIZE
        # l0, r0, t0, b0 = l0 + PAD, r0 - PAD, t0 - PAD, b0 + PAD
        # for idx in snake_idxs[1:]:
        #     idx = idx[0]
        #     l, r, t, b = idx[1] * CELL_SIZE, (idx[1] + 1) * CELL_SIZE, (self.H - idx[0]) * CELL_SIZE, (self.H - idx[0] - 1) * CELL_SIZE
        #     l, r, t, b = l + PAD, r - PAD, t - PAD, b + PAD
        #     step = idx-idx0
        #     if np.array_equal(step, SnakeEnv.step_directions[0]):
        #         cell = rendering.FilledPolygon([(l0,t0), (r0,t0), (r0,b), (l,b)])
        #     elif np.array_equal(step, SnakeEnv.step_directions[1]):
        #         cell = rendering.FilledPolygon([(l,t0), (r0,t0), (r0,b0), (l,b0)])
        #     elif np.array_equal(step, SnakeEnv.step_directions[2]):
        #         cell = rendering.FilledPolygon([(l,t), (r,t), (r,b0), (l,b0)])
        #     elif np.array_equal(step, SnakeEnv.step_directions[3]):
        #         cell = rendering.FilledPolygon([(l0,t0), (r,t), (r,b), (l0,b0)])
        #     else:
        #         print('oops')
        #     cell.set_color(0.5, 0.5, 0.5)
        #     self.viewer.add_geom(cell)
        #     idx0 = idx
        #     l0, r0, t0, b0 = l, r, t, b

        # # for i in range(11):
        # #     print(self.field[i,:])
        # return self.viewer.render(return_rgb_array = mode=='rgb_array')
 


if __name__ == '__main__':
    env = gym.make('Snake-v0')
    env.reset()
    # action_chain = [2,3,2,3,2,1,1,1,1,0,1,2,2,3,3,3,3,2,3,2,3,2,3,2,3,0]
    action_chain = [3,2,3,2,3]
    # for action in action_chain:
    done = False
    i = 0
    while not done:
        env.render()
        if True:
            import msvcrt
            key = ord(msvcrt.getch())
            if key == 224: #Special keys (arrows, f keys, ins, del, etc.)
                key = ord(msvcrt.getch())
                if key == 80: #Down arrow
                    action = 0
                elif key == 72: #Up arrow
                    action = 2
                elif key == 75: #left arrow
                    action = 1
                elif key == 77: #right arrow
                    action = 3
            else:
                pass
        else:
            action = action_chain[i]
            i += 1
            time.sleep(0.4)
        state, reward, done, info = env.step(action)
        if done:
            state = env.reset()
            break