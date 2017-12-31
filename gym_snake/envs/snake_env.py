import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time
import itertools
from gym.envs.classic_control import rendering

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human'],
                'video.frames_per_second': 50}
    W = 10  # 20
    H = 5  # 11
    L0 = 4  # 9
    step_diffs = np.array([W, -1, -W, 1])
    step_dict = {W:'00', -1:'01', -W:'10', 1:'11'}
    every_space_flat = np.array(range(W * H))
    action_space = spaces.Discrete(4)
    lows = np.array([0, 0, 0])
    highs = np.array([W*H, 2**(W*H), W*H])
    observation_space = spaces.Box(lows, highs)
    R_DIE = -100
    R_STEP = 1
    R_FOOD = 100
    R_ILLEGAL = 0

    def __init__(self):
        self.viewer = None
        self.head_new = None
        self.food_new = None
        self.body_new = None
        self.empty_spaces_new = None
        self.state = None


    def addFood(self):
        snake = np.append([self.head_new], self.body_new, 0)
        empty_spaces = np.setdiff1d(SnakeEnv.every_space_flat, snake, True)
        if np.size(empty_spaces) > 0:
            self.food_new = np.random.choice(empty_spaces)
        else:
            pass

        
    def _step(self, action):
        # 0 1 2 3 = Down Left Up Right
        head_diff = SnakeEnv.step_diffs[action]
        head_try = self.head_new + head_diff
        body_try = np.insert(self.body_new, 0, [self.head_new], 0)  # add old head to body
        if head_try < 0 or head_try >= SnakeEnv.W * SnakeEnv.H or (head_try % SnakeEnv.W == 0 and action == 3) or (head_try % SnakeEnv.W == SnakeEnv.W - 1 and action == 1):  # hit a wall
            # print('hit a wall')
            reward = SnakeEnv.R_DIE
            done = True
        elif np.isin([head_try], self.body_new, True):  # hit body
            # print('hit body...')
            if head_try == self.body_new[0]:  # try to hit neck, illegal move
                # print('hit neck')
                reward = SnakeEnv.R_ILLEGAL
                done = False
            else:  # hit rest of body, lose
                # print('die')
                reward = SnakeEnv.R_DIE
                done = True
        else:  # move
            # print('move...')
            self.body_new = body_try
            # print('head_try: {}'.format(head_try))
            # print('self.food_new: {}'.format(self.food_new))
            if head_try == self.food_new:  # eat a food
                # print('eat a food')
                self.head_new = head_try
                self.addFood()
                reward = SnakeEnv.R_FOOD
                if np.size(self.body_new) == SnakeEnv.W * SnakeEnv.H - 1: #  won the game
                    reward = 1000
                    done = True
            else:
                # print('remove tail')
                self.body_new = np.delete(self.body_new, np.size(self.body_new,0) - 1, 0)  # remove tail from body
                reward = SnakeEnv.R_STEP
                self.head_new = head_try
            done = False
        relative_snake = np.subtract(self.body_new, np.insert(self.body_new[0:np.size(self.body_new) - 1], 0, [self.head_new], 0))
        snake_shape_string = '0b1' + ''.join([SnakeEnv.step_dict[x] for x in relative_snake])  # 1bit prefix to differentiate different length zerostrings
        snake_shape_code = int(snake_shape_string, 2)
        state = (self.head_new, snake_shape_code, self.food_new)
        return state, reward, done, {}

    def _reset(self):
        self.head_new = SnakeEnv.W * (SnakeEnv.H - 1) + SnakeEnv.L0 - 1
        self.body_new =np.array([SnakeEnv.W * (SnakeEnv.H - 1) + x for x in range(SnakeEnv.L0 - 2,-1,-1)])
        self.addFood()
        snake_shape_code = int('0b1' + '01' * (SnakeEnv.L0 - 1), 2)
        state = (self.head_new, snake_shape_code, self.food_new)
        return state

    def state2field(self, head, body, food):
        field = np.zeros((SnakeEnv.H, SnakeEnv.W))
        age = np.size(body, 0) + 1
        field[np.unravel_index(head, (SnakeEnv.H, SnakeEnv.W))] = age
        for cell in body:
            age -= 1
            field[np.unravel_index(cell, (SnakeEnv.H, SnakeEnv.W))] = age
        field[np.unravel_index(food, (SnakeEnv.H, SnakeEnv.W))] = -1
        return field


    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        
        CELL_SIZE = 30
        PAD = 2
        screen_width = CELL_SIZE * SnakeEnv.W + 2
        screen_height = CELL_SIZE * SnakeEnv.H + 2


        

        
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
        
    # lines
        self.viewer.geoms = []
        snake = np.append([self.head_new], self.body_new, 0)
        body_unravel = np.unravel_index(snake, (SnakeEnv.H, SnakeEnv.W))
        body_unravel = np.transpose(body_unravel)
        body_unravel = np.multiply(body_unravel, screen_height/SnakeEnv.H + 0)
        body_unravel = np.add(body_unravel, CELL_SIZE / 2)
        body_unravel[:,0] = screen_height - body_unravel[:,0]
        body_unravel = np.flip(body_unravel, 1)
        body_unravel[:,0] = body_unravel[:,0]
        line = rendering.PolyLine(body_unravel, False)
        self.viewer.add_geom(line)
        



        food_idx = np.unravel_index(self.food_new, (SnakeEnv.H, SnakeEnv.W))
        l, r, t, b = food_idx[1] * CELL_SIZE, (food_idx[1] + 1) * CELL_SIZE, (SnakeEnv.H - food_idx[0]) * CELL_SIZE, (SnakeEnv.H - food_idx[0] - 1) * CELL_SIZE
        l, r, t, b = l + PAD, r - PAD, t - PAD, b + PAD
        food = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        food.set_color(0, 1, 0)
        self.viewer.add_geom(food)
        




        # field = self.state2field(self.head_new, self.body_new, self.food_new)
        # # food_idx = np.unravel_index(self.food_new, (SnakeEnv.H, SnakeEnv.W))
        # # l, r, t, b = food_idx[1] * CELL_SIZE, (food_idx[1] + 1) * CELL_SIZE, (SnakeEnv.H - food_idx[0]) * CELL_SIZE, (SnakeEnv.H - food_idx[0] - 1) * CELL_SIZE
        # # l, r, t, b = l + PAD, r - PAD, t - PAD, b + PAD
        # # food = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        # # food.set_color(0, 1, 0)
        # # self.viewer.add_geom(food)

        # snake_idxs = np.argwhere(field>0)
        # snake_lifes = [field[tuple(idx)] for idx in snake_idxs]
        # snake_idxs = [(idx, field[tuple(idx)]) for idx in snake_idxs]
        # snake_idxs.sort(key=lambda x: x[1])
        # idx0 = snake_idxs[0][0]
        # l0, r0, t0, b0 = idx0[1] * CELL_SIZE, (idx0[1] + 1) * CELL_SIZE, (SnakeEnv.H - idx0[0]) * CELL_SIZE, (SnakeEnv.H - idx0[0] - 1) * CELL_SIZE
        # l0, r0, t0, b0 = l0 + PAD, r0 - PAD, t0 - PAD, b0 + PAD

        # for idx in snake_idxs[1:]:
        #     idx = idx[0]
        #     l, r, t, b = idx[1] * CELL_SIZE, (idx[1] + 1) * CELL_SIZE, (SnakeEnv.H - idx[0]) * CELL_SIZE, (SnakeEnv.H - idx[0] - 1) * CELL_SIZE
        #     l, r, t, b = l + PAD, r - PAD, t - PAD, b + PAD
        #     c = [idx[1] * CELL_SIZE + CELL_SIZE / 2, (SnakeEnv.H - idx[0]) * CELL_SIZE + CELL_SIZE / 2]
            
        #     step = idx-idx0
        #     if np.array_equal(step, [1,0]):
        #         cell = rendering.FilledPolygon([(l0,t0), (r0,t0), (r0,b), (l,b)])
        #     elif np.array_equal(step, [0,-1]):
        #         cell = rendering.FilledPolygon([(l,t0), (r0,t0), (r0,b0), (l,b0)])
        #     elif np.array_equal(step, [-1,0]):
        #         cell = rendering.FilledPolygon([(l,t), (r,t), (r,b0), (l,b0)])
        #     elif np.array_equal(step, [0,1]):
        #         cell = rendering.FilledPolygon([(l0,t0), (r,t), (r,b), (l0,b0)])
        #     else:
        #         print('oops')
        #     cell.set_color(0.5, 0.5, 0.5)
        #     self.viewer.add_geom(cell)
        #     idx0 = idx
        #     l0, r0, t0, b0 = l, r, t, b

        # # for i in range(11):
        # #     print(self.field[i,:])
        return self.viewer.render()
 


if __name__ == '__main__':
    env = gym.make('Snake-v0')
    env.reset()
    # action_chain = [2,3,2,3,2,1,1,1,1,0,1,2,2,3,3,3,3,2,3,2,3,2,3,2,3,0]
    action_chain = [3]*(SnakeEnv.W - SnakeEnv.L0)+[2]+([2]*(SnakeEnv.H - 2)+[1]+[0]*(SnakeEnv.H - 2)+[1])*(int(SnakeEnv.W / 2) - 1)+[2]*(SnakeEnv.H - 2)+[1]+[0]*(SnakeEnv.H - 2)+[0]+[3]*(SnakeEnv.L0 - 1)
    # for action in action_chain:
    done = False
    i = 0
    while not done:
        env.render()
        if False:
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
            if i == len(action_chain):
                i = 0
            # time.sleep(0.01)
        state, reward, done, info = env.step(action)
        
        
        if done:
            state = env.reset()
            break