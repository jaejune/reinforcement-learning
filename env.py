from gym import Env, utils
from gym.spaces import Discrete, Box
import numpy as np
from numpy.core.fromnumeric import reshape
from io import StringIO
import time
from terminaltables import SingleTable

# 액션 값
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# 목표지점 도달 시 받는 리워드 값
GOAL = 1000

# 장애물에 갔을 시 받는 리워드 값
DEAD = -100

# S 시작점
# H 장애물
# G 목표지점
MAPS = {
    "6x6": [
        "H H H H H H",
        "H S -5 -1 -5 H",
        "H -1 -1 H -1 H",
        "H -3 H -1 1 H",
        "H 2 -1 -1 -1 H",
        "H H H G H H",
        ],
    "4x4": [
        "H H H H H",
        "H S -1 -1 H",
        "H -1 -1 -1 H",
        "H -1 -1 -1 H",
        "H H G H H",
    ]
}

class CustomEnv(Env):
    def __init__(self, map_name):
        map = MAPS[map_name]

        self.map = map = np.array([i.split(' ') for i in map])

        self.ncol, self.nrow = nrow, ncol = self.map.shape
        
        actions = 4

        self.observation_space = Box(low=np.array([-np.inf]), high=np.array([np.inf]))

        self.state_size = states = nrow * ncol

        self.action_space = Discrete(actions)

        self.observation = observation = {state: {action: [] for action in range(actions)} for state in range(states)}        

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):

            if a == LEFT:
                col = max(col - 1, 0)
    
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
           
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
     
            elif a == UP:
                row = max(row - 1, 0)
    
            return (row, col)

        def get_next_state(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = map[newrow, newcol]
            done = newletter in 'GH'

            if newletter == 'G':
                reward = GOAL
            elif newletter == 'H':
                reward = DEAD
            elif newletter == 'S':
                reward = 0
            else:
                reward = int(newletter)

            return newstate, reward, done
            

        for row in range(nrow):
            for col in range(ncol):
                state = to_s(row, col)
                for action in range(actions):
                    info = observation[state][action]
                    letter = map[row][col]
                    if letter == 'S':
                        self.state = self.start = state
                    if letter == 'G':
                        info.append((state, GOAL, True, letter))
                    elif letter == 'H':
                        info.append((state, DEAD, True, letter))
                    else:
                        newstate, reward, done = get_next_state(row, col, action)
                        info.append((newstate, reward, done, letter))
        


    def step(self, action):
        self.state, reward, done, info = self.observation[self.state][action][0]
        self.lastaction = action
        return self.state, reward, done, info

    def reset(self):
        self.state = self.start
        self.lastaction = None
        return self.state

    def render(self, render=True):
        if render:
            outfile = StringIO()
            row, col = self.state // self.ncol, self.state % self.ncol
            map = self.map.tolist()

            map[row][col] = utils.colorize(map[row][col], 'red', highlight=True)
            table_instance = SingleTable(map)
            table_instance.inner_heading_row_border = False
            table_instance.inner_row_border = True
            if self.lastaction is not None:
                print(
                    "  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction])
                )
            else:
                print("\n")
            print(table_instance.table)
            # for visualize
            time.sleep(0.2)

    
        
    
