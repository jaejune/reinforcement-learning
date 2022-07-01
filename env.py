from gym import Env, utils
from gym.spaces import Discrete, Box
import numpy as np
from numpy.core.fromnumeric import reshape
from io import StringIO
import time
from terminaltables import SingleTable
import pandas as pd
import os 

# 액션 값
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# 목표지점 도달 시 받는 리워드 값
GOAL = 1

# 장애물에 갔을 시 받는 리워드 값
DEAD = -1

# S 시작점
# H 장애물
# G 목표지점

class CustomEnv(Env):
    def __init__(self, map_name):
        path = os.getcwd()
        self.map = map = pd.read_excel(f'{path}/{map_name}.xlsx', header=None).to_numpy()

        self.ncol, self.nrow = nrow, ncol = self.map.shape
        
        actions = 4

        self.step_count = 0

        self.observation_space = Box(low=np.array([-np.inf]), high=np.array([np.inf]))

        self.prob = self.state_size = states = nrow * ncol

        self.action_space = Discrete(actions)

        self.observation = observation = {state: {action: [] for action in range(actions)} for state in range(states)}        

        self.hall_coordis = list()

        self.goal_coordi = list()

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
            new_letter = map[newrow, newcol]
            
            done = str(new_letter) in 'G'

            if new_letter == 'G':
                reward = GOAL
            elif new_letter == 'H':
                reward = DEAD
            elif new_letter == 'S':
                reward = 0
            else:
                reward = int(new_letter)

            return newstate, reward, done, new_letter
            

        for row in range(nrow):
            for col in range(ncol):
                state = to_s(row, col)
                for action in range(actions):
                    info = observation[state][action]
                    letter = map[row][col]
                    if letter == 'S':
                        self.state = self.start = state

                    elif letter == 'H':
                        self.hall_coordis.append([row, col])

                    elif letter == 'G':
                        self.goal_coordi.append([row, col])
                    
                    newstate, reward, done, newletter = get_next_state(row, col, action)
                    if newletter == 'H':
                        info.append((state, reward, done, newletter))

                    info.append((newstate, reward, done, newletter))
        
    def step(self, action):
        self.state, reward, done, info = self.observation[self.state][action][0]

        if reward > 0 and self.state not in self.good_way_list:
            reward += self.prob ** 2
            self.good_way_list.append(self.state)
            self.prob -= 1

        elif reward < 0 and self.state not in self.good_way_list:
            reward -= self.prob ** 2
            self.prob -= 1

        else:
            reward -= self.step_count ** 2
            self.step_count += 1


        self.lastaction = action
        return self.state, reward, done, info

    def reset(self):
        self.good_way_list = list()
        self.step_count = 0
        self.state = self.start
        self.prob = self.state_size
        
        self.lastaction = None
        return self.state

    def render(self, render=True):
        if render:
            outfile = StringIO()
            row, col = self.state // self.ncol, self.state % self.ncol
            map = self.map.tolist()

            map[row][col] = utils.colorize(map[row][col], 'red', highlight=True)
            for rowCol in self.hall_coordis:
                map[rowCol[0]][rowCol[1]] = utils.colorize(map[rowCol[0]][rowCol[1]], 'blue', highlight=True)
            map[self.goal_coordi[0][0]][self.goal_coordi[0][1]] = utils.colorize( map[self.goal_coordi[0][0]][self.goal_coordi[0][1]], 'yellow', highlight=True)

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
            time.sleep(0.1)

    
        
    
