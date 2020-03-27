"""
The Maze environment: A grid of tiles.

Red rectangle:          explorer object.

Black rectangles:       hells       [reward = -1].
Yellow bin circle:      gold        [reward = +1].
All other states:       ground      [reward = 0].

"""

import numpy as np
import time
import sys

# import appropriate tkinter package based on your python version
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 6  # grid height
MAZE_W = 6  # grid width

class Maze():
    def __init__(self):
        # We need an instance of tk.Tk class. The tk.Tk class is a top-level widget of Tk and serves as the main window of the application.
        self.window = tk.Tk()
        self.window.title('maze with Q-Learning')
        self.window.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self.action_space = ['u', 'd', 'l', 'r'] #ToDo: Fill the list with all possible actions
        self.n_actions = len(self.action_space)
        self.build_grid()

    def build_grid(self):
        self.canvas = tk.Canvas(self.window, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_W * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin point ( It is the center of the first cell in the first row)
        origin = np.array([20, 20])

        # create 2 hell points
        # hell 1
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell 2
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # create oval (the goal point)
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect (the agent)
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def render(self):
        time.sleep(0.1)
        self.window.update()

    def reset(self):
        '''
        Reset the explorer agent at the origin position.
        :return: canvas with the explorer agent at the origin position.
        '''
        self.window.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def get_state_reward(self, action):
        # get the current coordinate of explorer
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1 #ToDo: this is our gold goal! Give it a positive reward
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1 #ToDo: fall in a hole! Negative reward!
            done = True
            s_ = 'terminal'
        else:
            reward = 0 #ToDo: just moving around! no reward.
            done = False

        return s_, reward, done