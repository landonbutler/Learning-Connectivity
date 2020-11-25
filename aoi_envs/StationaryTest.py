import unittest
import numpy as np
from Stationary import StationaryEnv
import imageio
import unittest
from graph_nets import utils_np
import networkx as nx
import matplotlib.pyplot as plt
import random
import gym
import os

# TODO update tests to work with communication index actions

class StationaryTest(unittest.TestCase):
    def setUp(self):
        self.env = StationaryEnv()
        self.env.reset()

    def test_update_buffers(self):
        # print("sample action")
        attempt_comm = self.env.action_space.sample()
        # print(attempt_comm)

        # print("new buffer states")
        self.env.update_buffers(attempt_comm)
        # print(self.env.network_buffer)

    """
    def test_render(self):
        n = 50
        for i in range(n):
            attempt_comm = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(attempt_comm)
            self.env.render()
        
        with imageio.get_writer('visuals/bufferTrees/randomSelection.gif', mode='I', duration=.3) as writer:
            for i in range(1,n+1):
                fileloc = 'visuals/bufferTrees/ts'+str(i)+'.png'
                image = imageio.imread(fileloc)
                writer.append_data(image)
                os.remove(fileloc)
    """

    def test_greedy(self):
        n = 50
        for i in range(n):
            attempt_comm = self.env.mst_controller()
            observation, reward, done, info = self.env.step(attempt_comm)
            self.env.render_interference()
        
        with imageio.get_writer('visuals/interference/mstInterference200.gif', mode='I', duration=.5) as writer:
            for i in range(1,n+1):
                fileloc = 'visuals/interference/ts'+str(i)+'.png'
                image = imageio.imread(fileloc)
                writer.append_data(image)
                os.remove(fileloc)

if __name__ == '__main__':
    unittest.main()
