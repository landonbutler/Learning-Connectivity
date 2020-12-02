import unittest
import numpy as np
from Stationary import StationaryEnv
from Mobile import MobileEnv
import imageio
import unittest
from graph_nets import utils_np
import networkx as nx
import matplotlib.pyplot as plt
import random
import gym
import os

class EnvTest(unittest.TestCase):
    def setUp(self):
        self.env = MobileEnv()
        self.env.reset()

    def test_controller(self):
        tot_ts = 200
        fp = "visuals/interference/mobile/"
        cont = "Neo" # Greedy, MST, Random, or Neopolitan
        for i in range(tot_ts):
            if cont is "Greedy":
                attempt_comm = self.env.greedy_controller()
            elif cont is "MST":
                attempt_comm = self.env.mst_controller()
            elif cont is "Random":
                attempt_comm = self.env.random_controller()
            else:
                attempt_comm = self.env.neopolitan_controller()
            observation, reward, done, info = self.env.step(attempt_comm)
            self.env.render_interference(controller=cont, filepath=fp)
        
        with imageio.get_writer(fp + cont +'Interference' + str(self.env.n_agents) + '.gif', mode='I', duration=.05) as writer:
            for i in range(1,tot_ts+1):
                fileloc = fp + 'ts'+str(i)+'.png'
                image = imageio.imread(fileloc)
                writer.append_data(image)
                os.remove(fileloc)

if __name__ == '__main__':
    unittest.main()
