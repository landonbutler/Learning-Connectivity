import unittest
import numpy as np
from envs.Stationary import StationaryEnv
from BufferToGraph import BufferToGraph as bfg
from graph_nets import utils_np
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sonnet as snt
import random

class BufferToGraphTest(unittest.TestCase):
    def setUp(self):
        self.stat = StationaryEnv()
        self.stat.reset()

    def test_graph_creation_from_buffer(self):
        for i in range(50):
            attempt_comm = self.generate_attempted_communications(int(self.stat.n_agents / 2), 15)
            self.stat.is_interference = False
            action = self.stat.action_space.sample()
            observation, reward, done, info = self.stat.step(action)
            if i == 40:
                print(observation)
        graph_model = bfg.networkBufferGraph(observation)
        graphs_nx = utils_np.graphs_tuple_to_networkxs(graph_model)
        ax = plt.figure(figsize=(25, 25)).gca()
        number_of_colors = self.stat.n_agents

        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
        nc = [i for i in color for j in range(self.stat.n_agents)]
        nx.draw_networkx(graphs_nx[0], ax=ax, node_color = nc, pos=nx.kamada_kawai_layout(graphs_nx[0]))
        _ = ax.set_title("Tree Structures from Buffers")
        plt.savefig('buffer_to_graphs_kk.png')
        print(graph_model)

        # just the n-vector of who to communicate with
    def generate_attempted_communications(self, trans_per_agent):
        action = self.stat.action_space.sample()

        for i in range (self.stat.n_agents):

        return np.random.choice(self.stat.n_agents, size=(self.stat.n_agents,trans_per_agent), replace=False)

if __name__ == '__main__':
    unittest.main()

