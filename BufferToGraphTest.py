import unittest
from graph_nets import utils_np
import networkx as nx
import matplotlib.pyplot as plt
import random
import gym
import aoi_envs


class BufferToGraphTest(unittest.TestCase):
    def setUp(self):
        self.stat = gym.make('StationaryEnv-v0')

    def test_graph_creation_from_buffer(self):
        observation = self.stat.reset()
        for i in range(50):
            self.stat.is_interference = False
            action = self.stat.action_space.sample()
            observation, reward, done, info = self.stat.step(action)
            if i == 40:
                print(observation)

        graphs_nx = utils_np.data_dict_to_networkx(observation)
        ax = plt.figure(figsize=(25, 25)).gca()
        number_of_colors = self.stat.env.n_agents

        color = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(number_of_colors)]
        nc = [i for i in color for _ in range(number_of_colors)]
        nx.draw_networkx(graphs_nx, ax=ax, node_color=nc, pos=nx.kamada_kawai_layout(graphs_nx))
        _ = ax.set_title("Tree Structures from Buffers")
        plt.savefig('buffer_to_graphs_kk.png')


if __name__ == '__main__':
    unittest.main()

