import unittest
import numpy as np
from aoi_envs.Stationary import StationaryEnv
import unittest
from graph_nets import utils_np
import networkx as nx
import matplotlib.pyplot as plt
import random
import gym

# TODO update tests to work with communication index actions

class StationaryTest(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('StationaryEnv-v0')
        self.env.reset()

    def test_reset_and_render(self):
        print("attempting reset")
        self.env.reset()
        self.env.render()

    # def test_interference_1(self):
    #     attempt_comm = self.generate_attempted_communications(1, 15)
    #     succ_comm = self.env.interference(attempt_comm)
    #     print(succ_comm)
    #
    # def test_interference_k(self):
    #     attempt_comm = self.generate_attempted_communications(int(self.env.env.n_agents / 2), 15)
    #     succ_comm = self.env.interference(attempt_comm)
    #     print(succ_comm)

    def test_update_buffers(self):
        attempt_comm = self.generate_attempted_communications(int(self.env.env.n_agents / 2), 15)
        print(attempt_comm)
        self.env.env.update_buffers(attempt_comm)
        print(self.env.env.network_buffer)

    def test_multiple_steps_without_interference(self):
        for i in range(5):
            attempt_comm = self.generate_attempted_communications(int(self.env.env.n_agents / 2), 15)
            self.env.is_interference = False
            observation, reward, done, info = self.env.step(attempt_comm)
        print(observation)

    def generate_attempted_communications(self, trans_per_agent, max_trans_pow):
        return self.env.action_space.sample()

    # TODO data_dict_to_networkx wants lists for the receivers and senders, not np arrays
    # def test_graph_creation_from_buffer(self):
    #     self.env = gym.make('StationaryEnv-v0')
    #     observation = self.env.reset()
    #     for i in range(50):
    #         self.env.is_interference = False
    #         action = self.env.action_space.sample()
    #         observation, reward, done, info = self.env.step(action)
    #         if i == 40:
    #             print(observation)
    #
    #     graphs_nx = utils_np.data_dict_to_networkx(observation)
    #     ax = plt.figure(figsize=(25, 25)).gca()
    #     number_of_colors = self.env.env.n_agents
    #
    #     color = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(number_of_colors)]
    #     nc = [i for i in color for _ in range(number_of_colors)]
    #     nx.draw_networkx(graphs_nx, ax=ax, node_color=nc, pos=nx.kamada_kawai_layout(graphs_nx))
    #     _ = ax.set_title("Tree Structures from Buffers")
    #     plt.savefig('buffer_to_graphs_kk.png')


if __name__ == '__main__':
    unittest.main()
