from aoi_envs.MultiAgent import MultiAgentEnv
import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from graph_nets import utils_np
import tensorflow as tf
import networkx as nx

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}
N_NODE_FEAT = 6


class MobileEnv(MultiAgentEnv):

    def __init__(self):
        super().__init__()

        self.constant_v = 10
        self.ts_length = 0.01

        self.n_features = N_NODE_FEAT  # (TransTime, Parent Agent, PosX, PosY, VelX, VelY)

    def reset(self):
        x = np.zeros((self.n_agents, 4))
        x[:, 0:2] = np.random.uniform(-self.r_max, self.r_max, size=(self.n_agents, 2))

        x_loc = np.reshape(x[:, 0:2], (self.n_agents, 2, 1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0, 2, 1)) - np.transpose(x_loc, (2, 0, 1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)

        angle = np.pi * np.random.uniform(0, 2, size=(self.n_agents,))
        x[:, 2] = self.constant_v * np.cos(angle)
        x[:, 3] = self.constant_v * np.sin(angle)

        self.mst_action = None

        self.timestep = 0
        self.x = x
        self.u = np.zeros((self.n_agents, 2))

        self.network_buffer = np.zeros((self.n_agents, self.n_agents, self.n_features))

        self.network_buffer[:, :, 1] = -1  # no parent references yet

        # TODO test this
        # If the agents were mobile, we need to add this code into the step() function too
        self.network_buffer[:, :, 2] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 0].reshape(self.n_agents, 1), self.network_buffer[:, :, 2])
        self.network_buffer[:, :, 3] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 1].reshape(self.n_agents, 1), self.network_buffer[:, :, 3])
        self.network_buffer[:, :, 4] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 2].reshape(self.n_agents, 1), self.network_buffer[:, :, 4])
        self.network_buffer[:, :, 5] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 3].reshape(self.n_agents, 1), self.network_buffer[:, :, 5])
        self.network_buffer[:, :, 0] = np.where(np.eye(self.n_agents, dtype=np.bool), 0, -100)

        if self.is_interference:
            self.compute_distances()
        return self.get_relative_network_buffer_as_dict()

    def step(self, attempted_transmissions):
        """
        Apply agent actions to update environment.
        In the future, we could create a nxn continuous action space of transmit powers, and keep max k transmits.
        :param attempted_transmissions: n-vector of index of who to communicate with
        :return: Environment observations as a dict representing the graph.
        """
        self.attempted_transmissions = attempted_transmissions
        successful_transmissions = attempted_transmissions

        # Transmit power can be incorporated later
        if self.is_interference:
            successful_transmissions, resp_trans = self.interference(
                attempted_transmissions)  # calculates interference from attempted transmissions
        self.successful_transmissions = successful_transmissions

        self.current_agents_choice = attempted_transmissions[0]

        if self.comm_model is "tw":
            self.update_buffers(successful_transmissions, resp_trans)
        else:
            self.update_buffers(successful_transmissions)
        # for successful transmissions, updates the buffers of those receiving information

        self.move_agents()

        self.timestep = self.timestep + 1
        return self.get_relative_network_buffer_as_dict(), - self.instant_cost(), False, {}

    def move_agents(self):
        new_pos = self.x[:, 0:2] + self.x[:, 2:4] * self.ts_length
        self.x[:, 0] = np.clip(new_pos[:, 0], -self.r_max, self.r_max)
        self.x[:, 1] = np.clip(new_pos[:, 1], -self.r_max, self.r_max)

        self.x[:, 2] = np.where((self.x[:, 0] - new_pos[:, 0]) == 0, self.x[:, 2], -self.x[:, 2])
        self.x[:, 3] = np.where((self.x[:, 1] - new_pos[:, 1]) == 0, self.x[:, 3], -self.x[:, 3])

        self.network_buffer[:, :, 4] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 2].reshape(self.n_agents, 1), self.network_buffer[:, :, 4])
        self.network_buffer[:, :, 5] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 3].reshape(self.n_agents, 1), self.network_buffer[:, :, 5])

    def render(self, mode='human', controller="Random", save_plots=False):
        super().render(controller=controller, save_plots=save_plots, mobile=True)

    # Given current positions, will return who agents should communicate with to form the Minimum Spanning Tree
    def mst_controller(self):
        self.compute_distances()
        self.dist = np.sqrt(self.r2)
        G = nx.from_numpy_array(self.dist, create_using=nx.Graph())
        T = nx.minimum_spanning_tree(G)
        degrees = [val for (node, val) in T.degree()]

        parent_refs = np.array(self.find_parents(T, [-1] * self.n_agents, degrees))
        self.mst_action = parent_refs.astype(int)
        tx_prob = np.random.uniform(size=(self.n_agents,))
        return np.where(tx_prob < self.transmission_probability, self.mst_action, np.arange(self.n_agents))
