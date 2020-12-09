from aoi_envs.MultiAgent import MultiAgentEnv
import numpy as np
import networkx as nx

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}
N_NODE_FEAT = 6


class MobileEnv(MultiAgentEnv):

    def __init__(self):
        super().__init__()

        self.constant_v = 1.0
        self.ts_length = 0.001

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

        # self.network_buffer[:, :, 4:6] = 0

        if self.is_interference:
            self.compute_distances()
        return self.get_relative_network_buffer_as_dict()

    def step(self, attempted_transmissions):
        self.move_agents()
        return super().step(attempted_transmissions)

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

        # self.network_buffer[:, :, 4:6] = 0

    def render(self, mode='human', controller="Random", save_plots=False):
        super().render(controller=controller, save_plots=save_plots, mobile=True)

    def mst_controller(self, selective_comms=True, transmission_probability=0.1, mobile = True):
        return super().mst_controller(selective_comms=selective_comms, transmission_probability=transmission_probability, mobile=mobile)
    