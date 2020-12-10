from aoi_envs.MultiAgent import MultiAgentEnv
import numpy as np

N_NODE_FEAT = 6


class MobileEnv(MultiAgentEnv):

    def __init__(self):
        super().__init__()
        self.constant_v = 0.0
        self.ts_length = 0.01

        self.n_features = N_NODE_FEAT  # (TransTime, Parent Agent, PosX, PosY, VelX, VelY)
        self.recompute_solution = True
        self.mobile_agents = True
        self.x = np.zeros((self.n_agents, self.n_features))

    def reset(self):
        super().reset()
        angle = np.pi * np.random.uniform(0, 2, size=(self.n_agents,))
        self.x[:, 2] = self.constant_v * np.cos(angle)
        self.x[:, 3] = self.constant_v * np.sin(angle)

        self.network_buffer[:, :, 4] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 2].reshape(self.n_agents, 1), self.network_buffer[:, :, 4])
        self.network_buffer[:, :, 5] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 3].reshape(self.n_agents, 1), self.network_buffer[:, :, 5])
        self.network_buffer[:, :, 0] = np.where(np.eye(self.n_agents, dtype=np.bool), 0, -100)
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
