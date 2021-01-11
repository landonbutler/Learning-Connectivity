from aoi_envs.MultiAgent import MultiAgentEnv
import numpy as np

N_NODE_FEAT = 6


class MobileEnv(MultiAgentEnv):

    def __init__(self, agent_velocity=1.0):
        super().__init__(eavesdropping=True, fractional_power_levels=[])
        self.max_v = agent_velocity * self.r_max  # for strictly mobile agents, this is the constant velocity
        self.ts_length = 0.01

        self.n_features = N_NODE_FEAT  # (TransTime, Parent Agent, PosX, PosY, VelX, VelY)
        self.recompute_solution = True
        self.mobile_agents = True
        self.x = np.zeros((self.n_agents, self.n_features))

        self.flocking = False
        self.biased_velocities = False

    def reset(self):
        super().reset()
        if self.flocking:
            self.x[:, 2] = np.random.uniform(0.5 * -self.max_v, 0.5 * self.max_v, size=(self.n_agents,))
            self.x[:, 3] = np.random.uniform(0.5 * -self.max_v, 0.5 * self.max_v, size=(self.n_agents,))
            if self.biased_velocities:
                bias = np.random.uniform(0.5 * -self.max_v, 0.5 * self.max_v, size=(1, 2))
                self.x[:, 2:4] = self.x[:, 2:4] + bias
        else:
            angle = np.pi * np.random.uniform(0, 2, size=(self.n_agents,))
            self.x[:, 2] = self.max_v * np.cos(angle)
            self.x[:, 3] = self.max_v * np.sin(angle)

        self.network_buffer[:, :, 4] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 2].reshape(self.n_agents, 1), self.network_buffer[:, :, 4])
        self.network_buffer[:, :, 5] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 3].reshape(self.n_agents, 1), self.network_buffer[:, :, 5])
        return self.get_relative_network_buffer_as_dict()

    def step(self, attempted_transmissions):
        self.move_agents()
        return super().step(attempted_transmissions)

    def move_agents(self):
        if self.flocking:
            known_x_velocities = self.network_buffer[:, :, 4]
            known_x_velocities[known_x_velocities == 0] = np.nan
            self.x[:, 2] = np.nanmean(known_x_velocities, axis=1)

            known_y_velocities = self.network_buffer[:, :, 5]
            known_y_velocities[known_y_velocities == 0] = np.nan
            self.x[:, 3] = np.nanmean(known_y_velocities, axis=1)

            self.x[:, 0:2] = self.x[:, 0:2] + self.x[:, 2:4] * self.ts_length

        else:
            new_pos = self.x[:, 0:2] + self.x[:, 2:4] * self.ts_length
            self.x[:, 0] = np.clip(new_pos[:, 0], -self.r_max, self.r_max)
            self.x[:, 1] = np.clip(new_pos[:, 1], -self.r_max, self.r_max)

            self.x[:, 2] = np.where((self.x[:, 0] - new_pos[:, 0]) == 0, self.x[:, 2], -self.x[:, 2])
            self.x[:, 3] = np.where((self.x[:, 1] - new_pos[:, 1]) == 0, self.x[:, 3], -self.x[:, 3])

        self.network_buffer[:, :, 2] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 0].reshape(self.n_agents, 1), self.network_buffer[:, :, 2])
        self.network_buffer[:, :, 3] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 1].reshape(self.n_agents, 1), self.network_buffer[:, :, 3])
        self.network_buffer[:, :, 4] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 2].reshape(self.n_agents, 1), self.network_buffer[:, :, 4])
        self.network_buffer[:, :, 5] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 3].reshape(self.n_agents, 1), self.network_buffer[:, :, 5])
