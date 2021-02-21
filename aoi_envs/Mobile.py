from aoi_envs.MultiAgent import MultiAgentEnv
import numpy as np

class MobileEnv(MultiAgentEnv):

    def __init__(self, agent_velocity=1.0, initialization='Random', biased_velocities=False, flocking=False,
                 random_acceleration=True, aoi_reward=True, flocking_position_control=False, num_agents=40):
        super().__init__(eavesdropping=True, fractional_power_levels=[0.25, 0.0], initialization=initialization,
                         aoi_reward=aoi_reward, num_agents=num_agents)

        self.ts_length = 0.1
        self.max_velocity = agent_velocity * self.distance_scale / self.ts_length / self.episode_length
        self.max_acceleration = 10.0
        self.gain = 1.0

        self.recompute_solution = True
        self.mobile_agents = True

        self.flocking = flocking
        self.flocking_position_control = flocking_position_control
        self.random_acceleration = random_acceleration
        self.biased_velocities = biased_velocities

    def reset(self):
        super().reset()

        if self.random_acceleration or (self.flocking and not self.biased_velocities):
            self.x[:, 2:4] = np.random.uniform(-self.max_velocity, self.max_velocity, size=(self.n_agents, 2))
        elif self.flocking and self.biased_velocities:
            self.x[:, 2:4] = np.random.uniform(0.5 * -self.max_velocity, 0.5 * self.max_velocity, size=(self.n_agents, 2))
            self.x[:, 2:4] = self.x[:, 2:4] + np.random.uniform(0.5 * -self.max_velocity, 0.5 * self.max_velocity, size=(1, 2))
        else:
            angle = np.pi * np.random.uniform(0, 2, size=(self.n_agents,))
            self.x[:, 2] = self.max_velocity * np.cos(angle)
            self.x[:, 3] = self.max_velocity * np.sin(angle)

        self.network_buffer[:, :, 4:6] = np.where(self.diag,
                                                  self.x[:, 2:4].reshape(self.n_agents, 1, 2),
                                                  self.network_buffer[:, :, 4:6])

        return self.get_relative_network_buffer_as_dict()

    def step(self, attempted_transmissions):
        self.move_agents()
        return super().step(attempted_transmissions)

    def potential_grad(self, pos_diff, r2):
        grad = -2.0 * np.divide(pos_diff, np.multiply(r2, r2)) + 2 * np.divide(pos_diff, r2)
        return grad

    def move_agents(self):
        new_pos = self.x[:, 0:2] + self.x[:, 2:4] * self.ts_length

        if self.flocking or self.random_acceleration:
            if self.flocking:
                known_velocities = np.copy(self.network_buffer[:, :, 4:6])
                known_velocities[known_velocities == 0] = np.nan
                known_velocities -= (self.x[:, 2:4])[:, np.newaxis, :]
                acceleration = np.nanmean(known_velocities, axis=1)

                if self.flocking_position_control:
                    steady_state_scale = self.r_max / 5.0
                    known_positions = np.copy(self.network_buffer[:, :, 2:4])
                    known_positions[known_positions == 0] = np.nan
                    known_positions = (known_positions - (self.x[:, 2:4])[:, np.newaxis, :]) / steady_state_scale
                    r2 = np.sum(known_positions ** 2, axis=2)[:, :, np.newaxis]
                    grad = -2.0 * np.divide(known_positions, np.multiply(r2, r2)) + 2 * np.divide(known_positions, r2)
                    acceleration += np.nansum(grad, axis=1) * steady_state_scale

            else:
                # acceleration = np.random.uniform(-self.max_acceleration, self.max_acceleration, size=(self.n_agents, 2))
                acceleration = np.random.normal(0., self.max_acceleration / 3.0, size=(self.n_agents, 2))

            acceleration = np.clip(acceleration, -self.max_acceleration, self.max_acceleration)
            self.x[:, 2:4] += self.gain * self.ts_length * acceleration
            self.x[:, 2:4] = np.clip(self.x[:, 2:4], -self.max_velocity, self.max_velocity)

        if self.flocking:
            self.x[:, 0:2] = new_pos
        else:
            self.x[:, 0:2] = np.clip(new_pos[:, 0:2], -self.r_max, self.r_max)
            self.x[:, 2:4] = np.where((self.x[:, 0:2] - new_pos[:, 0:2]) == 0, self.x[:, 2:4], -self.x[:, 2:4])

        self.network_buffer[:, :, 2:6] = np.where(self.diag,
                                                  self.x[:, 0:4].reshape(self.n_agents, 1, 4),
                                                  self.network_buffer[:, :, 2:6])
