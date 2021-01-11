from aoi_envs.MultiAgent import MultiAgentEnv
import numpy as np

EPISODE_LENGTH = 500


class LastAttemptEnv(MultiAgentEnv):

    def __init__(self):
        super().__init__(eavesdropping=False, fractional_power_levels=[0.25])

    def reset(self):
        return super().reset()

    def step(self, attempted_transmissions):
        """
        Apply agent actions to update environment.
        In the future, we could create a nxn continuous action space of transmit powers, and keep max k transmits.
        :param attempted_transmissions: n-vector of index of who to communicate with
        :return: Environment observations as a dict representing the graph.
        """
        assert (self.comm_model is "push" or self.comm_model is "tw")

        if self.comm_model is "tw":
            self.timestep = self.timestep + 0.5
            # my information is updated
            self.network_buffer[:, :, 0] += np.eye(self.n_agents) * 0.5

        if self.comm_model is "push":
            self.timestep = self.timestep + 1.0
            # my information is updated
            self.network_buffer[:, :, 0] += np.eye(self.n_agents) * 1.0

        self.attempted_transmissions = attempted_transmissions // len(self.power_levels)
        transmission_indexes = attempted_transmissions // len(self.power_levels)

        self.tx_power = attempted_transmissions % len(self.power_levels)

        if self.is_interference:
            # calculates interference from attempted transmissions
            transmission_indexes, response_indexes = self.interference(self.attempted_transmissions, self.tx_power)

        self.successful_transmissions = transmission_indexes

        self.update_buffers(transmission_indexes)

        if self.comm_model is "tw":
            # Two-Way Communications can be modeled as a sequence of a push and a response
            self.timestep = self.timestep + 0.5
            # my information is updated
            self.network_buffer[:, :, 0] += np.eye(self.n_agents) * 0.5

            self.update_buffers(response_indexes, push=False)

        if not self.network_connected:
            self.is_network_connected()

        self.network_buffer[np.arange(self.n_agents), self.attempted_transmissions, 4] = self.timestep

        return self.get_relative_network_buffer_as_dict(), - self.instant_cost() / EPISODE_LENGTH, False, {}


    def get_relative_network_buffer_as_dict(self):
        """
        Compute local node observations.
        :return: A dict representing the current routing buffers.
        """
        # timesteps and positions won't be relative within env, but need to be when passed out
        self.relative_buffer[:] = self.network_buffer
        self.relative_buffer[:, :, 0] -= self.timestep
        self.relative_buffer[:, :, 4] -= self.timestep

        self.relative_buffer[:, :, 4] /= EPISODE_LENGTH
        self.relative_buffer[:, :, 0] /= EPISODE_LENGTH

        # fills rows of a nxn matrix, subtract that from relative_network_buffer
        self.relative_buffer[:, :, 2:4] -= self.x[:, 0:2].reshape(self.n_agents, 1, 2)
        self.relative_buffer[:, :, 2:4] /= self.r_max

        # align to the observation space and then pass that input out MAKE SURE THESE ARE INCREMENTED
        return self.map_to_observation_space(self.relative_buffer)
