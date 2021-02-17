import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import networkx as nx
import matplotlib.colors as mc
import matplotlib.ticker as mticker
import colorsys

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 11}

N_NODE_FEAT = 7
N_EDGE_FEAT = 1
TIMESTEP = 0.5

class MultiAgentEnv(gym.Env):

    def __init__(self, fractional_power_levels=[0.25, 0.0], eavesdropping=True, num_agents=40, initialization="Random",
                 aoi_reward=True, episode_length=500.0, comm_model="tw", min_sinr=1.0, last_comms=True):
        super(MultiAgentEnv, self).__init__()

        # Problem parameters
        self.last_comms = last_comms
        self.n_agents = num_agents
        self.n_nodes = self.n_agents * self.n_agents
        self.r_max = 5000.0
        self.n_features = N_NODE_FEAT  # (TransTime, Parent Agent, PosX, PosY, VelX, VelY)
        self.n_edges = self.n_agents * self.n_agents

        self.carrier_frequency_ghz = 2.4
        self.min_SINR_dbm = min_sinr  # 10-15 is consider unreliable, cited paper uses -4
        self.gaussian_noise_dBm = -90
        self.gaussian_noise_mW = 10 ** (self.gaussian_noise_dBm / 10)
        self.path_loss_exponent = 2
        self.aoi_reward = aoi_reward
        self.distance_scale = self.r_max

        self.fraction_of_rmax = fractional_power_levels  # [0.25, 0.125]
        self.power_levels = self.find_power_levels()  # method finding

        self.r_max *= np.sqrt(self.n_agents / 20)

        # initialize state matrices
        self.edge_features = np.zeros((self.n_nodes, 1))
        self.episode_length = episode_length
        self.penalty = 0.0
        self.x = np.zeros((self.n_agents, self.n_features))
        self.network_buffer = np.zeros((self.n_agents, self.n_agents, self.n_features))
        self.old_buffer = np.zeros((self.n_agents, self.n_agents, self.n_features))
        self.relative_buffer = np.zeros((self.n_agents, self.n_agents, self.n_features))
        self.diag = np.eye(self.n_agents, dtype=np.bool).reshape(self.n_agents, self.n_agents, 1)

        # each agent has their own action space of a n_agent vector of weights
        self.action_space = spaces.MultiDiscrete([self.n_agents * len(self.power_levels)] * self.n_agents)

        self.observation_space = spaces.Dict(
            [
                # (nxn) by (features-1) we maintain parent references by edges
                ("nodes", spaces.Box(shape=(self.n_agents * self.n_agents, N_NODE_FEAT), low=-np.Inf, high=np.Inf,
                                     dtype=np.float32)),
                # upperbound, n fully connected trees (n-1) edges
                # To-Do ensure these bounds don't affect anything
                ("edges", spaces.Box(shape=(self.n_edges, N_EDGE_FEAT), low=-np.Inf, high=np.Inf,
                                     dtype=np.float32)),
                # senders and receivers will each be one endpoint of an edge, and thus should be same size as edges
                ("senders", spaces.Box(shape=(self.n_edges, 1), low=0, high=self.n_agents,
                                       dtype=np.float32)),
                ("receivers", spaces.Box(shape=(self.n_edges, 1), low=0, high=self.n_agents,
                                         dtype=np.float32)),
                ("globals", spaces.Box(shape=(1, 1), low=0, high=self.episode_length, dtype=np.float32)),
            ]
        )

        # Plotting placeholders
        self.fig = None
        self.agent_markers = None
        self.np_random = None
        self.ax = None
        self.agent0_marker = None
        self._plot_text = None
        self.arrows = None
        self.current_arrow = None

        self.diff = None
        self.r2 = None

        self.timestep = 0
        self.avg_transmit_distance = 0

        self.symmetric_comms = True
        self.is_interference = True
        self.mst_action = None

        self.network_connected = False
        self.recompute_solution = False
        self.mobile_agents = False

        self.flocking = False
        self.biased_velocities = False

        self.known_initial_positions = False

        self.tx_power = None
        self.eavesdroppers = None
        self.eavesdroppers_response = None
        self.eavesdropping = eavesdropping

        if self.flocking:
            self.render_radius = 2 * self.r_max
        else:
            self.render_radius = self.r_max

        # Push Model: At each time step, agent selects which agent they want to 'push' their buffer to
        # Two-Way Model: An agent requests/pushes their buffer to an agent, with hopes of getting their information back
        # self.comm_model = "push"  # push or tw
        self.comm_model = comm_model  #"tw"  # push or tw

        self.attempted_transmissions = None
        self.successful_transmissions = None

        self.initial_formation = initialization

        # Packing and unpacking information
        self.keys = ['nodes', 'edges', 'senders', 'receivers', 'globals']
        self.save_plots = False
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, attempted_transmissions):
        """
        Apply agent actions to update environment.
        In the future, we could create a nxn continuous action space of transmit powers, and keep max k transmits.
        :param attempted_transmissions: n-vector of index of who to communicate with
        :return: Environment observations as a dict representing the graph.
        """
        assert (self.comm_model is "push" or self.comm_model is "tw")

        self.timestep = self.timestep + TIMESTEP
        # my information is updated
        self.network_buffer[:, :, 0] += np.eye(self.n_agents) * TIMESTEP

        self.attempted_transmissions = attempted_transmissions // len(self.power_levels)
        transmission_indexes = attempted_transmissions // len(self.power_levels)

        self.tx_power = attempted_transmissions % len(self.power_levels)

        self.attempted_transmissions = np.where(self.power_levels[self.tx_power.astype(np.int)] > 0.0,
                                                self.attempted_transmissions, np.arange(self.n_agents))

        if self.last_comms:
            self.network_buffer[np.arange(self.n_agents), self.attempted_transmissions, 6] = self.timestep

        if self.is_interference:
            # calculates interference from attempted transmissions
            transmission_indexes, response_indexes = self.interference(self.attempted_transmissions, self.tx_power)

        self.successful_transmissions = transmission_indexes

        self.update_buffers(transmission_indexes)

        if self.comm_model is "tw":
            # Two-Way Communications can be modeled as a sequence of a push and a response
            self.timestep = self.timestep + TIMESTEP
            # my information is updated
            self.network_buffer[:, :, 0] += np.eye(self.n_agents) * TIMESTEP

            self.update_buffers(response_indexes, push=False)

        if not self.network_connected:
            self.is_network_connected()

        if self.timestep / TIMESTEP % 2 == 1:
            reward = 0
        else:
            reward = - self.instant_cost() / self.episode_length

        return self.get_relative_network_buffer_as_dict(), reward, self.timestep >= self.episode_length, {}

    def get_relative_network_buffer_as_dict(self):
        """
        Compute local node observations.
        :return: A dict representing the current routing buffers.
        """
        # timesteps and positions won't be relative within env, but need to be when passed out
        self.relative_buffer[:] = self.network_buffer
        self.relative_buffer[:, :, 0] -= self.timestep
        self.relative_buffer[:, :, 0] /= self.episode_length

        if self.last_comms:
            self.relative_buffer[:, :, 6] -= self.timestep
            self.relative_buffer[:, :, 6] /= self.episode_length

        # fills rows of a nxn matrix, subtract that from relative_network_buffer
        self.relative_buffer[:, :, 2:4] -= self.x[:, 0:2].reshape(self.n_agents, 1, 2)
        self.relative_buffer[:, :, 2:4] /= self.distance_scale

        if self.mobile_agents:
            self.relative_buffer[:, :, 4:6] -= self.x[:, 2:4].reshape(self.n_agents, 1, 2)
            self.relative_buffer[:, :, 4:6] /= self.distance_scale

        # align to the observation space and then pass that input out MAKE SURE THESE ARE INCREMENTED
        return self.map_to_observation_space(self.relative_buffer)

    def map_to_observation_space(self, network_buffer):
        """
        Compute local buffers as a Dict of representing a graph.
        :return: A dict representing the current routing buffers.
        """
        no_edge = np.not_equal(network_buffer[:, :, 1], -1)
        senders = np.where(no_edge, self.n_agents * np.arange(self.n_agents)[:, np.newaxis] + network_buffer[:, :, 1],
                           -1)
        receivers = np.where(no_edge, np.reshape(np.arange(self.n_nodes), (self.n_agents, self.n_agents)), -1)

        # TODO add distances between nodes as edge features
        step = np.reshape([self.timestep], (1, 1))
        senders = np.reshape(senders.flatten(), (-1, 1))
        receivers = np.reshape(receivers.flatten(), (-1, 1))
        nodes = np.reshape(network_buffer, (self.n_nodes, -1))
        nodes[:, 1] = 0  # zero out the neighbor node index

        data_dict = {
            "n_node": self.n_nodes,
            "senders": senders,
            "receivers": receivers,
            "edges": self.edge_features,
            "nodes": nodes,
            "globals": step
        }
        return data_dict

    def algebraic_connectivity(self, adjacency_matrix):
        graph_laplacian = np.diag(np.sum(adjacency_matrix, axis=1)) - adjacency_matrix
        v, _ = np.linalg.eigh(graph_laplacian)
        return v[1]

    def reset(self):
        if self.initial_formation is "Grid":
            x, y = self.compute_grid_with_bias(1, 1, self.n_agents)
            perm = np.random.permutation(self.n_agents)
            self.x[:, 0] = x[perm]
            self.x[:, 1] = y[perm]
        elif self.initial_formation is "Clusters":
            n_clusters = int(np.sqrt(self.n_agents))
            cluster_offset = self.r_max / (n_clusters * 1.5)
            cent_x, cent_y = self.compute_grid_with_bias(1.5, 1.5, n_clusters, additional_offset=cluster_offset)
            max_agents_per_cluster = int(np.ceil(self.n_agents / n_clusters))

            agent_cluster_assignment_x = np.reshape(np.tile(cent_x, max_agents_per_cluster).T,
                                                    (max_agents_per_cluster * n_clusters))[:self.n_agents]
            agent_cluster_assignment_y = np.reshape(np.tile(cent_y, max_agents_per_cluster).T,
                                                    (max_agents_per_cluster * n_clusters))[:self.n_agents]
            perm = np.random.permutation(self.n_agents)
            self.x[:, 0] = agent_cluster_assignment_x[perm] + np.random.uniform(-cluster_offset, cluster_offset,
                                                                                size=(self.n_agents,))
            self.x[:, 1] = agent_cluster_assignment_y[perm] + np.random.uniform(-cluster_offset, cluster_offset,
                                                                                size=(self.n_agents,))
        else:
            alg_connect = 0.0
            while np.around(alg_connect, 10) == 0.0:
                self.x[:, 0:2] = np.random.uniform(-self.r_max, self.r_max, size=(self.n_agents, 2))
                dist = self.compute_distances()
                np.fill_diagonal(dist, 0.0)
                dist = (dist <= self.fraction_of_rmax[0] * self.distance_scale * 2 * np.sqrt(2)).astype(np.float)
                alg_connect = self.algebraic_connectivity(dist)

        self.mst_action = None
        self.network_connected = False
        self.timestep = 0

        self.network_buffer = np.zeros((self.n_agents, self.n_agents, self.n_features))
        if self.known_initial_positions:
            self.network_buffer[:, :, 2] = self.x[:, 0]
            self.network_buffer[:, :, 3] = self.x[:, 1]
        else:
            self.network_buffer[:, :, 2] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                    self.x[:, 0].reshape(self.n_agents, 1),
                                                    self.network_buffer[:, :, 2])
            self.network_buffer[:, :, 3] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                    self.x[:, 1].reshape(self.n_agents, 1),
                                                    self.network_buffer[:, :, 3])

        # motivates agents to get information in the first time step
        self.network_buffer[:, :, 0] = np.where(np.eye(self.n_agents, dtype=np.bool), 0, self.penalty)
        self.network_buffer[:, :, 1] = -1  # no parent references yet

        self.old_buffer[:] = self.network_buffer
        self.relative_buffer[:] = self.network_buffer

        if self.fig != None:
            plt.close(self.fig)
        self.fig = None

        self.tx_power = None
        self.eavesdroppers = None
        self.eavesdroppers_response = None

        return self.get_relative_network_buffer_as_dict()

    def render(self, mode='human', save_plots=False, controller="Random"):
        """
        Render the environment with agents as points in 2D space
        """
        if mode == 'human':
            if self.fig == None:
                plt.ion()
                self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
                self.ax1.set_aspect('equal')
                self.ax2.set_aspect('equal')

                self.ax1.set_ylim(-1.0 * self.render_radius - 0.075, 1.0 * self.render_radius + 0.075)
                self.ax1.set_xlim(-1.0 * self.render_radius - 0.075, 1.0 * self.render_radius + 0.075)
                self.ax2.set_ylim(-1.0 * self.render_radius - 0.075, 1.0 * self.render_radius + 0.075)
                self.ax2.set_xlim(-1.0 * self.render_radius - 0.075, 1.0 * self.render_radius + 0.075)

                self.ax1.set_xticklabels(self.ax1.get_xticks(), font)
                self.ax1.set_yticklabels(self.ax1.get_yticks(), font)
                self.ax2.set_xticklabels(self.ax2.get_xticks(), font)
                self.ax2.set_yticklabels(self.ax2.get_yticks(), font)
                self.ax1.set_title('Network Interference')
                self.ax2.set_title('Agent 0\'s Buffer Tree')

                if self.flocking:
                    type_agents = "Flocking"
                elif self.mobile_agents:
                    type_agents = "Mobile"
                else:
                    type_agents = "Stationary"
                self.fig.suptitle('{0} Control Policy of {1} Agents'.format(controller, type_agents), fontsize=16)

                self.fig.subplots_adjust(top=0.9, left=0.1, right=0.9,
                                         bottom=0.12)  # create some space below the plots by increasing the bottom-value
                self._plot_text = plt.text(x=-1.21 * self.render_radius, y=-1.28 * self.render_radius, ha='center',
                                           va='center', s="", fontsize=11,
                                           bbox={'facecolor': 'lightsteelblue', 'alpha': 0.5, 'pad': 5})

                self.agent_markers1, = self.ax1.plot([], [], marker='o', color='royalblue', linestyle = '')  # Returns a tuple of line objects, thus the comma
                self.agent0_marker1, = self.ax1.plot([], [], 'go')
                self.agent_markers1_eaves, = self.ax1.plot([], [], marker='o', color='lightsteelblue', linestyle = '')  # Returns a tuple of line objects, thus the comma

                self.agent_markers2, = self.ax2.plot([], [], marker='o', color='royalblue', linestyle = '')
                self.agent0_marker2, = self.ax2.plot([], [], 'go')

                self.arrows = []
                self.failed_arrows = []
                self.paths = []
                for i in range(self.n_agents):
                    temp_arrow = self.ax1.quiver(self.x[i, 0], self.x[i, 1], 0, 0, scale=1, color='k', units='xy',
                                                 width=.015 * self.render_radius,
                                                 minshaft=.001, minlength=0)
                    self.arrows.append(temp_arrow)
                    temp_failed_arrow = self.ax1.quiver(self.x[i, 0], self.x[i, 1], 0, 0, color='r', scale=1,
                                                        units='xy',
                                                        width=.015 * self.render_radius, minshaft=.001, minlength=0)
                    self.failed_arrows.append(temp_failed_arrow)

                    temp_line, = self.ax2.plot([], [], 'k')
                    self.paths.append(temp_line)
                if self.r_max >= 1000:
                    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
                    g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.1e' % x))
                    self.ax1.xaxis.set_major_formatter(mticker.FuncFormatter(g))
                    self.ax1.yaxis.set_major_formatter(mticker.FuncFormatter(g))
                    self.ax2.xaxis.set_major_formatter(mticker.FuncFormatter(g))
                    self.ax2.yaxis.set_major_formatter(mticker.FuncFormatter(g))
            eaves_x = np.where(np.sum(self.eavesdroppers, axis=0) > 0, self.x[:, 0], 0)
            eaves_y = np.where(np.sum(self.eavesdroppers, axis=0) > 0, self.x[:, 1], 0)
            noneaves_x = np.where(np.sum(self.eavesdroppers, axis=0) == 0, self.x[:, 0], 0)
            noneaves_y = np.where(np.sum(self.eavesdroppers, axis=0) == 0, self.x[:, 1], 0)

            self.agent_markers1.set_xdata(np.ma.masked_equal(noneaves_x,0))
            self.agent_markers1.set_ydata(np.ma.masked_equal(noneaves_y,0))
            self.agent0_marker1.set_xdata(self.x[0, 0])
            self.agent0_marker1.set_ydata(self.x[0, 1])
            self.agent_markers1_eaves.set_xdata(np.ma.masked_equal(eaves_x,0))
            self.agent_markers1_eaves.set_ydata(np.ma.masked_equal(eaves_y,0))

            if self.mobile_agents or self.timestep <= 1:
                # Plot the agent locations at the start of the episode
                self.agent_markers2.set_xdata(self.x[:, 0])
                self.agent_markers2.set_ydata(self.x[:, 1])
                self.agent0_marker2.set_xdata(self.x[0, 0])
                self.agent0_marker2.set_ydata(self.x[0, 1])

            if self.mobile_agents or len(self.power_levels) > 1:
                for i in range(self.n_agents):
                    self.arrows[i].remove()
                    succ_color = self.lighten_color('k', 1 - (self.tx_power[i] / len(self.power_levels)))
                    temp_arrow = self.ax1.quiver(self.x[i, 0], self.x[i, 1], 0, 0, scale=1, color=succ_color,
                                                 units='xy',
                                                 width=.015 * self.render_radius,
                                                 minshaft=.001, minlength=0)
                    self.arrows[i] = temp_arrow

                    self.failed_arrows[i].remove()
                    fail_color = self.lighten_color('r', 1 - (self.tx_power[i] / len(self.power_levels)))
                    temp_failed_arrow = self.ax1.quiver(self.x[i, 0], self.x[i, 1], 0, 0, color=fail_color, scale=1,
                                                        units='xy',
                                                        width=.015 * self.render_radius, minshaft=.001, minlength=0)
                    self.failed_arrows[i] = temp_failed_arrow

            transmit_distances = []
            for i in range(self.n_agents):
                j = int(self.network_buffer[0, i, 1])
                if j != -1:
                    self.paths[i].set_xdata([self.x[i, 0], self.x[j, 0]])
                    self.paths[i].set_ydata([self.x[i, 1], self.x[j, 1]])

                else:
                    self.paths[i].set_xdata([])
                    self.paths[i].set_ydata([])

                if i != self.attempted_transmissions[i] and self.attempted_transmissions[i] != -1:
                    # agent chose to attempt transmission
                    transmit_distances.append(np.linalg.norm(self.x[i, 0:2] - self.x[j, 0:2]))
                    # agent chooses to communicate with j
                    j = self.attempted_transmissions[i]
                    # print(self.successful_transmissions[i][0][0])
                    if len(self.successful_transmissions[i]) > 0 and j == self.successful_transmissions[i][0]:
                        # communication linkage is successful - black
                        self.arrows[i].set_UVC(self.x[j, 0] - self.x[i, 0], self.x[j, 1] - self.x[i, 1])
                        self.failed_arrows[i].set_UVC(0, 0)

                    else:
                        # communication linkage is unsuccessful - red
                        self.arrows[i].set_UVC(0, 0)
                        self.failed_arrows[i].set_UVC(self.x[j, 0] - self.x[i, 0], self.x[j, 1] - self.x[i, 1])
                else:
                    # agent chose to not attempt transmission
                    self.arrows[i].set_UVC(0, 0)
                    self.failed_arrows[i].set_UVC(0, 0)

            cost = self.compute_current_aoi()
            if len(transmit_distances) is 0:
                self.avg_transmit_distance = 0.0
            else:
                self.avg_transmit_distance = np.mean(transmit_distances)
            mean_hops = self.find_tree_hops()
            succ_communication_percent = self.get_successful_communication_percent()
            plot_str = 'Mean AoI: {0:2.2f} | Mean Hops: {1:2.2f} | Mean TX Dist: {2:2.2f} | Comm %: {3} | Connected Network: {4}'.format(
                cost,
                mean_hops,
                self.avg_transmit_distance,
                succ_communication_percent,
                self.network_connected)
            self._plot_text.set_text(plot_str)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if save_plots:
                plt.savefig('visuals/bufferTrees/ts' + str(int(self.timestep)) + '.png')

    def get_successful_communication_percent(self):
        count_succ_comm = 0
        count_att_comm = 0
        for i in range(self.n_agents):
            if i != self.attempted_transmissions[i] and self.attempted_transmissions[i] != -1:
                # agent chose to attempt transmission
                count_att_comm += 1
                # agent chooses to communicate with j
                j = self.attempted_transmissions[i]
                if len(self.successful_transmissions[i]) > 0 and j == self.successful_transmissions[i][0]:
                    # communication linkage is successful - black
                    count_succ_comm += 1
        if count_att_comm > 0:
            succ_communication_percent = round((count_succ_comm / count_att_comm) * 100, 1)
        else:
            succ_communication_percent = 0.0
        return succ_communication_percent

    def close(self):
        pass

    def compute_distances(self):
        diff = self.x[:, 0:2].reshape((self.n_agents, 1, 2)) - self.x[:, 0:2].reshape((1, self.n_agents, 2))
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.PINF)
        return dist

    def compute_current_aoi(self):
        return - np.mean(self.network_buffer[:, :, 0] - self.timestep)

    def instant_cost(self):  # average time_delay for a piece of information plus comm distance
        if self.flocking and not self.aoi_reward:
            return np.sum(np.var(self.x[:, 2:4]/self.distance_scale, axis=0)) * 10000
        elif self.is_interference or self.aoi_reward:
            return self.compute_current_aoi()
        else:
            return self.compute_current_aoi() + self.avg_transmit_distance * 0.05

    def interference(self, attempted_transmissions, tx_power):
        # converts attempted transmissions list to an adjacency matrix
        # 0's - no communication, tx_power on indices that are communicating
        # rows are transmitting agent, columns are receiver agents
        tx_adj_mat = np.zeros((self.n_agents, self.n_agents)) + np.NINF
        tx_adj_mat[np.arange(self.n_agents), attempted_transmissions] = self.power_levels[tx_power.astype(np.int)]
        np.fill_diagonal(tx_adj_mat, np.NINF)

        successful_tx_power, self.eavesdroppers = self.calculate_sinr(tx_adj_mat)
        tx_idx = [np.nonzero(t)[0] for t in np.nan_to_num(successful_tx_power, nan=0.0, neginf=0.0)]

        if self.comm_model is "push":
            resp_idx = None
        else:
            resp_adj_mat = np.transpose(np.where(np.not_equal(successful_tx_power, np.NINF), tx_adj_mat, np.NINF))
            successful_responses, self.eavesdroppers_response = self.calculate_sinr(resp_adj_mat)
            resp_idx = [np.nonzero(t)[0] for t in np.nan_to_num(successful_responses, nan=0.0, neginf=0.0)]

        return tx_idx, resp_idx

    def calculate_sinr(self, tx_adj_mat_power_db):
        # Calculate SINR for each possible transmission
        free_space_path_loss = 10 * self.path_loss_exponent * np.log10(self.compute_distances()) + 20 * np.log10(
            self.carrier_frequency_ghz * 10 ** 9) - 147.55  # dB
        channel_gain = np.power(.1, free_space_path_loss / 10)  # this is now a unitless ratio

        # Max power each agent is transmitting with (in mW)
        power_mw = np.max(10 ** (tx_adj_mat_power_db / 10), axis=1, keepdims=True)
        numerator = channel_gain * power_mw
        denominator = self.gaussian_noise_mW + np.sum(numerator, axis=0, keepdims=True) - numerator

        with np.errstate(divide='ignore'):
            tx_sinr = 10 * np.log10(np.divide(numerator, denominator))

        # find channels where transmission would be possible
        successful_tx = np.zeros((self.n_agents, self.n_agents))
        successful_tx[tx_sinr >= self.min_SINR_dbm] = 1

        # only keep those that did try to communicate
        successful_tx_power = np.where(successful_tx, tx_adj_mat_power_db, np.NINF)
        # successful_tx_power = np.nan_to_num(successful_tx_power, nan=0.0, neginf=0.0)

        # TODO check this
        if not self.eavesdropping:
            eavesdroppers = None
        else:
            eavesdroppers = np.where(successful_tx == 1,
                                     np.where(np.transpose(power_mw) == 0,
                                              np.where(tx_adj_mat_power_db == np.NINF, 1, 0), 0), 0)

        return successful_tx_power, eavesdroppers

    # Given current positions, will return who agents should communicate with to form the Minimum Spanning Tree
    def mst_controller(self, mst_p=0.1, selective_comms=True):
        self.comm_model = "tw"
        if self.recompute_solution or self.mst_action is None:
            distances = self.compute_distances()
            G = nx.from_numpy_array(distances, create_using=nx.Graph())
            T = nx.minimum_spanning_tree(G)
            degrees = [val for (node, val) in T.degree()]

            parent_refs = np.array(self.find_parents(T, [-1] * self.n_agents, degrees))
            self.mst_action = parent_refs.astype(int) * len(self.power_levels)

        if not selective_comms:
            return self.mst_action
        else:
            tx_prob = np.random.uniform(size=(self.n_agents,))
            return np.where(tx_prob < mst_p, self.mst_action,
                            np.arange(self.n_agents) * len(self.power_levels))

    # Chooses a random action from the action space
    def random_controller(self, random_p=0.1):
        self.comm_model = "push"
        attempted_trans = np.random.choice(self.n_agents, size=(self.n_agents,))
        tx_prob = np.random.uniform(size=(self.n_agents,))
        return np.where(tx_prob < random_p, attempted_trans,
                        np.arange(self.n_agents)) * len(self.power_levels)

    # Chooses a random action from the action space
    def roundrobin_controller(self):
        self.comm_model = "tw"
        center_agent = np.argmin(np.power(self.x[:, 0], 2) + np.power(self.x[:, 1], 2))
        tx_choice = np.arange(self.n_agents) * len(self.power_levels)
        tx_idx = self.timestep % (self.n_agents - 1)
        tx_idx = tx_idx if tx_idx < center_agent else tx_idx + 1

        tx_choice[int(tx_idx)] = center_agent * len(self.power_levels)
        return tx_choice

    def find_parents(self, T, parent_ref, degrees):
        leaves = [i for i in range(self.n_agents) if degrees[i] == 1]
        assert len(leaves) != 0, "graph is not a tree"
        for j in leaves:
            parent = list(T.edges(j))[0][1]
            parent_ref[j] = parent
            T.remove_edge(j, parent)
            degrees[j] = 0
            degrees[parent] -= 1
            if (len(T.edges()) == 0):
                return parent_ref
        return self.find_parents(T, parent_ref, degrees)

    def update_buffers(self, transmission_idx, push=True):
        self.old_buffer[:] = self.network_buffer

        for i in range(self.n_agents):
            self.update_receiver_buffers(i, transmission_idx[i])

        if self.eavesdropping:

            # self.old_buffer[:] = self.network_buffer

            if push:
                eavesdroppers = self.eavesdroppers
            else:
                eavesdroppers = self.eavesdroppers_response

            for i in range(self.n_agents):
                self.update_receiver_buffers(i, np.where(eavesdroppers[i] == 1)[0])

    def update_receiver_buffers(self, tx_idx, receivers):
        if len(receivers) == 0:
            return
        self.network_buffer[receivers, :, 0:6] = np.where(
            (self.old_buffer[tx_idx, :, 0] > self.network_buffer[receivers, :, 0])[:, :, np.newaxis],
            self.old_buffer[tx_idx, :, 0:6][np.newaxis, :], self.network_buffer[receivers, :, 0:6])
        self.network_buffer[receivers, tx_idx, 1] = receivers

    def find_tree_hops(self):
        total_depth = 0
        for i in range(self.n_agents):
            local_buffer = self.network_buffer[i, :, 1]
            for j in range(self.n_agents):
                total_depth += self.find_hops(-1, j, local_buffer)
        return total_depth / (self.n_agents ** 2)

    def find_hops(self, cur_count, agent, local_buffer):
        if agent == -1:
            return cur_count
        else:
            new_agent = int(local_buffer[int(agent)])
            return self.find_hops(cur_count + 1, new_agent, local_buffer)

    def is_network_connected(self):
        if np.nonzero(self.network_buffer[:, :, 1] + 1)[0].shape[0] == (self.n_agents ** 2 - self.n_agents):
            self.network_connected = True

    def noise_floor_distance(self):
        power_mW = 10 ** (np.max(self.power_levels) / 10)
        snr_term = np.divide(power_mW, self.gaussian_noise_mW * np.power(10, self.min_SINR_dbm / 10))
        right_exp_term = (10 * np.log10(snr_term)) - (20 * np.log10(self.carrier_frequency_ghz * 10 ** 9)) + 147.55
        exponent = np.divide(1, 10 * self.path_loss_exponent) * right_exp_term
        return np.power(10, exponent)

    def find_power_levels(self):
        # returns python list of the various power levels expressed in dBm
        power_levels = []
        for i in self.fraction_of_rmax:
            power_levels.append(self.find_power_level_by_dist(
                i * self.distance_scale * 2 * np.sqrt(2)))  # Should this be r_max * 2 sqrt(2) to cover diagonal?
        return np.array(power_levels)

    def find_power_level_by_dist(self, distance):
        if distance == 0.0:
            return 0.0

        free_space_path_loss = 10 * self.path_loss_exponent * np.log10(distance) + 20 * np.log10(
            self.carrier_frequency_ghz * 10 ** 9) - 147.55  # dB
        channel_gain = np.power(10, free_space_path_loss / 10)  # this is now a unitless ratio
        gamma = np.power(10, self.min_SINR_dbm / 10)

        return 10 * np.log10(channel_gain * gamma * self.gaussian_noise_mW)

    def lighten_color(self, color, amount=0.5):
        """
        Lightens the given color by multiplying (1-luminosity) by the given amount.
        Input can be matplotlib color string, hex string, or RGB tuple.

        Examples:
        >> lighten_color('g', 0.3)
        >> lighten_color('#F034A3', 0.6)
        >> lighten_color((.3,.55,.1), 0.5)
        """

        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

    def compute_grid_with_bias(self, x_offset_coef, y_offset_coef, n_points, additional_offset=0):
        n_rows = 0
        n_cols = 0
        for i in range(int(np.sqrt(n_points)), 0, -1):
            if n_points % i is 0:
                n_rows = i
                n_cols = int(n_points / i)
                break
        x_offset = self.r_max / (n_rows * x_offset_coef)
        y_offset = self.r_max / (n_cols * y_offset_coef)
        x = np.linspace(-self.r_max + x_offset + additional_offset, self.r_max - x_offset - additional_offset,
                        num=n_rows)
        y = np.linspace(-self.r_max + y_offset + additional_offset, self.r_max - y_offset - additional_offset,
                        num=n_cols)
        xx, yy = np.meshgrid(x, y)
        coords = np.array((xx.ravel(), yy.ravel())).T
        biased_x_pos = coords[:, 0] + np.random.uniform(-x_offset, x_offset, size=(n_points,))
        biased_y_pos = coords[:, 1] + np.random.uniform(-y_offset, y_offset, size=(n_points,))
        return biased_x_pos, biased_y_pos

    @staticmethod
    def unpack_obs(obs, ob_space):
        assert tf is not None, "Function unpack_obs() is not available if Tensorflow is not imported."

        # assume flattened box
        n_nodes = (ob_space.shape[0] - 1) // (2 + N_EDGE_FEAT + N_NODE_FEAT)

        # unpack node and edge data from flattened array
        # order given by self.keys = ['nodes', 'edges', 'senders', 'receivers', 'globals']
        shapes = ((n_nodes, N_NODE_FEAT), (n_nodes, N_EDGE_FEAT), (n_nodes, 1), (n_nodes, 1), (1, 1))
        sizes = [np.prod(s) for s in shapes]
        tensors = tf.split(obs, sizes, axis=1)
        tensors = [tf.reshape(t, (-1,) + s) for (t, s) in zip(tensors, shapes)]
        nodes, edges, senders, receivers, globs = tensors

        batch_size = tf.shape(nodes)[0]
        nodes = tf.reshape(nodes, (-1, N_NODE_FEAT))
        n_node = tf.fill((batch_size,), n_nodes)  # assume n nodes is fixed

        cum_n_nodes = tf.cast(tf.reshape(tf.math.cumsum(n_node, exclusive=True), (-1, 1, 1)), dtype=tf.float32)
        senders = senders + cum_n_nodes
        receivers = receivers + cum_n_nodes

        # compute edge mask and number of edges per graph
        mask = tf.reshape(tf.not_equal(senders, -1), (batch_size, -1))  # padded edges have sender = -1
        n_edge = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1)
        mask = tf.reshape(mask, (-1,))

        # flatten edge data to be n_batch_size * n_nodes
        edges = tf.reshape(edges, (-1, N_EDGE_FEAT))
        senders = tf.reshape(senders, (-1,))
        receivers = tf.reshape(receivers, (-1,))

        # mask edges
        edges = tf.boolean_mask(edges, mask, axis=0)
        senders = tf.boolean_mask(senders, mask)
        receivers = tf.boolean_mask(receivers, mask)

        globs = tf.reshape(globs, (batch_size, 1))

        # cast all indices to int
        n_node = tf.cast(n_node, tf.int32)
        n_edge = tf.cast(n_edge, tf.int32)
        senders = tf.cast(senders, tf.int32)
        receivers = tf.cast(receivers, tf.int32)

        return batch_size, n_node, nodes, n_edge, edges, senders, receivers, globs
