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
import math

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}

EPISODE_LENGTH = 500
N_NODE_FEAT = 6
N_EDGE_FEAT = 1

save_positions = False
load_positions = False

PENALTY = -10


class MultiAgentEnv(gym.Env):

    def __init__(self):
        super(MultiAgentEnv, self).__init__()

        # default problem parameters
        self.n_agents = 20  # int(config['network_size'])
        self.r_max = 50.0  # 10.0  #  float(config['max_rad_init'])
        self.n_features = N_NODE_FEAT  # (TransTime, Parent Agent, PosX, PosY, Value (like temperature), TransmitPower)

        # initialize state matrices
        self.x = np.zeros((self.n_agents, self.n_features))

        self.action_space = spaces.MultiDiscrete([self.n_agents] * self.n_agents)
        # each agent has their own action space of a n_agent vector of weights

        self.observation_space = spaces.Dict(
            [
                # (nxn) by (features-1) we maintain parent references by edges
                ("nodes", spaces.Box(shape=(self.n_agents * self.n_agents, N_NODE_FEAT), low=-np.Inf, high=np.Inf,
                                     dtype=np.float32)),
                # upperbound, n fully connected trees (n-1) edges
                # To-Do ensure these bounds don't affect anything
                ("edges", spaces.Box(shape=(self.n_agents * self.n_agents, N_EDGE_FEAT), low=-np.Inf, high=np.Inf,
                                     dtype=np.float32)),
                # senders and receivers will each be one endpoint of an edge, and thus should be same size as edges
                ("senders", spaces.Box(shape=(self.n_agents * self.n_agents, 1), low=0, high=self.n_agents,
                                       dtype=np.float32)),
                ("receivers", spaces.Box(shape=(self.n_agents * self.n_agents, 1), low=0, high=self.n_agents,
                                         dtype=np.float32)),
                ("globals", spaces.Box(shape=(1, 1), low=0, high=EPISODE_LENGTH, dtype=np.float32)),
            ]
        )

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
        self.saved_pos = None
        self.carrier_frequency_ghz = 2.4
        self.min_SINR = -4  # 10-15 is consider unreliable, cited paper uses -4
        self.gaussian_noise_dBm = -90
        self.gaussian_noise_mW = 10 ** (self.gaussian_noise_dBm / 10)
        self.path_loss_exponent = 2
        self.tx_power = 20  # in dBm

        self.network_buffer = np.zeros((self.n_agents, self.n_agents, self.n_features))
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

        if self.flocking:
            self.render_radius = 2 * self.r_max
        else:
            self.render_radius = self.r_max

        # self.transmission_probability = .33  # Probability an agent will transmit at a given time step [0,1]

        # Push Model: At each time step, agent selects which agent they want to 'push' their buffer to
        # Two-Way Model: An agent requests/pushes their buffer to an agent, with hopes of getting their information back
        # self.comm_model = "tw"  # push or tw
        self.comm_model = "tw"  # push or tw

        self.attempted_transmissions = None
        self.successful_transmissions = None

        # Packing and unpacking information
        self.keys = ['nodes', 'edges', 'senders', 'receivers', 'globals']
        self.save_plots = False
        self.seed()

    def params_from_cfg(self, args):
        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_agents,),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)

        self.carrier_frequency_ghz = args.getfloat('carrier_frequency_ghz')
        self.min_SINR = args.getfloat('min_SINR')
        self.gaussian_noise_dBm = args.getfloat('gaussian_noise_dBm')
        self.gaussian_noise_mW = 10 ** (self.gaussian_noise_dBm / 10)
        self.path_loss_exponent = args.getfloat('path_loss_exponent')

        self.is_interference = args.getbool('is_interference')

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
        self.attempted_transmissions = attempted_transmissions
        successful_transmissions = attempted_transmissions
        # Transmit power can be incorporated later
        if self.is_interference:
            successful_transmissions, resp_trans = self.interference(
                attempted_transmissions)  # calculates interference from attempted transmissions
        self.successful_transmissions = successful_transmissions

        if not self.network_connected:
            self.is_network_connected()

        if self.comm_model is "tw":
            self.update_buffers(successful_transmissions, resp_trans)
        else:
            self.update_buffers(successful_transmissions)
        # for successful transmissions, updates the buffers of those receiving information

        self.timestep = self.timestep + 1
        return self.get_relative_network_buffer_as_dict(), - self.instant_cost() / 100.0, False, {}

    def get_relative_network_buffer_as_dict(self):
        """
        Compute local node observations.
        :return: A dict representing the current routing buffers.
        """
        # timesteps and positions won't be relative within env, but need to be when passed out
        relative_network_buffer = self.network_buffer.copy()
        relative_network_buffer[:, :, 0] = self.network_buffer[:, :, 0] - self.timestep

        # fills rows of a nxn matrix, subtract that from relative_network_buffer
        relative_network_buffer[:, :, 2:4] = self.network_buffer[:, :, 2:4] - self.x[:, 0:2].reshape(self.n_agents, 1,
                                                                                                     2)

        # align to the observation space and then pass that input out MAKE SURE THESE ARE INCREMENTED
        return self.map_to_observation_space(relative_network_buffer)

    def map_to_observation_space(self, network_buffer):
        """
        Compute local buffers as a Dict of representing a graph.
        :return: A dict representing the current routing buffers.
        """
        n = network_buffer.shape[0]
        n_nodes = n * n

        senders = []  # Indices of nodes transmitting the edges
        receivers = []  # Indices of nodes receiving the edges
        for i in range(n):
            for j in range(n):
                agent_buffer = network_buffer[i, :, :]
                # agent_buffer[j,0] should always be the timestep delay
                # agent_buffer[j,1] should always be the parent node (transmitter)
                if agent_buffer[j, 1] != -1:
                    senders.append(i * n + agent_buffer[j, 1])
                    receivers.append(i * n + j)
                else:
                    senders.append(-1)
                    receivers.append(-1)

        # TODO add distances between nodes as edge features
        edges = np.zeros(shape=(len(receivers), 1))
        nodes = np.reshape(network_buffer, (n_nodes, -1))
        nodes[:, 1] = 0  # zero out the neighbor node index

        step = np.reshape([self.timestep], (1, 1))
        senders = np.reshape(senders, (-1, 1))
        receivers = np.reshape(receivers, (-1, 1))

        data_dict = {
            "n_node": n_nodes,
            "senders": senders,
            "receivers": receivers,
            "edges": edges,
            "nodes": nodes,
            "globals": step
        }
        return data_dict

    def reset(self):

        self.network_connected = False
        self.x[:, 0:2] = np.random.uniform(-self.r_max, self.r_max, size=(self.n_agents, 2))

        # x_loc = np.reshape(self.x, (self.n_agents, 2, 1))
        # a_net = np.sum(np.square(np.transpose(x_loc, (0, 2, 1)) - np.transpose(x_loc, (2, 0, 1))), axis=2)
        # np.fill_diagonal(a_net, np.Inf)

        self.mst_action = None
        self.network_connected = False

        self.timestep = 0
        if load_positions:
            self.x = np.load("saved_positions.npy")
        elif save_positions:
            np.save("saved_positions", self.x)

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

        self.network_buffer[:, :, 0] = np.where(np.eye(self.n_agents, dtype=np.bool), 0,
                                                -10)  # motivates agents to get information in the first time step

        self.network_buffer[:, :, 1] = -1  # no parent references yet

        if self.fig != None:
            plt.close(self.fig)
        self.fig = None

        if self.is_interference:
            self.compute_distances()
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

                self.agent_markers1, = self.ax1.plot([], [], 'bo')  # Returns a tuple of line objects, thus the comma
                self.agent0_marker1, = self.ax1.plot([], [], 'go')
                self.agent_markers2, = self.ax2.plot([], [], 'bo')
                self.agent0_marker2, = self.ax2.plot([], [], 'go')

                self.arrows = []
                self.failed_arrows = []
                self.paths = []
                for i in range(self.n_agents):
                    temp_arrow = self.ax1.quiver(self.x[i, 0], self.x[i, 1], 0, 0, scale=1, units='xy',
                                                 width=.015 * self.render_radius,
                                                 minshaft=.001, minlength=0)
                    self.arrows.append(temp_arrow)
                    temp_failed_arrow = self.ax1.quiver(self.x[i, 0], self.x[i, 1], 0, 0, color='r', scale=1,
                                                        units='xy',
                                                        width=.015 * self.render_radius, minshaft=.001, minlength=0)
                    self.failed_arrows.append(temp_failed_arrow)

                    temp_line, = self.ax2.plot([], [], 'k')
                    self.paths.append(temp_line)

            if self.mobile_agents or self.timestep <= 1:
                # Plot the agent locations at the start of the episode
                self.agent_markers1.set_xdata(self.x[:, 0])
                self.agent_markers1.set_ydata(self.x[:, 1])
                self.agent0_marker1.set_xdata(self.x[0, 0])
                self.agent0_marker1.set_ydata(self.x[0, 1])
                self.agent_markers2.set_xdata(self.x[:, 0])
                self.agent_markers2.set_ydata(self.x[:, 1])
                self.agent0_marker2.set_xdata(self.x[0, 0])
                self.agent0_marker2.set_ydata(self.x[0, 1])

            if self.mobile_agents:
                for i in range(self.n_agents):
                    self.arrows[i].remove()
                    temp_arrow = self.ax1.quiver(self.x[i, 0], self.x[i, 1], 0, 0, scale=1, units='xy',
                                                 width=.015 * self.render_radius,
                                                 minshaft=.001, minlength=0)
                    self.arrows[i] = temp_arrow

                    self.failed_arrows[i].remove()
                    temp_failed_arrow = self.ax1.quiver(self.x[i, 0], self.x[i, 1], 0, 0, color='r', scale=1,
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

                    if j == self.successful_transmissions[i]:
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
                plt.savefig('visuals/bufferTrees/ts' + str(self.timestep) + '.png')

    def get_successful_communication_percent(self):
        count_succ_comm = 0
        count_att_comm = 0
        for i in range(self.n_agents):
            if i != self.attempted_transmissions[i] and self.attempted_transmissions[i] != -1:
                # agent chose to attempt transmission
                count_att_comm += 1
                # agent chooses to communicate with j
                j = self.attempted_transmissions[i]
                if j == self.successful_transmissions[i]:
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
        self.diff = self.x[:, 0:2].reshape((self.n_agents, 1, 2)) - self.x[:, 0:2].reshape((1, self.n_agents, 2))
        self.r2 = np.multiply(self.diff[:, :, 0], self.diff[:, :, 0]) + np.multiply(self.diff[:, :, 1],
                                                                                    self.diff[:, :, 1])
        np.fill_diagonal(self.r2, np.Inf)

    def compute_current_aoi(self):
        return - np.mean(self.network_buffer[:, :, 0] - self.timestep)

    def instant_cost(self):  # average time_delay for a piece of information plus comm distance
        if self.is_interference:
            return self.compute_current_aoi()
        else:
            return self.compute_current_aoi() + self.avg_transmit_distance * 0.05

    def interference(self, attempted_transmissions):

        # converts attempted transmissions list to an adjacency matrix
        # 0's - no communication, tx_power on indices that are communicating
        # rows are transmitting agent, columns are receiver agents
        trans_adj_mat = np.zeros((self.n_agents, self.n_agents)) + np.NINF
        indices = np.arange(self.n_agents)
        trans_adj_mat[indices, attempted_transmissions] = self.tx_power
        np.fill_diagonal(trans_adj_mat, np.NINF)
        if self.comm_model is "tw":
            resp_adj_mat = np.transpose(trans_adj_mat)

        successful_transmissions = self.calculate_SINR(trans_adj_mat)

        # from the adj mat, find the indices of those who succesfully communicated
        # -1's indicate unsuccessful communications
        # TODO : Change this logic for when agents can communicate with multiple
        find_who_tx = np.c_[np.zeros(self.n_agents), successful_transmissions]
        transmission_idx = (np.argmax(find_who_tx, axis=1) - 1).astype(int)

        # remove -1's and replace with respective agent's index (similar to self-communication)
        # this is necessary to be compatible with update_buffer since our action space is
        # {0,...,n-1} (doesn't include -1)
        successful_transmissions = np.where(transmission_idx != -1, transmission_idx, np.arange(self.n_agents))
        if self.comm_model is "push":
            return successful_transmissions, None
        else:
            successful_reponses = self.calculate_SINR(resp_adj_mat)

            # successful_reponses is an adj matrix of successful responses
            # we need to convert this to a python list of np arrays 
            resp_idx = []
            for i in range(self.n_agents):
                resp_idx_i = np.nonzero(successful_reponses[i, :])
                resp_idx.append(resp_idx_i)

            return successful_transmissions, resp_idx

    def calculate_SINR(self, trans_adj_mat):
        np.seterr('ignore')
        # Calculate SINR for each possible transmission
        self.compute_distances()
        power_mW = 10 ** (trans_adj_mat / 10)
        free_space_path_loss = 10 * self.path_loss_exponent * np.log10(np.sqrt(self.r2)) + 20 * np.log10(
            self.carrier_frequency_ghz * 10 ** 9) - 147.55  # dB
        channel_gain = np.power(.1, free_space_path_loss / 10)  # this is now a unitless ratio
        numerator = np.multiply(power_mW, channel_gain)  # this is in mW, numerator of the SINR

        # power each agent is transmitting with (in mW)
        trans_pow_vector = np.sum(power_mW, axis=1)

        # interference felt by each agent by all network transmissions
        interference_sum = np.matmul(channel_gain, trans_pow_vector)

        denominator = self.gaussian_noise_mW + np.expand_dims(interference_sum, axis=0) - numerator

        SINR = 10 * np.log10(np.divide(numerator, denominator))

        # find channels where transmission would be possible
        past_thresh = np.zeros((self.n_agents, self.n_agents))
        past_thresh[SINR >= self.min_SINR] = 1

        # only keep those that did try to communicate
        successful_transmissions = np.multiply(past_thresh, trans_adj_mat)
        successful_transmissions = np.nan_to_num(successful_transmissions)
        np.seterr('warn')
        return successful_transmissions

    # Given current buffer states, will pick agent with oldest AoI to communicate with
    def greedy_controller(self, selective_comms=True, transmission_probability=0.1):
        comm_choice = np.zeros((self.n_agents))
        for i in range(self.n_agents):
            my_buffer_ts = self.network_buffer[i, :, 0]
            comm_choice[i] = np.random.choice(np.flatnonzero(my_buffer_ts == my_buffer_ts.min()))

        if not selective_comms:
            return comm_choice.astype(int)
        else:
            tx_prob = np.random.uniform(size=(self.n_agents,))
            return np.where(tx_prob < transmission_probability, comm_choice.astype(int), np.arange(self.n_agents))

    # Given current positions, will return who agents should communicate with to form the Minimum Spanning Tree
    def mst_controller(self, selective_comms=True, transmission_probability=0.33):
        if self.recompute_solution or self.mst_action is None:
            self.compute_distances()
            self.dist = np.sqrt(self.r2)
            G = nx.from_numpy_array(self.dist, create_using=nx.Graph())
            T = nx.minimum_spanning_tree(G)
            degrees = [val for (node, val) in T.degree()]

            parent_refs = np.array(self.find_parents(T, [-1] * self.n_agents, degrees))
            self.mst_action = parent_refs.astype(int)

        if not selective_comms:
            return self.mst_action
        else:
            tx_prob = np.random.uniform(size=(self.n_agents,))
            return np.where(tx_prob < transmission_probability, self.mst_action, np.arange(self.n_agents))

    # Chooses a random action from the action space
    def random_controller(self, transmission_probability=0.1):
        attempted_trans = self.action_space.sample()
        tx_prob = np.random.uniform(size=(self.n_agents,))
        return np.where(tx_prob < transmission_probability, attempted_trans, np.arange(self.n_agents))

    # 33% MST, 33% Greedy, 33% Random
    def neopolitan_controller(self, selective_comms=True, transmission_probability=0.33):
        random_action = self.action_space.sample()
        mst_action = self.mst_controller(False)
        greedy_action = self.greedy_controller(False)

        rand_3 = np.random.randint(0, 3, size=(self.n_agents,))
        neopolitan_action = np.where(rand_3 == 0, random_action, (np.where(rand_3 == 1, mst_action, greedy_action)))
        if not selective_comms:
            return neopolitan_action
        else:
            tx_prob = np.random.uniform(size=(self.n_agents,))
            return np.where(tx_prob < transmission_probability, neopolitan_action, np.arange(self.n_agents))

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

    def update_buffers(self, transmission_idx, response_idx=None):
        assert (self.comm_model is "push" or self.comm_model is "tw")

        if self.comm_model is "push":
            self.update_buffers_push(transmission_idx)

        elif self.comm_model is "tw":
            # Two-Way Communications can be modeled as a sequence of a push and a response
            self.update_buffers_push(transmission_idx)
            self.update_buffers_push(response_idx, push=False)

        # my information is updated
        self.network_buffer[:, :, 0] += np.eye(self.n_agents)
        # TODO divide by number of transmissions per agent

    def update_buffers_push(self, transmission_idx, push=True):
        # TODO : Convert this to NumPy vector operations
        for i in range(self.n_agents):
            if push:
                agent_idxes = [transmission_idx[i]]
            else:
                agent_idxes = transmission_idx[i][0]

            # j is the agent that will receive i's buffer
            for j in agent_idxes:
                if i != j:
                    for k in range(self.n_agents):
                        # if received info is newer than known info
                        if self.network_buffer[i, k, 0] > self.network_buffer[j, k, 0]:
                            self.network_buffer[j, k, :] = self.network_buffer[i, k, :]
                    self.network_buffer[j, i, 1] = j
                    # agents_information[j, 5] = successful_transmissions[i, j]  # TODO update transmit power

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
        power_mW = 10 ** (self.tx_power / 10)
        snr_term = np.divide(power_mW, self.gaussian_noise_mW * np.power(10, self.min_SINR / 10))
        right_exp_term = (10 * np.log10(snr_term)) - (20 * np.log10(self.carrier_frequency_ghz * 10 ** 9)) + 147.55
        exponent = np.divide(1, 10 * self.path_loss_exponent) * right_exp_term
        return np.power(10, exponent)

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