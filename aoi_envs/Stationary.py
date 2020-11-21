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

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}

EPISODE_LENGTH = 500
N_NODE_FEAT = 6
N_EDGE_FEAT = 1

save_positions = False
load_positions = False

class StationaryEnv(gym.Env):

    def __init__(self):
        super(StationaryEnv, self).__init__()
        # default problem parameters
        self.n_agents = 200  # int(config['network_size'])
        self.r_max = 2.5  # 10.0  #  float(config['max_rad_init'])
        self.n_features = N_NODE_FEAT  # (TransTime, Parent Agent, PosX, PosY, Value (like temperature), TransmitPower)

        # intitialize state matrices
        self.x = None

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
        self.min_SINR = -4
        self.gaussian_noise_dBm = -90
        self.gaussian_noise_mW = 10 ** (self.gaussian_noise_dBm / 10)
        self.path_loss_exponent = 2
        self.tx_power = 30 # in dBm

        self.network_buffer = np.zeros((self.n_agents, self.n_agents, self.n_features))
        self.network_buffer[:, :, 0] = -100  # motivates agents to get information in the first time step
        self.network_buffer[:, :, 1] = -1  # no parent references yet
        self.timestep = 0
        self.avg_transmit_distance = 0.0

        self.symmetric_comms = True
        self.is_interference = True
        self.current_agents_choice = -1

        # Packing and unpacking information
        self.keys = ['nodes', 'edges', 'senders', 'receivers', 'globals']
        self.save_plots = True
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
        successful_transmissions = attempted_transmissions

        # Transmit power can be incorporated later
        if self.is_interference:
             successful_transmissions  = self.interference(attempted_transmissions) # calculates interference from attempted transmissions

        self.current_agents_choice = attempted_transmissions[0]
        self.update_buffers(successful_transmissions)
        # for successful transmissions, updates the buffers of those receiving information

        self.timestep = self.timestep + 1
        return self.get_relative_network_buffer_as_dict(), - self.instant_cost(), False, {}

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
        x = np.zeros((self.n_agents, 2))
        length = np.random.uniform(0, self.r_max, size=(self.n_agents,))
        angle = np.pi * np.random.uniform(0, 2, size=(self.n_agents,))
        x[:, 0] = length * np.cos(angle)
        x[:, 1] = length * np.sin(angle)

        x_loc = np.reshape(x, (self.n_agents, 2, 1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0, 2, 1)) - np.transpose(x_loc, (2, 0, 1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)

        self.timestep = 0
        if load_positions:
            self.x = np.load("saved_positions.npy")
        elif save_positions:
            np.save("saved_positions", x)
            self.x = x
        else:
            self.x = x
        self.network_buffer = np.zeros((self.n_agents, self.n_agents, self.n_features))
        # self.network_buffer[:, :, 0] = -100  # motivates agents to get information in the first time step
        self.network_buffer[:, :, 1] = -1  # no parent references yet

        # TODO test this
        # If the agents were mobile, we need to add this code into the step() function too
        self.network_buffer[:, :, 2] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 0].reshape(self.n_agents, 1), self.network_buffer[:, :, 2])
        self.network_buffer[:, :, 3] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 1].reshape(self.n_agents, 1), self.network_buffer[:, :, 3])
        self.network_buffer[:, :, 0] = np.where(np.eye(self.n_agents, dtype=np.bool), 0, -100)

        if self.is_interference:
            self.compute_distances()
        return self.get_relative_network_buffer_as_dict()

    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """
        if mode == 'human':
            if self.fig == None:
                plt.ion()
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111)
                self.agent_markers, = self.ax.plot([], [], 'bo')  # Returns a tuple of line objects, thus the comma
                self.agent0_marker, = self.ax.plot([], [], 'go')

                # Make extra space for the legend
                plt.ylim(-.8 + -1.0 * self.r_max, 1.0 * self.r_max)
                plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
                self._plot_text = plt.text(x=0, y=-1.2 * self.r_max, s="", fontsize=9, ha='center',
                                           bbox={'facecolor': 'lightsteelblue', 'alpha': 0.5, 'pad': 6})
                # -0.5 * self.r_max
                a = gca()
                a.set_xticklabels(a.get_xticks(), font)
                a.set_yticklabels(a.get_yticks(), font)
                plt.title('Stationary Agent\'s Buffer Tree w/ Greedy Control Policy')
                self.arrows = []

                for i in range(self.n_agents):
                    temp_line, = self.ax.plot([], [], 'k')
                    self.arrows.append(temp_line)

                self.current_arrow, = self.ax.plot([], [], 'r')

            if self.timestep <= 1:
                # Plot the agent locations at the start of the episode
                self.agent_markers.set_xdata(self.x[:, 0])
                self.agent_markers.set_ydata(self.x[:, 1])
                self.agent0_marker.set_xdata(self.x[0, 0])
                self.agent0_marker.set_ydata(self.x[0, 1])

            for i in range(self.n_agents):
                j = int(self.network_buffer[0, i, 1])
                if j != -1:
                    self.arrows[i].set_xdata([self.x[i, 0], self.x[j, 0]])
                    self.arrows[i].set_ydata([self.x[i, 1], self.x[j, 1]])
                    if i == self.current_agents_choice:
                        self.current_arrow.set_xdata([self.x[i, 0], self.x[0, 0]])
                        self.current_arrow.set_ydata([self.x[i, 1], self.x[0, 1]])
                else:
                    self.arrows[i].set_xdata([])
                    self.arrows[i].set_ydata([])

            cost = self.compute_current_aoi()
            tree_depth = self.find_tree_depth(self.network_buffer[0, :, 1])
            plot_str = 'Mean AoI: {0:2.2f} | Mean Depth: {1:2.2f} | Mean TX Dist: {2:2.2f} | Comm %: {3}'.format(cost, tree_depth,
                                                                           self.avg_transmit_distance, self.communication_percent)
            self._plot_text.set_text(plot_str)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if self.save_plots:
                plt.savefig('visuals/bufferTrees/ts' + str(self.timestep) + '.png')

    def close(self):
        pass

    def compute_distances(self):
        self.diff = self.x.reshape((self.n_agents, 1, 2)) - self.x.reshape((1, self.n_agents, 2))
        self.r2 = np.multiply(self.diff[:, :, 0], self.diff[:, :, 0]) + np.multiply(self.diff[:, :, 1],
                                                                                    self.diff[:, :, 1])
        np.fill_diagonal(self.r2, np.Inf)

    def compute_current_aoi(self):
        return - np.mean(self.network_buffer[:, :, 0] - self.timestep)

    def instant_cost(self):  # average time_delay for a piece of information plus comm distance
        return self.compute_current_aoi() + self.avg_transmit_distance * 0.05

    def interference(self, attempted_transmissions):
        # network_transmission_power is a list of who an agent chooses to communicate with

        # converted attempted transmissions list to an adjacency matrix
        # 0's - no communication, tx_power on indices that are communicating
        trans_adj_mat = np.zeros((self.n_agents, self.n_agents))
        indices = np.arange(self.n_agents)
        trans_adj_mat[attempted_transmissions,indices] = self.tx_power
        np.fill_diagonal(trans_adj_mat, 0)

        # Calculate SINR for each possible transmission
        self.compute_distances()
        power_mW = 10**(trans_adj_mat / 10)
        free_space_path_loss = 10*self.path_loss_exponent*np.log10(np.sqrt(self.r2)) + 20*np.log10(self.carrier_frequency_ghz*10**9)-147.55 #dB
        channel_gain = np.power(.1,(free_space_path_loss)/10) # this is now a unitless ratio
        numerator = np.multiply(power_mW, channel_gain) # this is in mW, numerator of the SINR
        interference_sum = np.sum(numerator,axis=1)
        denominator = self.gaussian_noise_mW + np.expand_dims(interference_sum, axis=1) - numerator
        np.seterr(divide = 'ignore')
        SINR = 10*np.log10(np.divide(numerator,denominator))
        np.seterr(divide = 'warn')

        # find channels where transmission would be possible
        past_thresh = np.zeros((self.n_agents,self.n_agents))
        past_thresh[SINR >= self.min_SINR] = 1

        # only keep those that did try to communicate
        successful_tranmissions = np.multiply(past_thresh, trans_adj_mat)
        
        # from the adj mat, find the indices of those who succesfully communicated
        # -1's indicate unsuccessful communications
        find_who_tx = np.vstack([np.zeros(self.n_agents), successful_tranmissions])
        transmission_idx = (np.argmax(find_who_tx, axis = 0) - 1).astype(int)
        self.communication_percent = round((((transmission_idx >= 0).sum()) / self.n_agents) * 100, 1)

        # remove -1's and replace with respective agent's index (similar to self-communication)
        # this is necessary to be compatible with update_buffer since our action space is {0,...,n-1} (doesn't include -1)
        return np.where(transmission_idx != -1, transmission_idx, np.arange(self.n_agents)) 

    # Given current buffer states, will pick agent with oldest AoI to communicate with
    def greedy_controller(self):
        comm_choice = np.zeros((self.n_agents))
        for i in range(self.n_agents):
            my_buffer_ts = self.network_buffer[i,:,0]
            comm_choice[i] = np.random.choice(np.flatnonzero(my_buffer_ts == my_buffer_ts.min()))
        return comm_choice.astype(int)

    def update_buffers(self, transmission_idx):
        # TODO : Convert this to NumPy vector operations
        transmit_distance = []
        # new_network_buffer = np.zeros((self.n_agents, self.n_agents, self.n_features))
        for i in range(self.n_agents):
            # agents_information = self.network_buffer[i, :, :].copy()
            j = transmission_idx[i]
            if i != j:
                transmit_distance.append(np.linalg.norm(self.x[i, 0:2] - self.x[j, 0:2]))
                # requested_information = self.network_buffer[j, :, :]
                for k in range(self.n_agents):
                    if self.network_buffer[j, k, 0] >= self.network_buffer[i, k, 0]:
                        self.network_buffer[i, k, :] = self.network_buffer[j, k, :]
                    elif self.symmetric_comms and self.network_buffer[j, k, 0] < self.network_buffer[i, k, 0]:
                        self.network_buffer[j, k, :] = self.network_buffer[i, k, :]
                self.network_buffer[i, j, 1] = i
                if self.symmetric_comms:
                    self.network_buffer[j, i, 1] = j
                # agents_information[j, 5] = successful_transmissions[i, j]  # TODO update transmit power
            # new_network_buffer[i, :, :] = agents_information
        # self.network_buffer = new_network_buffer

        # my information is updated
        self.network_buffer[:, :, 0] += np.eye(self.n_agents)
        self.avg_transmit_distance = np.sum(np.power(transmit_distance, 3)) / self.n_agents
        # TODO divide by number of transmissions per agent

    def find_tree_depth(self, local_buffer):
        total_depth = 0
        for i in range(self.n_agents):
            total_depth += self.find_depth(0, i, local_buffer)
        return total_depth / self.n_agents

    def find_depth(self, cur_count, agent, local_buffer):
        if agent == -1:
            return cur_count
        else:
            new_agent = int(local_buffer[int(agent)])
            return self.find_depth(cur_count + 1, new_agent, local_buffer)

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
