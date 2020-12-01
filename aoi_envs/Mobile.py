from MultiAgent import MultiAgentEnv
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

        self.v_max = 5
        self.a_max = 30
        self.ts_length = 0.01

        self.u = None

        self.n_features = N_NODE_FEAT  # (TransTime, Parent Agent, PosX, PosY, VelX, VelY)


    def reset(self):
        x = np.zeros((self.n_agents, 4))
        length = np.random.uniform(0, self.r_max, size=(self.n_agents,))
        angle = np.pi * np.random.uniform(0, 2, size=(self.n_agents,))
        x[:, 0] = length * np.cos(angle)
        x[:, 1] = length * np.sin(angle)

        x_loc = np.reshape(x[:,0:2], (self.n_agents, 2, 1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0, 2, 1)) - np.transpose(x_loc, (2, 0, 1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)

        x[:,2:4] = np.random.uniform(.5 * -self.v_max, .5 * self.v_max, size=(self.n_agents,2))

        self.mst_action = None

        self.timestep = 0
        self.x = x
        self.u = np.zeros((self.n_agents, 2))

        self.network_buffer = np.zeros((self.n_agents, self.n_agents, self.n_features))
        # self.network_buffer[:, :, 0] = -100  # motivates agents to get information in the first time step
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
             successful_transmissions, resp_trans  = self.interference(attempted_transmissions) # calculates interference from attempted transmissions
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
        self.u = np.random.uniform(-self.a_max, self.a_max, size=(self.n_agents,2))

        self.x[:,2:4] = self.x[:,2:4] + self.u * self.ts_length
        self.x[:,2:4] = np.clip(self.x[:,2:4], -self.v_max, self.v_max)
        self.x[:,0:2] = self.x[:,0:2] + self.x[:,2:4] * self.ts_length + 0.5 * self.u * (self.ts_length ** 2)

        self.network_buffer[:, :, 4] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 2].reshape(self.n_agents, 1), self.network_buffer[:, :, 4])
        self.network_buffer[:, :, 5] = np.where(np.eye(self.n_agents, dtype=np.bool),
                                                self.x[:, 3].reshape(self.n_agents, 1), self.network_buffer[:, :, 5])

    def render(self, mode='human', controller = "Random", filepath = 'visuals/interference/'):
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
                plt.ylim(-2.0 * self.r_max, 2.0 * self.r_max)
                plt.xlim(-2.0 * self.r_max, 2.0 * self.r_max)
                self._plot_text = plt.text(x=0, y=-1.8 * self.r_max, s="", fontsize=12, ha='center',
                                           bbox={'facecolor': 'lightsteelblue', 'alpha': 0.5, 'pad': 6})
                a = gca()
                a.set_xticklabels(a.get_xticks(), font)
                a.set_yticklabels(a.get_yticks(), font)
                plt.title('Mobile Agent\'s Buffer Tree w/ ' + controller + ' Control Policy')
                self.arrows = []

                for i in range(self.n_agents):
                    temp_line, = self.ax.plot([], [], 'k')
                    self.arrows.append(temp_line)

                self.current_arrow, = self.ax.plot([], [], 'r')

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
            plot_str = 'Mean AoI: {0:2.2f} | Mean Depth: {1:2.2f} | Mean TX Dist: {2:2.2f}'.format(cost, tree_depth,
                                                                           self.avg_transmit_distance)
            self._plot_text.set_text(plot_str)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if self.save_plots:
                plt.savefig(filepath + 'ts' + str(self.timestep) + '.png')
    
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
    
    def render_interference(self, mode='human', controller = "Random", filepath = 'visuals/interference/'):
        """
        Render the interference of the environment with agents as points in 2D space
        """
        if mode == 'human':
            if self.fig == None:
                plt.ion()
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111)
                self.agent_markers, = self.ax.plot([], [], 'bo')  # Returns a tuple of line objects, thus the comma
                self.agent0_marker, = self.ax.plot([], [], 'go')

                # Make extra space for the legend
                plt.axis('equal')
                plt.ylim(-3.0 * self.r_max, 3.0 * self.r_max)
                plt.xlim(-3.0 * self.r_max, 3.0 * self.r_max)
                self._plot_text = plt.text(x=0, y=-2 * self.r_max, s="", fontsize=9, ha='center',
                                           bbox={'facecolor': 'lightsteelblue', 'alpha': 0.5, 'pad':4})
                a = gca()
                a.set_xticklabels(a.get_xticks(), font)
                a.set_yticklabels(a.get_yticks(), font)
                plt.title('Interference between Mobile Agents w/ ' + controller + ' Control Policy')
                self.arrows = []
                self.failed_arrows = []
                #plt.axis('equal')
                for i in range(self.n_agents):
                    # temp_line = self.ax.quiver(self.x[i, 0], self.x[i, 1], 0, 0, scale=1, units='xy', width = .03, minshaft = .001, minlength=0)
                    temp_line, = self.ax.plot([], [], 'k') # black
                    self.arrows.append(temp_line)
                    temp_failed_arrow, = self.ax.plot([], [], 'r') # red
                    # temp_failed_arrow = self.ax.quiver(self.x[i, 0], self.x[i, 1], 0, 0, color='r', scale=1, units='xy', width = .03, minshaft = .001, minlength=0)
                    self.failed_arrows.append(temp_failed_arrow)

            self.agent_markers.set_xdata(self.x[:, 0])
            self.agent_markers.set_ydata(self.x[:, 1])
            self.agent0_marker.set_xdata(self.x[0, 0])
            self.agent0_marker.set_ydata(self.x[0, 1])

            count_succ_comm = 0
            count_att_comm = 0
            for i in range(self.n_agents):
                if i != self.attempted_transmissions[i] and self.attempted_transmissions[i] != -1:
                    # agent chose to attempt transmission
                    count_att_comm += 1
                    
                    # agent chooses to communicate with j
                    j = self.attempted_transmissions[i]
                    k = self.successful_transmissions[i]

                    if j == self.successful_transmissions[i]:
                        # communication linkage is successful - black
                        count_succ_comm += 1
                        # self.arrows[i].set_UVC(self.x[j, 0]-self.x[i, 0], self.x[j, 1]-self.x[i, 1])
                        self.arrows[i].set_xdata([self.x[i, 0], self.x[j, 0]])
                        self.arrows[i].set_ydata([self.x[i, 1], self.x[j, 1]])
                        # self.failed_arrows[i].set_UVC(0,0)
                        self.failed_arrows[i].set_xdata([])
                        self.failed_arrows[i].set_ydata([])
                    else:
                        # communication linkage is unsuccessful - red
                        self.failed_arrows[i].set_xdata([self.x[i, 0], self.x[j, 0]])
                        self.failed_arrows[i].set_ydata([self.x[i, 1], self.x[j, 1]])
                        # self.arrows[i].set_UVC(0, 0)
                        # self.failed_arrows[i].set_UVC(self.x[j, 0]-self.x[i, 0], self.x[j, 1]-self.x[i, 1])
                        self.arrows[i].set_xdata([])
                        self.arrows[i].set_ydata([])
                else:
                    # agent chose to not attempt transmission
                    # self.arrows[i].set_UVC(0, 0)
                    self.arrows[i].set_xdata([])
                    self.arrows[i].set_ydata([])
                    self.failed_arrows[i].set_xdata([])
                    self.failed_arrows[i].set_ydata([])
                    # self.failed_arrows[i].set_UVC(0,0)

            cost = self.compute_current_aoi()
            tree_depth = self.find_tree_depth(self.network_buffer[0, :, 1])
            communication_percent = round((count_succ_comm / self.n_agents) * 100, 1)
            if count_att_comm > 0:
                succ_communication_percent = round((count_succ_comm / count_att_comm) * 100, 1)
            else:
                succ_communication_percent = 0.0
            
            plot_str = 'Mean AoI: {0:2.2f} | Mean TX Dist: {1:2.2f} | Comm %: {2} | Suc Comm %: {3}'.format(cost,
                                                                           self.avg_transmit_distance, communication_percent,
                                                                           succ_communication_percent)
            self._plot_text.set_text(plot_str)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if self.save_plots:
                plt.savefig(filepath + 'ts' + str(self.timestep) + '.png')
    