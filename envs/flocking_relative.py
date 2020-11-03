import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}

# TODO: add functions to change # of agents, comm radius, and initial velocity, and initial radius (and then reset)
# and adjust the initialization radius accordingly


class FlockingRelativeEnv(gym.Env):

    def __init__(self):

        # config_file = path.join(path.dirname(__file__), "params_flock.cfg")
        # config = configparser.ConfigParser()
        # config.read(config_file)
        # config = config['flock']


        # number states per agent
        self.nx_system = 4
        # numer of observations per agent
        self.n_features = 6
        # number of actions per agent
        self.nu = 2 


        # default problem parameters
        self.n_agents = 100  # int(config['network_size'])
        self.dt = 0.01  # #float(config['system_dt'])
        self.v_max = 5.0  #  float(config['max_vel_init'])
        self.r_max = 1.0 #10.0  #  float(config['max_rad_init'])
        #self.std_dev = 0.1  #  float(config['std_dev']) * self.dt

        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.vr = 1 / self.comm_radius2 + np.log(self.comm_radius2)
        self.v_bias = self.v_max 

        # intitialize state matrices
        self.x = None
        self.u = None
        self.mean_vel = None
        self.init_vel = None

        self.max_accel = 1
        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_agents,),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)

        self.fig = None
        self.line1 = None
        self.action_scalar = 10.0

        self.carrier_frequency_ghz = 2.4
        self.min_SINR = 5
        self.gaussian_noise_dBm = -90
        self.gaussian_noise_mW = 10**(self.gaussian_noise_dBm/10)
        self.path_loss_exponent = 2

        # (Buffer for each agent) x (Field for each other agent in an agent's buffer) x (posX, posY, velX, velY, TransTime, NumHops)
        self.buffer_features = 6
        self.network_buffer = np.zeros((self.n_agents, self.n_agents, self.buffer_features))
        self.network_buffer[:,:,4] = -1 # motivates agents to get information in the first time step
        self.timestep = 0
        self.is_interference = True

        self.seed()

    def params_from_cfg(self, args):
        self.comm_radius = args.getfloat('comm_radius')
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.vr = 1 / self.comm_radius2 + np.log(self.comm_radius2)


        self.n_agents = args.getint('n_agents')
        self.r_max = self.r_max * np.sqrt(self.n_agents)

        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_agents,),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)

        self.v_max = args.getfloat('v_max')
        self.v_bias = self.v_max
        self.dt = args.getfloat('dt')

        self.carrier_frequency_ghz = args.getfloat('carrier_frequency_ghz')
        self.min_SINR = args.getfloat('min_SINR')
        self.gaussian_noise_dBm = args.getfloat('gaussian_noise_dBm')
        self.gaussian_noise_mW = 10**(self.gaussian_noise_dBm/10)
        self.path_loss_exponent = args.getfloat('path_loss_exponent')


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, attempted_transmissions):
        move() # using current acceleration/velocity/position, moves the agent to their current position

        successful_tranmissions = attempted_transmission
        if self.is_interference:
            successful_tranmissions  = interference(attempted_transmissions) # calculates interference from attempted transmissions

        update_buffers(successful_tranmissions) # for successful transmissions, updates the buffers of those receiving information

        compute_accelerations() # uses buffers to decide new acceleration values

        self.timestep = self.timestep + 1

        return (self.state_values, self.state_network), self.instant_cost(), False, {}


    def move():
        # x position
        self.x[:, 0] = self.x[:, 0] + self.x[:, 2] * self.dt + self.u[:, 0] * self.dt * self.dt * 0.5
        # y position
        self.x[:, 1] = self.x[:, 1] + self.x[:, 3] * self.dt + self.u[:, 1] * self.dt * self.dt * 0.5
        # x velocity
        self.x[:, 2] = self.x[:, 2] + self.u[:, 0] * self.dt
        # y velocity
        self.x[:, 3] = self.x[:, 3] + self.u[:, 1] * self.dt


    def compute_helpers(self):

        self.diff = self.x.reshape((self.n_agents, 1, self.nx_system)) - self.x.reshape((1, self.n_agents, self.nx_system))
        self.r2 =  np.multiply(self.diff[:, :, 0], self.diff[:, :, 0]) + np.multiply(self.diff[:, :, 1], self.diff[:, :, 1])
        np.fill_diagonal(self.r2, np.Inf)

        self.adj_mat = (self.r2 < self.comm_radius2).astype(float)

        # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
        n_neighbors = np.reshape(np.sum(self.adj_mat, axis=1), (self.n_agents,1)) # correct - checked this
        n_neighbors[n_neighbors == 0] = 1
        self.adj_mat_mean = self.adj_mat / n_neighbors 

        self.x_features = np.dstack((self.diff[:, :, 2], np.divide(self.diff[:, :, 0], np.multiply(self.r2, self.r2)), np.divide(self.diff[:, :, 0], self.r2),
                          self.diff[:, :, 3], np.divide(self.diff[:, :, 1], np.multiply(self.r2, self.r2)), np.divide(self.diff[:, :, 1], self.r2)))


        self.state_values = np.sum(self.x_features * self.adj_mat.reshape(self.n_agents, self.n_agents, 1), axis=1)
        self.state_values = self.state_values.reshape((self.n_agents, self.n_features))

        if self.mean_pooling:
            self.state_network = self.adj_mat_mean
        else:
            self.state_network = self.adj_mat

    def get_stats(self):

        stats = {}

        stats['vel_diffs'] = np.sqrt(np.sum(np.power(self.x[:, 2:4] - np.mean(self.x[:, 2:4], axis=0), 2), axis=1))

        stats['min_dists'] = np.min(np.sqrt(self.r2), axis=0)
        return stats

    def instant_cost(self):  # sum of differences in velocities
         curr_variance = -1.0 * np.sum((np.var(self.x[:, 2:4], axis=0)))
         return curr_variance
         # return curr_variance #+ self.potential(self.r2)
         # versus_initial_vel = -1.0 * np.sum(np.sum(np.square(self.x[:, 2:4] - self.mean_vel), axis=1))
         # return versus_initial_vel

         # squares = np.multiply(self.diff[:, :, 2], self.diff[:, :, 2]) + np.multiply(self.diff[:, :, 3], self.diff[:, :, 3])
         # # # return -1.0  * self.dt * (np.sum(np.sum(squares)) + self.potential(self.r2)) / self.n_agents #/ self.n_agents
         # return -1.0  * (np.sum(np.sum(squares))) / self.n_agents / self.n_agents

    def reset(self):
        x = np.zeros((self.n_agents, self.nx_system))
        degree = 0
        min_dist = 0
        min_dist_thresh = 0.1  # 0.25

        # generate an initial configuration with all agents connected,
        # and minimum distance between agents > min_dist_thresh
        while degree < 2 or min_dist < min_dist_thresh: 

            # randomly initialize the location and velocity of all agents
            length = np.sqrt(np.random.uniform(0, self.r_max, size=(self.n_agents,)))
            angle = np.pi * np.random.uniform(0, 2, size=(self.n_agents,))
            x[:, 0] = length * np.cos(angle)
            x[:, 1] = length * np.sin(angle)

            bias = np.random.uniform(low=-self.v_bias, high=self.v_bias, size=(2,))
            x[:, 2] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[0] 
            x[:, 3] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[1] 

            # compute distances between agents
            x_loc = np.reshape(x[:, 0:2], (self.n_agents,2,1))
            a_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
            np.fill_diagonal(a_net, np.Inf)

            # compute minimum distance between agents and degree of network to check if good initial configuration
            min_dist = np.sqrt(np.min(np.min(a_net)))
            a_net = a_net < self.comm_radius2
            degree = np.min(np.sum(a_net.astype(int), axis=1))

        # keep good initialization
        self.mean_vel = np.mean(x[:, 2:4], axis=0)
        self.init_vel = x[:, 2:4]
        self.x = x
        #self.a_net = self.get_connectivity(self.x)
        self.compute_helpers()
        return (self.state_values, self.state_network)

    def controller(self, centralized=None):
        """
        The controller for flocking from Turner 2003.
        Returns: the optimal action
        """

        if centralized is None:
            centralized = self.centralized

        # TODO use the helper quantities here more? 
        potentials = np.dstack((self.diff, self.potential_grad(self.diff[:, :, 0], self.r2), self.potential_grad(self.diff[:, :, 1], self.r2)))
        if not centralized:
            potentials = potentials * self.adj_mat.reshape(self.n_agents, self.n_agents, 1) 

        p_sum = np.sum(potentials, axis=1).reshape((self.n_agents, self.nx_system + 2))
        controls =  np.hstack(((-  p_sum[:, 4] - p_sum[:, 2]).reshape((-1, 1)), (- p_sum[:, 3] - p_sum[:, 5]).reshape(-1, 1)))
        controls = np.clip(controls, -10, 10)
        controls = controls / self.action_scalar
        return controls

    def potential_grad(self, pos_diff, r2):
        """
        Computes the gradient of the potential function for flocking proposed in Turner 2003.
        Args:
            pos_diff (): difference in a component of position among all agents
            r2 (): distance squared between agents
        Returns: corresponding component of the gradient of the potential
        """
        grad = -2.0 * np.divide(pos_diff, np.multiply(r2, r2)) + 2 * np.divide(pos_diff, r2)
        grad[r2 > self.comm_radius] = 0
        return grad 

    def potential(self, r2):
        p = np.reciprocal(r2) + np.log(r2)
        p[r2 > self.comm_radius2] = self.vr
        np.fill_diagonal(p, 0)
        return np.sum(np.sum(p))

    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """
        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(111)
            line1, = self.ax.plot(self.x[:, 0], self.x[:, 1],
                                  'bo')  # Returns a tuple of line objects, thus the comma
            self.ax.plot([0], [0], 'kx')
            plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
            plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            plt.title('GNN Controller')
            self.fig = fig
            self.line1 = line1

        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        pass

    def interference(attempted_transmissions):
        # network_transmission_power is an adjacency matrix, containing the power of each attempted
        # transmissions in dBm
        power_mw = 10**(network_transmission_power / 10)
        free_space_path_loss = 10*self.path_loss_exponent*np.log10(np.sqrt(self.r2)) + 20*np.log10(self.carrier_frequency_ghz*10**9)-147.55 #dB
        channel_gain = np.power(.1,(free_space_path_loss)/10) # this is now a unitless ratio
        numerator = np.multiply(power_mW, channel_gain) # this is in mW, numerator of the SINR 
        interference_sum = np.sum(numerator,axis=1)
        denominator = this.gaussian_noise_mW + np.expand_dims(interference_sum, axis=1) - numerator
        np.seterr(divide = 'ignore') 
        SINR = 10*np.log10(np.divide(numerator,denominator))
        np.seterr(divide = 'warn') 
        successful_tranmissions = np.zeros((self.n_agents,self.n_agents))
        successful_tranmissions[SINR >= self.min_SINR] = 1
        return successful_tranmissions

    def update_buffers(successful_tranmissions):
        # Given successful transmissions, update the buffers of those agents that need it
        # rows = transmitting agent
        # columns = agent being requested information from

        # TO-DO : Convert this to NumPy vector operations
        for i in range(self.n_agents):
            agents_information = self.network_buffer[i,:,:]
            for j in range(self.n_agents):
                if successful_tranmissions[i,j] != 0:
                    requested_information = self.network_buffer[j,:,:]
                    for k in range(self.n_agents): 
                        if requested_information[k,4] > agents_information[k,4]:
                            agents_information[k,:] = requested_information[k,:]
                            agents_information[k,5] = requested_information[k,5] + 1 # add a hop to num hops
            self.network_buffer[i,:,:] = agents_information

        # now, update each agent's information in their buffer with current state information
        for i in range(self.n_agents):
            self.network_buffer[i,i,0:4] = self.x[i,0:4]
            self.network_buffer[i,i,4] = self.timestep















