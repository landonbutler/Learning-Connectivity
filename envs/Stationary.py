import gym
from gym import spaces, error, utils
from gym.spaces import Box
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}

N_AGENTS = 10
R_MAX = 50
N_FEATURES = 6

ACTION_SCALAR = 10

CARRIER_FREQUENCY_GHZ = 2.4
MIN_SINR = 2
GAUSSIAN_NOISE_DBM = -90
PATH_LOSS_EXPONENT = 2

SHAPE = "circle"
EPISODE_LENGTH = 500

SEED = None

class StationaryEnv(gym.Env):

    def __init__(self):
        # default problem parameters
        self.n_agents = N_AGENTS # int(config['network_size'])
        self.r_max = R_MAX #10.0  #  float(config['max_rad_init'])
        self.n_features = N_FEATURES # (TransTime, Parent Agent, PosX, PosY, Value (like temperature), TransmitPower)
        
        # intitialize state matrices
        self.x = None

        self.episode_length = EPISODE_LENGTH

        self.action_space = spaces.MultiDiscrete([self.n_agents] * self.n_agents) # each agent has their own action space of a n_agent vector of weights
        
        nodes_space = Box(shape=(self.n_agents*self.n_agents,self.n_features-1), low=-np.Inf, high=np.Inf, dtype=np.float32)

        self.observation_space = spaces.Dict(
            [
                # (nxn) by (features-1) we maintain parent references by edges
                ("nodes", nodes_space),
                # upperbound, n fully connected trees (n-1) edges
                # To-Do ensure these bounds don't affect anything
                ("edges", spaces.Box(shape=(self.n_agents*(self.n_agents-1),1), low=-np.Inf, high=np.Inf, dtype=np.float32)), 
                # senders and receivers will each be one endpoint of an edge, and thus should be same size as edges
                ("senders", Box(shape=(self.n_agents*(self.n_agents-1),1), low=0, high=self.n_agents**2, dtype=np.float32)),
                ("receivers", Box(shape=(self.n_agents*(self.n_agents-1),1), low=0, high=self.n_agents**2, dtype=np.float32)),
                ("step", Box(shape=(1, 1), low=0, high=self.episode_length, dtype=np.float32)),
            ]
        )

        self.fig = None
        self.line1 = None
        self.action_scalar = ACTION_SCALAR

        """
        self.carrier_frequency_ghz = CARRIER_FREQUENCY_GHZ
        self.min_SINR = MIN_SINR
        self.gaussian_noise_dBm = GAUSSIAN_NOISE_DBM
        self.gaussian_noise_mW = 10**(self.gaussian_noise_dBm/10)
        self.path_loss_exponent = PATH_LOSS_EXPONENT
        """


        self.network_buffer = np.zeros((self.n_agents, self.n_agents, self.n_features))
        self.network_buffer[:,:,0] = -100 # motivates agents to get information in the first time step
        self.network_buffer[:,:,1] = -1 # no parent references yet
        
        self.shape = SHAPE # square or circle, default is square
        self.timestep = 0
        
        self.is_interference = True

        self.seed(SEED)

    """
    def params_from_cfg(self, args):
        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_agents,),
                                      dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)
        self.carrier_frequency_ghz = args.getfloat('carrier_frequency_ghz')
        self.min_SINR = args.getfloat('min_SINR')
        self.gaussian_noise_dBm = args.getfloat('gaussian_noise_dBm')
        self.gaussian_noise_mW = 10**(self.gaussian_noise_dBm/10)
        self.path_loss_exponent = args.getfloat('path_loss_exponent')
        self.is_interference = args.getbool('is_interference')
    """

    def seed(self, seed=None):
        self.np_random, self.seed = seeding.np_random(seed)

    # input will be an n-vector of index of who to communicate with
    # in the future, we could create a nxn continuous action space of transmit powers,
    # we would keep the max k transmits
    def step(self, transmission_indicies):
        # Transmit power can be incorporated later, will need new input
        """
        if self.is_interference:
            successful_tranmissions  = self.interference(attempted_transmissions) # calculates interference from attempted transmissions
        """

        # for successful transmissions, updates the buffers of those receiving information
        average_dist = self.update_buffers(transmission_indicies) 

        self.timestep = self.timestep + 1

        # timesteps and positions won't be relative within env, but need to be when passed out 
        relative_network_buffer = self.network_buffer.copy()
        relative_network_buffer[:,:,0] = self.network_buffer[:,:,0] - self.timestep

        # fills rows of a nxn matrix, subtract that from relative_network_buffer
        relative_network_buffer[:,:,2:4] = self.network_buffer[:,:,2:4] - self.x[:,0:2].reshape(self.n_agents,1,2)

        # align to the observation space and then pass that input out MAKE SURE THESE ARE INCREMENTED
        obs_space = self.map_to_observation_space(relative_network_buffer)
        return obs_space, self.instant_cost(average_dist), False, {}
    
    def reset(self):
        x = np.zeros((self.n_agents, 2))
        min_dist = 0
        min_dist_thresh = 0.2 # 0.25

        while min_dist < min_dist_thresh:
            if self.shape is "circle":
                length = np.random.uniform(0, self.r_max, size=(self.n_agents,))
                angle = np.pi * np.random.uniform(0, 2, size=(self.n_agents,))
                x[:, 0] = length * np.cos(angle)
                x[:, 1] = length * np.sin(angle)
            else:
                # rectangular initialization
                x = np.random.uniform(-self.r_max, self.r_max, (self.n_agents, 2))
            x_loc = np.reshape(x, (self.n_agents,2,1))
            a_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
            np.fill_diagonal(a_net, np.Inf)

            # compute minimum distance between agents and degree of network to check if good initial configuration
            min_dist = np.sqrt(np.min(a_net))

        self.x = x

        # test this
        self.network_buffer[:,:,2] = np.where(np.eye(self.n_agents, dtype=np.bool), self.x[:,0].reshape(self.n_agents,1), self.network_buffer[:,:,2])
        self.network_buffer[:,:,3] = np.where(np.eye(self.n_agents, dtype=np.bool), self.x[:,1].reshape(self.n_agents,1), self.network_buffer[:,:,3])
        self.network_buffer[:,:,0] = np.where(np.eye(self.n_agents, dtype=np.bool), 0, self.network_buffer[:,:,0])
        
        """
        for i in range(self.n_agents):
            self.network_buffer[i,i,2] = self.x[i,0] # an agent's x position is stored in its respective buffer
            self.network_buffer[i,i,3] = self.x[i,1] # y position
            self.network_buffer[i,i,0] = 0 # I know my location at timestep 0
        """
        self.compute_distances()


    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """
        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(111)
            line1, = self.ax.plot(self.x[:, 0], self.x[:, 1], 'bo')  # Returns a tuple of line objects, thus the comma
            self.ax.plot([0], [0], 'kx')
            plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
            plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            plt.title('Stationary Agent Positions')
            self.fig = fig
            self.line1 = line1

        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        pass

    def compute_distances(self):
        self.diff = self.x.reshape((self.n_agents, 1, 2)) - self.x.reshape((1, self.n_agents, 2))
        self.r2 =  np.multiply(self.diff[:, :, 0], self.diff[:, :, 0]) + np.multiply(self.diff[:, :, 1], self.diff[:, :, 1])
        np.fill_diagonal(self.r2, np.Inf)

    def get_stats(self):
        stats = {}
        stats['average_time_delay'] = instant_cost()
        return stats

    def instant_cost(self, ave_dist):  # average time_delay for a piece of information plus comm distance
        return - np.mean(self.network_buffer[:,:,0] - self.timestep) + ave_dist
    

    # Will possibly be used at a later date
    # def interference(self, attempted_transmissions):
        # network_transmission_power is an adjacency matrix, containing the power of each attempted
        # transmissions in dBm
        # power_mW = 10**(attempted_transmissions / 10)
        # free_space_path_loss = 10*self.path_loss_exponent*np.log10(np.sqrt(self.r2)) + 20*np.log10(self.carrier_frequency_ghz*10**9)-147.55 #dB
        # channel_gain = np.power(.1,(free_space_path_loss)/10) # this is now a unitless ratio
        # numerator = np.multiply(power_mW, channel_gain) # this is in mW, numerator of the SINR 
        # interference_sum = np.sum(numerator,axis=1)
        # denominator = self.gaussian_noise_mW + np.expand_dims(interference_sum, axis=1) - numerator
        # np.seterr(divide = 'ignore') 
        # SINR = 10*np.log10(np.divide(numerator,denominator))
        # np.seterr(divide = 'warn') 
        # successful_tranmissions = np.zeros((self.n_agents,self.n_agents))
        # successful_tranmissions[SINR >= self.min_SINR] = 1
        # return np.multiply(successful_tranmissions, attempted_transmissions)

    # just take indices, not array
    def update_buffers(self, transmissions_indices):
        # Given transmissions_indices, update the buffers of those agents that need it

        total_dist = 0
        # TO-DO : Convert this to NumPy vector operations
        new_network_buffer = np.zeros((self.n_agents, self.n_agents, self.n_features))
        for i in range(self.n_agents):
            agents_information = self.network_buffer[i,:,:].copy()
            target = transmissions_indices[i]
            assert(target > 0 and target < self.n_agents, "index to transmit to is OOB")
            if i != target:
                requested_information = self.network_buffer[target,:,:]
                for j in range(self.n_agents): 
                    if requested_information[k,0] > agents_information[k,0]:
                        agents_information[k,:] = requested_information[k,:]  
                agents_information[j,1] = i
                agents_information[j,5] = successful_tranmissions[i,j]
                total_dist += np.sqrt((agents_information[i,i,2]-requested_information[target,target,2])**2 
                                      + (agents_information[i,i,3]-requested_information[target,target,3])**2)
            new_network_buffer[i,:,:] = agents_information
        self.network_buffer = new_network_buffer

        # my information is updated
        self.network_buffer[:,:,0] += np.eye(self.n_agents)
        
        """
        for i in range(self.n_agents):
            self.network_buffer[i,i,0] = self.timestep + 1
        """

        return total_dist / self.n_agents

    def map_to_observation_space(self, relative_network_buffer):
        n = self.n_agents

        edges = [] # We don't have any edge features, fill with 0
        senders = []  # Indices of nodes transmitting the edges
        receivers = []  # Indices of nodes receiving the edges
        for i in range(n):
            for j in range(n):
                agent_buffer = network_buffer[i,:,:]
                # agent_buffer[j,0] should always be the timestep delay
                # agent_buffer[j,1] should always be the parent node (transmitter)
                if i != j:
                    if agent_buffer[j,1] != -1:
                        sender = i * n + agent_buffer[j,1]
                        receiver = i * n + j
                        edges.append(0)
                        senders.append(sender)
                        receivers.append(receiver)
                    else:
                        edges.append(-1)
                        senders.append(-1)
                        receivers.append(-1)
        
        # delete parent references
        node_features = np.delete(relative_network_buffer,1,3) 
        node_features_flat = np.reshape(node_features, (n * n, 5))

        obs_space_dict = {
            "nodes": node_features_flat,
            "edges": TEMP,
            "senders": senders,
            "receivers": receivers,
            "step": self.timestep
        }

        return obs_space_dict