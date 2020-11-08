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

class StationaryEnv(gym.Env):

    def __init__(self):
        # default problem parameters
        self.n_agents = 10 # int(config['network_size'])
        self.r_max = 50 #10.0  #  float(config['max_rad_init'])
        self.n_features = 6 # (PosX, PosY, Value (like temp), TransmitPower, TransTime, Parent Agent)
        
        # intitialize state matrices
        self.x = None
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_agents,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_agents, self.n_features),
                                            dtype=np.float32)
        self.fig = None
        self.line1 = None
        self.action_scalar = 10.0

        self.carrier_frequency_ghz = 2.4
        self.min_SINR = 2
        self.gaussian_noise_dBm = -90
        self.gaussian_noise_mW = 10**(self.gaussian_noise_dBm/10)
        self.path_loss_exponent = 2

        self.network_buffer = np.zeros((self.n_agents, self.n_agents, self.n_features))
        self.network_buffer[:,:,4] = -100 # motivates agents to get information in the first time step
        self.network_buffer[:,:,5] = -1 # no parent references yet
        
        self.shape = "circle" # square or circle, default is square
        self.timestep = 0
        
        self.is_interference = True

        self.seed()

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


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, attempted_transmissions):
        successful_tranmissions = attempted_transmissions
        if self.is_interference:
            successful_tranmissions  = self.interference(attempted_transmissions) # calculates interference from attempted transmissions

        self.update_buffers(successful_tranmissions) # for successful transmissions, updates the buffers of those receiving information

        self.timestep = self.timestep + 1

        # timesteps won't be relative within env, but need to be when passed out 
        relative_network_buffer = self.network_buffer.copy()
        relative_network_buffer[:,:,4] = self.network_buffer[:,:,4] - self.timestep
        return relative_network_buffer, self.instant_cost(), False, {}
    
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
        for i in range(self.n_agents):
            self.network_buffer[i,i,0] = self.x[i,0] # an agent's x position is stored in its respective buffer
            self.network_buffer[i,i,1] = self.x[i,1] # y position
            self.network_buffer[i,i,4] = 0 # I know my location at timestep 0
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

    def instant_cost(self):  # average time_delay for a piece of information
        return np.mean(self.network_buffer[:,:,4] - self.timestep)
    
    def interference(self, attempted_transmissions):
        # network_transmission_power is an adjacency matrix, containing the power of each attempted
        # transmissions in dBm
        power_mW = 10**(attempted_transmissions / 10)
        free_space_path_loss = 10*self.path_loss_exponent*np.log10(np.sqrt(self.r2)) + 20*np.log10(self.carrier_frequency_ghz*10**9)-147.55 #dB
        channel_gain = np.power(.1,(free_space_path_loss)/10) # this is now a unitless ratio
        numerator = np.multiply(power_mW, channel_gain) # this is in mW, numerator of the SINR 
        interference_sum = np.sum(numerator,axis=1)
        denominator = self.gaussian_noise_mW + np.expand_dims(interference_sum, axis=1) - numerator
        np.seterr(divide = 'ignore') 
        SINR = 10*np.log10(np.divide(numerator,denominator))
        np.seterr(divide = 'warn') 
        successful_tranmissions = np.zeros((self.n_agents,self.n_agents))
        successful_tranmissions[SINR >= self.min_SINR] = 1
        return np.multiply(successful_tranmissions, attempted_transmissions)

    def update_buffers(self, successful_tranmissions):
        # Given successful transmissions, update the buffers of those agents that need it
        # rows = transmitting agent
        # columns = agent being requested information from

        # TO-DO : Convert this to NumPy vector operations
        new_network_buffer = np.zeros((self.n_agents, self.n_agents, self.n_features))
        for i in range(self.n_agents):
            agents_information = self.network_buffer[i,:,:].copy()
            for j in range(self.n_agents):
                if i != j and successful_tranmissions[i,j] != 0:
                    requested_information = self.network_buffer[j,:,:]
                    for k in range(self.n_agents): 
                        if requested_information[k,4] > agents_information[k,4]:
                            agents_information[k,:] = requested_information[k,:]  
                    agents_information[j,5] = i
                    agents_information[j,3] = successful_tranmissions[i,j]
            new_network_buffer[i,:,:] = agents_information
        self.network_buffer = new_network_buffer

        # my information is updated
        for i in range(self.n_agents):
            self.network_buffer[i,i,4] = self.timestep + 1