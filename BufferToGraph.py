from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf

class BufferToGraph():

    def networkBufferGraph(network_buffer):
        n = network_buffer.shape[0]
        n_nodes = n * n

        transmitters = []  # Indices of nodes transmitting the edges
        receivers = []  # Indices of nodes receiving the edges
        for i in range(n):
            for j in range(n):
                agent_buffer = network_buffer[i,:,:]
                # agent_buffer[j,4] should always be the timestep delay
                # agent_buffer[j,5] should always be the parent node (transmitter)
                if agent_buffer[j,5] != -1:
                    transmitters.append(i * n + agent_buffer[j,5])
                    receivers.append(i * n + j)

        data_dict = {
            "n_node": n_nodes,
            "senders": transmitters,
            "receivers": receivers,
        }

        return utils_np.data_dicts_to_graphs_tuple([data_dict])