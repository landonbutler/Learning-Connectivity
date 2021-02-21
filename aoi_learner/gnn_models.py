from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt
from graph_nets import modules, blocks

from graph_nets import graphs
from stable_baselines.a2c.utils import ortho_init
from graph_nets.blocks import unsorted_segment_max_or_zero
from graph_nets import utils_tf


class Identity(snt.AbstractModule):
    """Sonnet module implementing the identity."""
    def __init__(self, name="identity"):
        super(Identity, self).__init__(name=name)

    def _build(self, inputs):
        return tf.identity(inputs)

class AggregationNet(snt.AbstractModule):
    """
    Aggregation Net with a linear aggregation filter
    """

    def __init__(self,
                 num_processing_steps=None,
                 latent_size=None,
                 n_layers=None,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 reducer=None,
                 out_init_scale=5.0,
                 name="AggregationNet"):
        super(AggregationNet, self).__init__(name=name)

        if num_processing_steps is None:
            self._num_processing_steps = 5
        else:
            self._num_processing_steps = num_processing_steps

        if reducer is None or reducer == 'max':
            reducer = unsorted_segment_max_or_zero
        elif reducer == 'mean':
            reducer = tf.math.unsorted_segment_mean
        elif reducer == 'sum':
            reducer = tf.math.unsorted_segment_sum
        else:
            raise ValueError('Unknown reducer!')

        if latent_size is None:
            latent_size = 16

        if n_layers is None:
            n_layers = 2

        def make_mlp():
            return snt.nets.MLP([latent_size] * n_layers, activate_final=True)

        if self._num_processing_steps > 0:
            # Edge block copies the node features onto the edges.
            core_a = blocks.EdgeBlock(
                edge_model_fn=lambda: Identity(),
                use_edges=False,
                use_receiver_nodes=False,
                use_sender_nodes=True,
                use_globals=False,
                name='LinearNodeAggGCN_core_a')

            # Then, edge data is aggregated onto the node by the reducer function.
            core_b = blocks.NodeBlock(
                node_model_fn=lambda: Identity(),
                use_received_edges=True,
                use_sent_edges=False,
                use_nodes=False,
                use_globals=False,
                received_edges_reducer=reducer,
                name='LinearNodeAggGCN_core_b')

            self._cores = [core_a, core_b]

        self._encoder = modules.GraphIndependent(make_mlp, make_mlp, make_mlp, name="encoder")
        self._decoder = modules.GraphIndependent(make_mlp, make_mlp, make_mlp, name="decoder")

        inits = {'w': ortho_init(out_init_scale), 'b': tf.constant_initializer(0.0)}

        # Transforms the outputs into the appropriate shapes.
        edge_fn = None if edge_output_size is None else lambda: snt.Linear(edge_output_size, initializers=inits,
                                                                           name="edge_output")
        node_fn = None if node_output_size is None else lambda: snt.Linear(node_output_size, initializers=inits,
                                                                           name="node_output")
        global_fn = None if global_output_size is None else lambda: snt.Linear(global_output_size,
                                                                               initializers=inits,
                                                                               name="global_output")
        with self._enter_variable_scope():
            self._output_transform = modules.GraphIndependent(edge_fn, node_fn, global_fn, name="output")

    def _build(self, input_op):
        latent = self._encoder(input_op)  # latent size = 16
        output_ops = [self._decoder(latent)]  # K = 0 data
        for i in range(self._num_processing_steps):
            for c in self._cores:
                latent = c(latent)
            decoded_op = self._decoder(latent)
            output_ops.append(decoded_op)  # K = 1, 2, 3... data
        return self._output_transform(utils_tf.concat(output_ops, axis=1))  # K * 16 for every node


class NonLinearGraphNet(snt.AbstractModule):
    """
    Aggregation Net with learned aggregation filter
    """

    def __init__(self,
                 num_processing_steps=None,
                 latent_size=None,
                 n_layers=None,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 reducer=None,
                 out_init_scale=5.0,
                 name="AggregationNet"):
        super(NonLinearGraphNet, self).__init__(name=name)

        if num_processing_steps is None:
            self._num_processing_steps = 5
        else:
            self._num_processing_steps = num_processing_steps

        if reducer is None or reducer == 'max':
            reducer = unsorted_segment_max_or_zero
        elif reducer == 'mean':
            reducer = tf.math.unsorted_segment_mean
        elif reducer == 'sum':
            reducer = tf.math.unsorted_segment_sum
        else:
            raise ValueError('Unknown reducer!')

        if latent_size is None:
            latent_size = 16

        if n_layers is None:
            n_layers = 2

        def make_mlp():
            return snt.nets.MLP([latent_size] * n_layers, activate_final=False)

        if self._num_processing_steps > 0:
            # Edge model f^e(v_sender, v_receiver, e)     -   in the linear linear model, f^e = v_sender
            # Average over all the received edge features to get e'
            # Node model f^v(v, e'), but in the linear model, it was just f^v = e'
            self._core = modules.GraphNetwork(
                edge_model_fn=make_mlp,
                node_model_fn=make_mlp,
                global_model_fn=make_mlp,
                edge_block_opt={'use_globals': False},
                node_block_opt={'use_globals': False, 'use_sent_edges': False},
                name="graph_net",
                reducer=reducer
            )

        self._encoder = modules.GraphIndependent(make_mlp, make_mlp, make_mlp, name="encoder")
        self._decoder = modules.GraphIndependent(make_mlp, make_mlp, make_mlp, name="decoder")

        inits = {'w': ortho_init(out_init_scale), 'b': tf.constant_initializer(0.0)}

        # Transforms the outputs into the appropriate shapes.
        edge_fn = None if edge_output_size is None else lambda: snt.Linear(edge_output_size, initializers=inits,
                                                                           name="edge_output")
        node_fn = None if node_output_size is None else lambda: snt.Linear(node_output_size, initializers=inits,
                                                                           name="node_output")
        global_fn = None if global_output_size is None else lambda: snt.Linear(global_output_size,
                                                                               initializers=inits,
                                                                               name="global_output")
        with self._enter_variable_scope():
            self._output_transform = modules.GraphIndependent(edge_fn, node_fn, global_fn, name="output")

    def _build(self, input_op):
        latent = self._encoder(input_op)
        output_ops = [self._decoder(latent)]  # K = 0
        for i in range(self._num_processing_steps):
            latent = self._core(latent)
            decoded_op = self._decoder(latent)
            output_ops.append(decoded_op)  # K = 1, 2, 3, ...
        return self._output_transform(utils_tf.concat(output_ops, axis=1))
