import tensorflow as tf
from graph_nets import graphs
from stable_baselines.common.policies import ActorCriticPolicy
import rl_comm.models as models
from aoi_envs.Stationary import StationaryEnv
from gym.spaces import MultiDiscrete


class GNNPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 num_processing_steps=None, latent_size=None, n_layers=None, reducer=None, n_gnn_layers=None,
                 model_type=None):

        super(GNNPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, scale=False)

        if model_type == 'identity':
            model_module = models.AggregationNet
        elif model_type == 'nonlinear':
            model_module = models.NonLinearGraphNet

        batch_size, n_node, nodes, n_edge, edges, senders, receivers, globs = StationaryEnv.unpack_obs(
            self.processed_obs, ob_space)

        agent_graph = graphs.GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=globs,
            receivers=receivers,
            senders=senders,
            n_node=n_node,
            n_edge=n_edge)

        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("value", reuse=reuse):
                for i in range(n_gnn_layers - 1):
                    self.value_model_i = model_module(num_processing_steps=num_processing_steps,
                                                      latent_size=latent_size,
                                                      n_layers=n_layers, reducer=reducer,
                                                      node_output_size=latent_size,
                                                      name="value_model" + str(i))
                    agent_graph = self.value_model_i(agent_graph)

                # The readout GNN layer for the value function
                self.value_model = model_module(num_processing_steps=num_processing_steps,
                                                latent_size=latent_size,
                                                n_layers=n_layers, reducer=reducer,
                                                node_output_size=1, name="value_model")
                value_graph = self.value_model(agent_graph)

                # sum the outputs of robot nodes to compute value
                reshaped_nodes = tf.reshape(value_graph.nodes, (batch_size, len(ac_space.nvec)))
                self._value_fn = tf.reduce_sum(reshaped_nodes, axis=1, keepdims=True)
                self.q_value = None  # unused by PPO2

            with tf.variable_scope("policy", reuse=reuse):
                for i in range(n_gnn_layers - 1):
                    self.policy_model_i = model_module(num_processing_steps=num_processing_steps,
                                                       latent_size=latent_size,
                                                       n_layers=n_layers, reducer=reducer,
                                                       node_output_size=latent_size,
                                                       name="policy_model" + str(i))
                    agent_graph = self.policy_model_i(agent_graph)

                # The readout GNN layer for the policy
                # TODO modify policy GNN outputs and remove masking
                self.policy_model = model_module(num_processing_steps=num_processing_steps,
                                                 latent_size=latent_size,
                                                 n_layers=n_layers, reducer=reducer,
                                                 edge_output_size=1, out_init_scale=1.0,
                                                 name="policy_model")
                policy_graph = self.policy_model(agent_graph)

                edge_values = policy_graph.edges

                # keep only edges for which senders are the landmarks, receivers are robots
                sender_type = tf.cast(tf.gather(nodes[:, 0], senders), tf.bool)
                receiver_type = tf.cast(tf.gather(nodes[:, 0], receivers), tf.bool)
                mask = tf.logical_and(tf.logical_not(sender_type), receiver_type)
                masked_edges = tf.boolean_mask(edge_values, tf.reshape(mask, (-1,)), axis=0)

                if isinstance(ac_space, MultiDiscrete):
                    n_actions = tf.cast(tf.reduce_sum(ac_space.nvec), tf.int32)
                else:
                    n_actions = tf.cast(ac_space.n, tf.int32)

                self._policy = tf.reshape(masked_edges, (batch_size, n_actions))
                self._proba_distribution = self.pdtype.proba_distribution_from_flat(self._policy)

        self._setup_init()

    def get_policy_model(self):
        return self.policy_model

    def get_value_model(self):
        return self.value_model

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})

        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    @staticmethod
    def policy_param_string(p):
        """Return identifier string for policy parameter dict."""
        return 'gnnfwd'

