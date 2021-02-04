import abc
import itertools
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import numpy as np

from cs285.infrastructure import tf_util as tfu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, tf.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            with tf.name_scope("logits") as scope:
                self.logits_na = tfu.build_mlp(input_size=self.ob_dim,
                                            output_size=self.ac_dim,
                                            n_layers=self.n_layers,
                                            size=self.size,
                                            scope=scope)
            self.mean_net = None
            self.logstd = None
            self.policy_params = [*self.logits_na.trainable_variables]

        else:
            self.logits_na = None
            with tf.name_scope("mean") as scope:
                self.mean_net = tfu.build_mlp(input_size=self.ob_dim,
                                                output_size=self.ac_dim,
                                                n_layers=self.n_layers,
                                                size=self.size,
                                                scope=scope)
            self.logstd = tf.Variable(initial_value=
                                      tf.zeros(shape=self.ac_dim, dtype=tf.dtypes.float32),
                                      trainable=True, name="logstd")
            self.policy_params = [*self.mean_net.trainable_variables, self.logstd]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        if nn_baseline:
            with tf.name_scope("baseline") as scope:
                self.baseline = tfu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
                scope=scope,
            )
            self.baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.trainable_variables, f)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: get this from Piazza
        if len(obs.shape) == 1:
            obs = obs[None]
        else:
            obs = obs

        # TODO return the action that the policy prescribes
        if self.discrete:
            logits = self.logits_na(obs)
            pi = tfp.distributions.Categorical(logits=logits)
            return pi.sample().numpy()
        else:
            mean = self.mean_net(obs)
            std = tf.exp(self.logstd)
            pi = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
            acts = pi.sample()
            return acts.numpy()

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: tf.Tensor) -> tfp.distributions.Distribution:
        # TODO: get this from Piazza
        if self.discrete:
            logits = self.logits_na(observation)
            pi = tfp.distributions.Categorical(logits=logits)
            return pi
        else:
            mean = self.mean_net(observation)
            std = tf.exp(self.logstd)
            pi = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
            return pi


#####################################################
#####################################################


class MLPPolicyAC(MLPPolicy):
    def update(self, observations: np.ndarray, actions: np.ndarray, adv_n=None):
        # TODO: update the policy and return the loss

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.policy_params)
            pi = self.forward(observations)
            logp = pi.log_prob(actions)
            loss = -tf.reduce_mean(logp * adv_n)
        gradients = tape.gradient(loss, self.policy_params)
        self.optimizer.apply_gradients(zip(gradients, self.policy_params))

        return loss.numpy()
