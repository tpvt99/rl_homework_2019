import abc
import itertools
from typing import Any
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

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
                self.logits_na = tfu.build_mlp(
                    input_size=self.ob_dim,
                    output_size=self.ac_dim,
                    n_layers=self.n_layers,
                    size=self.size,
                    scope=scope
                )
            self.mean_net = None
            self.logstd = None
        else:
            self.logits_na = None
            with tf.name_scope("mean") as scope:
                self.mean_net = tfu.build_mlp(
                    input_size=self.ob_dim,
                    output_size=self.ac_dim,
                    n_layers=self.n_layers,
                    size=self.size,
                    scope=scope
                )
            self.logstd = tf.Variable(initial_value=tf.zeros(shape = self.ac_dim, dtype = tf.dtypes.float32),
                                      trainable=True, name = "logstd")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

    ##################################

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.trainable_variables, f)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        if self.discrete:
            logits = self.logits_na(observation)
            pi = tfp.distributions.Categorical(logits = logits)
            return pi.sample().numpy()
        else:
            mean = self.mean_net(observation)
            std = tf.exp(self.logstd)
            pi = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
            acts = mean + tf.random.normal(shape = mean.shape) * std
            return acts.numpy()

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: np.ndarray) -> Any:
        assert len(observation.shape) == 2

        # TODO return the action that the policy prescribes
        if self.discrete:
            logits = self.logits_na(observation)
            #pi = tfp.distributions.Categorical(logits=logits)
            #acts = pi.sample()
            return logits
        else:
            mean = self.mean_net(observation)
            std = tf.exp(self.logstd)
            pi = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
            acts = pi.sample()
            acts = mean + tf.random.normal(shape=mean.shape) * std
            return acts


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        if self.discrete:
            print('Using Categorical Crossentropy Loss')
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            self.loss = tf.keras.losses.MeanSquaredError()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # TODO: update the policy and return the loss
        with tf.GradientTape() as tape:
            prediction_actions = self.forward(observations)
            loss = self.loss(prediction_actions, actions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': loss.numpy(),
        }
