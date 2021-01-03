import abc
import itertools
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

import pickle
from cs285.infrastructure import tf_util as tfu
from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure.utils import normalize

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

            # Call to initialize the params
            assert self.logits_na.trainable_variables != []
            #self.logits_na(tf.random.normal(shape=(1, self.ob_dim)))
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

            # Call to initialize the params
            assert self.mean_net.trainable_variables != []
            # self.mean_net(tf.random.normal(shape=(1, self.ob_dim)))
            self.policy_params = [*self.mean_net.trainable_variables, self.logstd]


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        if nn_baseline:
            with tf.name_scope("baseline") as scope:
                self.baseline = tfu.build_mlp(
                    input_size=self.ob_dim,
                    output_size=1,
                    n_layers=self.n_layers,
                    size=self.size,
                    scope=scope
                )
            self.baseline_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.trainable_variables, f)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: get this from hw1
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
    def forward(self, observation: tf.Tensor):
        # TODO: get this from hw1
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

class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = tf.keras.losses.MeanSquaredError()

    def update(self, observations, actions, advantages, q_values=None):
        # Not necessarily to conver to tensor type
        observations = tf.constant(observations, dtype=tf.float32)
        actions = tf.constant(actions, dtype=tf.float32)
        advantages = tf.constant(advantages, dtype=tf.float32)

        # TODO: compute the loss that should be optimized when training with policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
            # is the expectation over collected trajectories of:
            # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
            # by the `forward` method
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.policy_params)
            pi = self.forward(observations)
            logp = pi.log_prob(actions)
            loss = -tf.reduce_mean(logp * advantages)
        gradients = tape.gradient(loss, self.policy_params)
        self.optimizer.apply_gradients(zip(gradients, self.policy_params))

        if self.nn_baseline:
            with tf.GradientTape() as tape:
                ## TODO: normalize the q_values to have a mean of zero and a standard deviation of one
                ## HINT: there is a `normalize` function in `infrastructure.utils`
                targets = normalize(q_values)
                targets = tf.Tensor(targets, dtype=tf.float32)

                ## TODO: use the `forward` method of `self.baseline` to get baseline predictions
                baseline_predictions = self.baseline(observations)

                ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
                ## [ N ] versus shape [ N x 1 ]
                ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
                assert baseline_predictions.shape == targets.shape

                # TODO: compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
                # HINT: use `F.mse_loss`
                baseline_loss = tf.keras.losses.mean_squared_error(baseline_predictions, targets)

                # TODO: optimize `baseline_loss` using `self.baseline_optimizer`
                # HINT: remember to `zero_grad` first
            gradients = tape.gradient(baseline_loss, self.baseline.trainable_variables)
            self.baseline_optimizer.apply_gradients(zip(gradients, self.baseline.trainable_variables))

        train_log = {
            'Training Loss': -loss.numpy(),
        }
        return train_log

    def run_baseline_prediction(self, obs):
        """
            Helper function that converts `obs` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `obs`: np.ndarray of size [N, obs_dim]
            Output: np.ndarray of size [N]

        """
        predictions = self.baseline(obs)
        return predictions.numpy()[:, 0]

