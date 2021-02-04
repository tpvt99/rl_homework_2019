from .base_critic import BaseCritic

from cs285.infrastructure import tf_util as tfu
import tensorflow as tf
import numpy as np

class BootstrappedContinuousCritic(tf.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        with tf.name_scope('critic') as scope:
            self.critic_network = tfu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
                scope=scope
            )
        self.loss = tf.keras.losses.MeanSquaredError()
        self.loss_reports = tf.keras.metrics.Mean()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def forward(self, obs):
        return self.critic_network(obs)

    def forward_np(self, obs):
        predictions = self.forward(obs)
        return predictions.numpy()

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        # TODO: Implement the pseudocode below: do the following (
        # self.num_grad_steps_per_target_update * self.num_target_updates)
        # times:
        # every self.num_grad_steps_per_target_update steps (which includes the
        # first step), recompute the target values by
        #     a) calculating V(s') by querying the critic with next_ob_no
        #     b) and computing the target values as r(s, a) + gamma * V(s')
        # every time, update this critic using the observations and targets
        #
        # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
        #       to 0) when a terminal state is reached
        # HINT: make sure to squeeze the output of the critic_network to ensure
        #       that its dimensions match the reward
        self.loss_reports.reset_states()
        for _ in range(self.num_target_updates):
            v_s_next = self.forward_np(next_ob_no)
            v_s_next = np.squeeze(v_s_next)
            target = reward_n + self.gamma * v_s_next * (1 - terminal_n)  # must be numpy else gradients go through target

            for _ in range(self.num_grad_steps_per_target_update):
                with tf.GradientTape() as tape:

                    pred = self.forward(ob_no)
                    pred = tf.squeeze(pred)

                    assert v_s_next.shape == reward_n.shape
                    assert pred.shape == target.shape

                    loss = self.loss(target, pred)
                    self.loss_reports.update_state(loss)
                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


        return self.loss_reports.result().numpy()
