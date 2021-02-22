from .base_critic import BaseCritic
import tensorflow as tf

class CustomLambdaLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initialLR: float, lr_lambda):
        self.initialLR = initialLR
        self.lr_lambda = lr_lambda

    def __call__(self, step):
        return self.initialLR * self.lr_lambda(step)

class DQNCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)

        self.learning_rate_scheduler = CustomLambdaLRSchedule(
            initialLR=self.optimizer_spec.optim_kwargs['learning_rate'],
            lr_lambda=self.optimizer_spec.learning_rate_schedule,
        )
        self.optimizer = self.optimizer_spec.constructor(
            self.learning_rate_scheduler
        )
        self.loss = tf.keras.losses.Huber()  # AKA Huber loss


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
                nothing
        """
        #ob_no = ob_no
        #ac_na = ac_na
        #next_ob_no = next_ob_no
        #reward_n = ptu.from_numpy(reward_n)
        #terminal_n = ptu.from_numpy(terminal_n)

        with tf.GradientTape() as tape:
            # Get the q_values(obs, act)
            qa_t_values = self.q_net(ob_no)
            # Get the q_values(obs) with act is implicitly as ac_na
            assert len(ac_na.shape) == 1, "To be workable, this ac_na must has shape of (batch_size,)"
            index = tf.transpose(tf.stack([tf.range(ac_na.shape[0]), ac_na]), (1,0))
            q_t_values = tf.gather_nd(qa_t_values, index)

            # Another way to find q_t_values using mask
            action_masks = tf.one_hot(ac_na, self.ac_dim)
            masked_q_t = tf.reduce_sum(action_masks * qa_t_values, axis=-1)

            # TODO compute the Q-values from the target network
            qa_tp1_values = self.q_net_target(next_ob_no)

            if self.double_q:
                # You must fill this part for Q2 of the Q-learning portion of the homework.
                # In double Q-learning, the best action is selected using the Q-network that
                # is being updated, but the Q-value for this action is obtained from the
                # target Q-network. See page 5 of https://arxiv.org/pdf/1509.06461.pdf for more details.
                qa_tp1_values_of_current = self.q_net(next_ob_no)
                max_acts = tf.reduce_max(qa_tp1_values_of_current, axis=1)
                index = tf.transpose(tf.stack([tf.range(max_acts.shape[0]), max_acts]), (1, 0))
                q_tp1 = tf.gather_nd(qa_tp1_values, index)
            else:
                q_tp1 = tf.math.reduce_max(qa_tp1_values, axis=1)

            # TODO compute targets for minimizing Bellman error
            # HINT: as you saw in lecture, this would be:
                #currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
            target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
            target = target.numpy()

            assert q_t_values.shape == target.shape

            loss = self.loss(target, q_t_values)

        gradients = tape.gradient(loss, self.q_net.trainable_variables)
        gradients = [tf.clip_by_norm(gradient, self.grad_norm_clipping) for gradient in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.q_net.trainable_variables))

        return {
                'Training Loss': loss.numpy(),
        }

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.trainable_variables, self.q_net.trainable_variables
        ):
            target_param.assign(param)
        pass

    def qa_values(self, obs):
        #obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return qa_values.numpy()
