from .base_critic import BaseCritic
import pdb

#from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import tf_util as tfu
import tensorflow as tf

class CustomLambdaLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initialLR: float, lr_lambda):
        self.initialLR = initialLR
        self.lr_lambda = lr_lambda

    def __call__(self, step):
        return self.initialLR * self.lr_lambda(step)

class CQLCritic(BaseCritic):

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

        self.loss = tf.keras.losses.MeanSquaredError()
        self.cql_alpha = hparams['cql_alpha']

    def dqn_loss(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        qa_t_values = self.q_net(ob_no)

        assert len(ac_na.shape) == 1, "Must be shape 1"
        masks = tf.one_hot(ac_na, self.ac_dim)
        #q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
        q_t_values = tf.reduce_sum(masks * qa_t_values, axis=-1)

        qa_tp1_values = self.q_net_target(next_ob_no)
        #next_actions = self.q_net(next_ob_no).argmax(dim=1)
        next_actions = tf.argmax(self.q_net(next_ob_no), axis=1, output_type=tf.dtypes.int32)
        assert len(next_actions.shape) == 1, "Must be shape 1"
        masks = tf.one_hot(next_actions, self.ac_dim)
        #q_tp1 = torch.gather(qa_tp1_values, 1, next_actions.unsqueeze(1)).squeeze(1)
        q_tp1 = tf.reduce_sum(masks * qa_tp1_values, axis=-1)


        target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
        #target = target.detach()
        loss = self.loss(q_t_values, target)

        return loss, qa_t_values, q_t_values


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
        #ob_no = ptu.from_numpy(ob_no)
        #ac_na = ptu.from_numpy(ac_na).to(torch.long)
        #next_ob_no = ptu.from_numpy(next_ob_no)
        #reward_n = ptu.from_numpy(reward_n)
        #terminal_n = ptu.from_numpy(terminal_n)

        loss, qa_t_values, q_t_values = self.dqn_loss(
            ob_no, ac_na, next_ob_no, reward_n, terminal_n
            )
        
        # CQL Implementation
        # TODO: Implement CQL as described in the pdf and paper
        # Hint: After calculating cql_loss, augment the loss appropriately
        cql_loss = None

        #self.optimizer.zero_grad()
        #loss.backward()
        #self.optimizer.step()

        info = {'Training Loss': loss.numpy()}

        # TODO: Uncomment these lines after implementing CQL
        # info['CQL Loss'] = ptu.to_numpy(cql_loss)
        # info['Data q-values'] = ptu.to_numpy(q_t_values).mean()
        # info['OOD q-values'] = ptu.to_numpy(q_t_logsumexp).mean()

        return info


    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.trainable_variables, self.q_net.trainable_variables
        ):
            target_param.assign(param)

    def qa_values(self, obs):
        qa_values = self.q_net(obs)
        return qa_values.numpy()
