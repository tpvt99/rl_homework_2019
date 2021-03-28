from cs285.infrastructure import tf_util as tfu
from .base_exploration_model import BaseExplorationModel
import tensorflow as tf

# def init_method_1(model):
#     model.weight.data.uniform_()
#     model.bias.data.uniform_()
#
# def init_method_2(model):
#     model.weight.data.normal_()
#     model.bias.data.normal_()

def init_method_1():
    return tf.random_uniform_initializer()

def init_method_2():
    return tf.random_normal_initializer()

class CustomLambdaLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initialLR: float, lr_lambda):
        self.initialLR = initialLR
        self.lr_lambda = lr_lambda

    def __call__(self, step):
        return self.initialLR * self.lr_lambda(step)


class RNDModel(tf.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # TODO: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        # WARNING: Make sure you use different types of weight 
        #          initializations for these two functions

        # HINT 1) Check out the method ptu.build_mlp
        # HINT 2) There are two weight init methods defined above

        with tf.name_scope("f"):
            self.f = tfu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.output_size,
                n_layers=self.n_layers,
                size=self.size,
                activation='tanh',
                output_activation='linear',
                init_method=init_method_1,
                scope = "f")

        with tf.name_scope("f_hat"):
            self.f_hat = tfu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.output_size,
                n_layers=self.n_layers,
                size=self.size,
                activation='tanh',
                output_activation='linear',
                init_method=init_method_2,
                scope = "f_hat")


        self.learning_rate_scheduler = CustomLambdaLRSchedule(
            initialLR=self.optimizer_spec.optim_kwargs['learning_rate'],
            lr_lambda=self.optimizer_spec.learning_rate_schedule,
        )

        self.optimizer = self.optimizer_spec.constructor(
            self.learning_rate_scheduler
        )


    def forward(self, ob_no):
        # TODO: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        prediction = self.f(ob_no)
        target = self.f_hat(ob_no)
        error = tf.reduce_sum(tf.square(prediction-target), axis=-1) # shape [batch_size,]
        return error

    def forward_np(self, ob_no):
        error = self.forward(ob_no)
        return error.numpy()

    def update(self, ob_no):
        # TODO: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        with tf.GradientTape() as tape:
            prediction = self.f_hat(ob_no)
            target = self.f(ob_no).numpy()
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction-target), axis=-1))
        gradients = tape.gradient(loss, self.f_hat.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.f_hat.trainable_variables))
        return loss.numpy()
