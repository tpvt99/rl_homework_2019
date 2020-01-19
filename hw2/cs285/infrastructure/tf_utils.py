import tensorflow as tf
import os
from tensorflow.keras import Model

############################################
############################################

class BUILD_MLP(Model): # PHONG
    def __init__(self, input_size, output_size, scope, n_layers, size, activation = tf.nn.tanh, output_activation=None):
        """
            Builds a feedforward neural network

            arguments:
                input_placeholder: placeholder variable for the state (batch_size, input_size)
                scope: variable scope of the network

                n_layers: number of hidden layers
                size: dimension of each hidden layer
                activation: activation of each hidden layer

                output_size: size of the output layer
                output_activation: activation of the output layer

            returns:
                output_placeholder: the result of a forward pass through the hidden layers + the output layer
        """
        super(BUILD_MLP, self).__init__()
        self.ff_layers = []
        self.n_layers = n_layers
        for i in range(n_layers):
            if i == 0:
                self.ff_layers.append(tf.keras.layers.Dense(units = size, activation = activation, input_shape = (input_size,)))
            else:
                self.ff_layers.append(tf.keras.layers.Dense(units=size, activation=activation))
        self.ff_layers.append(tf.keras.layers.Dense(units = output_size, activation = output_activation))
        self.logstd = tf.Variable(initial_value=tf.zeros(output_size), name='logstd')

    def call(self, input):
        for layer in self.ff_layers:
            input = layer(input)
        output = input + tf.math.exp(self.logstd) * tf.random.normal(tf.shape(input), 0, 1)
        return output

def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    
    # TODO: GETTHIS from HW1
    pass

############################################
############################################


def create_tf_session(use_gpu, gpu_frac=0.6, allow_gpu_growth=True, which_gpu=0):
    if use_gpu:
        # gpu options
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_frac,
            allow_growth=allow_gpu_growth)
        # TF config
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False,
            allow_soft_placement=True,
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
        # set env variable to specify which gpu to use
        os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu)
    else:
        # TF config without gpu
        config = tf.ConfigProto(device_count={'GPU': 0})

    # use config to create TF session
    sess = tf.Session(config=config)
    return sess

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)
