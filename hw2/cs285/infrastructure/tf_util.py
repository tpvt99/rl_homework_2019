from typing import Union
import tensorflow as tf


Activation = Union[str, tf.Module]


_str_to_activation = {
    'relu': tf.keras.layers.Activation('relu'),
    'tanh': tf.keras.layers.Activation('tanh'),
    'leaky_relu': tf.keras.layers.LeakyReLU(),
    'sigmoid': tf.keras.layers.Activation('sigmoid'),
    'selu': tf.keras.layers.Activation('selu'),
    'softplus': tf.keras.layers.Activation('softplus'),
    'identity': tf.keras.layers.Activation('linear'),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: str = 'tanh',
        output_activation: str = 'linear',
        scope = "mlp/") -> tf.keras.Model:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (tf.keras.Model)
    """
    # if isinstance(activation, str):
    #     activation = _str_to_activation[activation]
    # if isinstance(output_activation, str):
    #     output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.

    input = tf.keras.Input(shape = (input_size,))
    x = input
    for i in range(n_layers):
        x = tf.keras.layers.Dense(name = f'{scope}dense_{i}/', units = size)(x)
        x = tf.keras.layers.Activation(activation = activation)(x)

    output = tf.keras.layers.Dense(name = f'{scope}output/',units = output_size)(x)
    output = tf.keras.layers.Activation(activation=output_activation)(output)
    model = tf.keras.Model(inputs = input, outputs = output)

    return model

class MLP(tf.keras.models.Model):
    def __init__(self,
                input_size: int,
                output_size: int,
                n_layers: int,
                size: int,
                activation: str = 'tanh',
                output_activation = None,
                 scope : str = "mlp/"):
        super(MLP, self).__init__()

        self.hidden_layers = []
        for i in range(n_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(units=size))
            self.hidden_layers.append(tf.keras.layers.Activation(activation=activation))

        self.output_layer = tf.keras.layers.Dense(units = output_size, activation=output_activation)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output
