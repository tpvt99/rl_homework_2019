from typing import Union
import tensorflow as tf


Activation = Union[str, tf.Module]


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
