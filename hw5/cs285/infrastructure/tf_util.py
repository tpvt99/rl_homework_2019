from typing import Union
from types import FunctionType
import tensorflow as tf

def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: str = 'tanh',
        output_activation: str = 'linear',
        init_method = None,
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
        if init_method is None:
            x = tf.keras.layers.Dense(name = f'{scope}dense_{i}/', units = size)(x)
        else:
            x = CustomDense(name=f'{scope}dense_{i}/', units=size, initializer=init_method)(x)
        x = tf.keras.layers.Activation(activation = activation)(x)

    if init_method is None:
        output = tf.keras.layers.Dense(name = f'{scope}output/',units = output_size)(x)
    else:
        output = CustomDense(name=f'{scope}output/', units=size, initializer=init_method)(x)
    output = tf.keras.layers.Activation(activation=output_activation)(output)
    model = tf.keras.Model(inputs = input, outputs = output)

    return model

def from_numpy(*args, **kwargs):
    return tf.constant(*args, **kwargs, dtype=tf.float32)

class CustomDense(tf.keras.layers.Layer):
    def __init__(self, name, units, initializer):
        super(CustomDense, self).__init__(name=name)
        self.units = units
        self.initializer = initializer

    def build(self, input_shape):
        w_init = self.initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units),dtype='float32'), trainable=True)
        b_init = self.initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(self.units,), dtype='float32'), trainable=True)

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.w) + self.b
