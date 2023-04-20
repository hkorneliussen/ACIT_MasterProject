"""
source: https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion/__internal__/layers/attention_block.py 
"""

#importing dependencies
import tensorflow as tf
from tensorflow import keras

#impoering custom dependencies
from layers.padded_conv2d import PaddedConv2D


#Defining a custom attention layer block, which implementa the attention mechanism - that helps the model focus on certain parts of the input
class AttentionBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        self.q = PaddedConv2D(output_dim, 1)
        self.k = PaddedConv2D(output_dim, 1)
        self.v = PaddedConv2D(output_dim, 1)
        self.proj_out = PaddedConv2D(output_dim, 1)

    #Defining the forward pass
    def call(self, inputs):
        #passing input through group normalization layer
        x = self.norm(inputs)
        #passing input thorugh the three padded conv layer q, k and v, which produce the query, key and value matrices
        q, k, v = self.q(x), self.k(x), self.v(x)

        # Compute attention
        _, h, w, c = q.shape
        q = tf.reshape(q, (-1, h * w, c))  # b, hw, c
        k = tf.transpose(k, (0, 3, 1, 2))
        k = tf.reshape(k, (-1, c, h * w))  # b, c, hw
        y = q @ k
        y = y * (c**-0.5)
        #getting the attention weights
        y = keras.activations.softmax(y)

        # Attend to values
        v = tf.transpose(v, (0, 3, 1, 2))
        v = tf.reshape(v, (-1, c, h * w))
        y = tf.transpose(y, (0, 2, 1))
        x = v @ y
        x = tf.transpose(x, (0, 2, 1))
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + inputs