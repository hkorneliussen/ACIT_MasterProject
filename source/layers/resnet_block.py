'''
source: https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion/__internal__/layers/resnet_block.py
'''

#importing dependencies
from tensorflow import keras

#import custom dependencies
from layers.padded_conv2d import PaddedConv2D

#Defining a custom resnet block layer, which implements residual connections
class ResnetBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm1 = keras.layers.GroupNormalization(epsilon=1e-5)
        self.conv1 = PaddedConv2D(output_dim, 3, padding=1)
        self.norm2 = keras.layers.GroupNormalization(epsilon=1e-5)
        self.conv2 = PaddedConv2D(output_dim, 3, padding=1)

    #layer checks if the number of channels in input_shape is equal to specified output dimension
    def build(self, input_shape):
        #if not, it creates a 1x1 conv layer to perform the residual projection
        if input_shape[-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        #if number of channels match, it creates a lambda function and returns the input unchanged
        else:
            self.residual_projection = lambda x: x

    #defining the forward process
    def call(self, inputs):
        #passing the input thorugh normalization layer and first convolutional layer
        x = self.conv1(keras.activations.swish(self.norm1(inputs)))
        #output from previous layer is passed through the second group normalization layer and conv layer. 
        #the swish activtaion function is used
        x = self.conv2(keras.activations.swish(self.norm2(x)))
        #the final output is the sum of the result from previous layer and the residual projection
        return x + self.residual_projection(inputs)