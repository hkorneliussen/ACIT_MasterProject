'''
source: https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion/__internal__/layers/padded_conv2d.py
'''

#importing dependencies
from tensorflow import keras

'''
The PaddedConv2D layer can be used in a keras model to perform 2D convolutional opreations on an input tensor with padding. 

By adding padding to the inputs, this layer can help maintain the spatial dimensions of the output and prevent information loss at the edges of the input tensor
'''

#Defining a custom keras layer called PaddedConv2d
class PaddedConv2D(keras.layers.Layer):
    #arguments: number of filters, kernel_size, padding and strides for a 2D convolution operation
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super().__init__(**kwargs)
        #creating a zeropadding2D layer with the specified padding
        self.padding2d = keras.layers.ZeroPadding2D(padding)
        #creating a conv2D layer with specified number of filters, kernel_size and strides
        self.conv2d = keras.layers.Conv2D(filters, kernel_size, strides=strides)

    #defining the forward process
    def call(self, inputs):
        #applying the padding to the inputs
        x = self.padding2d(inputs)
        #passing the result to the conv2D layer and outputs the result
        return self.conv2d(x)