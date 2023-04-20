"""
source: https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion/decoder.py
"""

#importing dependencies
from tensorflow import keras

#importing custom dependencies 
from layers.attention_block import AttentionBlock
from layers.padded_conv2d import PaddedConv2D
from layers.resnet_block import ResnetBlock

#Defining the deocder class
class Decoder(keras.Sequential):
    def __init__(self, img_height, img_width, name=None, download_weights=True):
        #defining layers that form the architecture of the deocder network
        super().__init__(
            [
                keras.layers.Input((img_height // 8, img_width // 8, 4)),
                keras.layers.Rescaling(1.0 / 0.18215),
                PaddedConv2D(4, 1),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512),
                AttentionBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(256),
                ResnetBlock(256),
                ResnetBlock(256),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(256, 3, padding=1),
                ResnetBlock(128),
                ResnetBlock(128),
                ResnetBlock(128),
                keras.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(3, 3, padding=1),
            ],
            name=name,
        )


        if download_weights:
            decoder_weights_fpath = 'models/kcv_decoder.h5'
            self.load_weights(decoder_weights_fpath)
            
        '''
        #downloading weights from external source
        if download_weights:
            decoder_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5",
                file_hash="ad350a65cc8bc4a80c8103367e039a3329b4231c2469a1093869a345f55b1962",
            )
            self.load_weights(decoder_weights_fpath)
        '''