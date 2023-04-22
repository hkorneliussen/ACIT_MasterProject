# Copyright 2023 Hanne Korneliussen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''This code is based on:
https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion/diffusion_model.py'''

#importing dependencies
import tensorflow as tf
from tensorflow import keras

#import custom dependencies
from layers.padded_conv2d import PaddedConv2D


#defining a custom neural network architecture called DiffusionModel
class DiffusionModel(keras.Model):
    def __init__(
        self, img_height, img_width, max_text_length, name=None, download_weights=True
    ):
        context = keras.layers.Input((max_text_length, 768))
        t_embed_input = keras.layers.Input((320,))
        latent = keras.layers.Input((img_height // 8, img_width // 8, 4))

        #processing t_embed_input thorugh two dense layers to obtain t_emb
        t_emb = keras.layers.Dense(1280)(t_embed_input)
        t_emb = keras.layers.Activation("swish")(t_emb)
        t_emb = keras.layers.Dense(1280)(t_emb)

        '''
        Defining a down-sampling flow, followed by a middle flow and an up-sampling flow. 
        In each flow, the output of the previous layer is processed through ResBlock layers and SpatialTransformer layers. The resblock layers contain a combination of conv layers and activation functions. The spatialtransformer layers use attention mechanisms to align feature maps with the context
        '''

        #Defining a Downsampling flow
        outputs = []
        x = PaddedConv2D(320, kernel_size=3, padding=1)(latent)
        outputs.append(x)

        
        for _ in range(2):
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(320, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(640, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(1280, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            outputs.append(x)

        # Middle flow

        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
        x = ResBlock(1280)([x, t_emb])

        # Upsampling flow

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
        x = Upsample(640)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])

        # Exit flow

        #applying a group normalization layer and an activation layer to the output of the up-sampling flow
        x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
        x = keras.layers.Activation("swish")(x)
        #the output is a 4-channel image produced by a padded convoluational layer
        output = PaddedConv2D(4, kernel_size=3, padding=1)(x)

        super().__init__([latent, t_embed_input, context], output, name=name)

        #The model can load pre-trained weights by setting download_weights to True
        
        if download_weights:
            diffusion_model_weights_fpath = "models/kcv_diffusion_model.h5"
            self.load_weights(diffusion_model_weights_fpath)
        
        '''
        if download_weights:
            diffusion_model_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5",
                file_hash="8799ff9763de13d7f30a683d653018e114ed24a6a819667da4f5ee10f9e805fe",
            )
            self.load_weights(diffusion_model_weights_fpath)
        ''' 


#Custom implementation of a Residual Block in keras
class ResBlock(keras.layers.Layer):
    #the output_dim argument defines the number of filters for the paddedconv2D layers
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        #setting up 3 flows that contains a series of layers that perform various transformations on the input
        self.entry_flow = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1),
        ]
        self.embedding_flow = [
            keras.layers.Activation("swish"),
            keras.layers.Dense(output_dim),
        ]
        self.exit_flow = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1),
        ]

    #build-method checks if the number of filters in the input tensor matches the specified output_dim..
    def build(self, input_shape):
        #if it is not, it creates a paddedconv2d layer to project the inputs to the desired number of filters
        if input_shape[0][-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    #defining the forward pass
    def call(self, inputs):
        #splitting the input argument into "inputs" and "embeddings"
        inputs, embeddings = inputs
        #setting the variable x equal to inputs
        x = inputs
        #starting a loop that iterates over the "entry_flow" list, which is a list of layers defined in the __init__ method
        for layer in self.entry_flow:
            #applying the current layer to the previous output 'x'
            x = layer(x)
        #starting a loop that iterates over the 'embedding_flow' list defined in the init method
        for layer in self.embedding_flow:
            #applying the current layer to the 'embeddings'
            embeddings = layer(embeddings)
        #adding the embeddings to x
        x = x + embeddings[:, None, None]
        #starting a loop that iterates over the exit_flow list layers, defined in the init method
        for layer in self.exit_flow:
            #applying the current layer to the previous output 'x'
            x = layer(x)
        #returining the sum of 'x' and the result of applying the residual_projection to the original inputs. 
        #the residual_projection matches the dimension of the inputs and outputs of the block, so that they can be added together
        return x + self.residual_projection(inputs)
        
        
#defining a custom layer called Spatial transformer, which implements spatial transformer attention, which is used to perform spatial manipulation of feature maps
class SpatialTransformer(keras.layers.Layer):
    def __init__(self, num_heads, head_size, fully_connected=False, **kwargs):
        super().__init__(**kwargs)
        #initializing a groupnormalization layer
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        #calculating number of channels
        channels = num_heads * head_size
        #based on the fully_connected argument, initialize either ...
        if fully_connected:
            #...a dense layer 
            self.proj1 = keras.layers.Dense(num_heads * head_size)
        else:
            #... a paddedconv2d layer
            self.proj1 = PaddedConv2D(num_heads * head_size, 1)
        #initializing basictransformer block class
        self.transformer_block = BasicTransformerBlock(channels, num_heads, head_size)
        if fully_connected:
            self.proj2 = keras.layers.Dense(channels)
        else:
            self.proj2 = PaddedConv2D(channels, 1)

    #defining the forward pass
    def call(self, inputs):
        inputs, context = inputs
        _, h, w, c = inputs.shape
        #the inputs are normalized using group normailzation
        x = self.norm(inputs)
        #the normalized inputs are transformed using a dense/paddedconv2D linear layer (depending on the value of fully_connected)
        x = self.proj1(x)
        #The output from the previous step is reshaped to a 3D tensor
        x = tf.reshape(x, (-1, h * w, c))
        #the reshaped output is processed by a basictransformerblock layer (custom layer)
        x = self.transformer_block([x, context])
        #the output from the previous step is reshaped bach to a 4D tensor 
        x = tf.reshape(x, (-1, h, w, c))
        #the output from proj2 (another dense/conv layer) is added to the original inputs, and returned 
        return self.proj2(x) + inputs
        
        
#Defining a block of a transformer architecture       
class BasicTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        #3 instances of layernormalization class are created and assigned to norm1, norm2, and norm3
        #2 instances of crosattention class are created and assigned to attn1 and attn2
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = CrossAttention(num_heads, head_size)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn2 = CrossAttention(num_heads, head_size)
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        #an instance of GEGLU is created and assigned to geglu
        self.geglu = GEGLU(dim * 4)
        self.dense = keras.layers.Dense(dim)

    #defining the forward pass
    def call(self, inputs):
        #splitting inputs into inputs and context
        inputs, context = inputs
        #x is assigned to the result of attn1 added to inputs
        x = self.attn1([self.norm1(inputs), None]) + inputs
        #x is assigned to the result of attn2 added to x
        x = self.attn2([self.norm2(x), context]) + x
        #the result of dense with geglu(norm3(x)) added to 'x' is returned
        return self.dense(self.geglu(self.norm3(x))) + x
        

#defining a crossattention layer, that computes a cross-attention mechanism between the two inputs, referred to as 'inputs' and 'context'
class CrossAttention(keras.layers.Layer):
    #initializing the layer with num_heads and head_size
    def __init__(self, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        #3 dense layers are created, which are used to project the 'inputs' and 'context' into queries (q), keys (k) and values (v). 
        self.to_q = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.to_k = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.to_v = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.scale = head_size**-0.5
        self.num_heads = num_heads
        self.head_size = head_size
        #creating a dense layer out_proj, used to project the final attention-weighted representation back to the original input size
        self.out_proj = keras.layers.Dense(num_heads * head_size)

    #defining the foward pass
    def call(self, inputs):
        inputs, context = inputs
        #if 'context' is None, 'inputs' is used as the 'context'
        context = inputs if context is None else context
        #projecting the 'inputs' and 'context' into queries (q), keys (k) and values (v) using to_q, to_k and to_v dense laayers.
        q, k, v = self.to_q(inputs), self.to_k(context), self.to_v(context)
        #reshaping q, k and v into 3D tensors with specified dimensions
        q = tf.reshape(q, (-1, inputs.shape[1], self.num_heads, self.head_size))
        k = tf.reshape(k, (-1, context.shape[1], self.num_heads, self.head_size))
        v = tf.reshape(v, (-1, context.shape[1], self.num_heads, self.head_size))

        #transposing the tensors to have the specified dimensions
        q = tf.transpose(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        k = tf.transpose(k, (0, 2, 3, 1))  # (bs, num_heads, head_size, time)
        v = tf.transpose(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)

        #finding the dot product of q and k, and passing it thorugh a softmax function to compute attention weights
        score = td_dot(q, k) * self.scale
        weights = keras.activations.softmax(score)  # (bs, num_heads, time, time)
        #the attention-weighted representation is computed by taking a dot product of the attention weights and the v-tensor
        attn = td_dot(weights, v)
        attn = tf.transpose(attn, (0, 2, 1, 3))  # (bs, time, num_heads, head_size)
        #the final attention-weighted representation is reshaped and passed through the out_proj dense layer to get the final output
        out = tf.reshape(attn, (-1, inputs.shape[1], self.num_heads * self.head_size))
        return self.out_proj(out)
        

#definign an upsample class
class Upsample(keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        #an instance of the upsampling2d layer with a factor of 2, meaning that the iamge will be upsampled by a factor of 2 in both height and witdth
        self.ups = keras.layers.UpSampling2D(2)
        #an instance of a 2D conv layer with 'channels' number of output filters and a kernel size of 3, with padding set to 1
        self.conv = PaddedConv2D(channels, 3, padding=1)

    #defining forward pass
    def call(self, inputs):
        #upsampling the input using self.ups(inputs). 
        #the result of the upsampling is passed to self.conf
        #the result of the conv operation is returned as the output of the call method
        return self.conv(self.ups(inputs))
        
'''
GEGLU = Gatet Exponential Linear Unit
used as a non-linear activation function
'''
        
#Defining a custom keras layer
class GEGLU(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        #creating a dense layer with 'output_dim*2'output units
        self.dense = keras.layers.Dense(output_dim * 2)

    #defining the forward pass, the 'inputs' argument is the data being passed thorugh the layer
    def call(self, inputs):
        #applying the defined dense layer to the input data inputs
        x = self.dense(inputs)
        #splitting up the output of the dense layer into x (the first output_dim values and gate (remaining values)
        x, gate = x[..., : self.output_dim], x[..., self.output_dim :]
        #calcullating the tanh-activation of the expression
        tanh_res = keras.activations.tanh(
            gate * 0.7978845608 * (1 + 0.044715 * (gate**2))
        )
        #returns the final output of the GEGLU layer
        return x * 0.5 * gate * (1 + tanh_res)
        
#Defining function that performs a dot product between to tensors 'a' and 'b'
def td_dot(a, b):
    aa = tf.reshape(a, (-1, a.shape[2], a.shape[3]))
    bb = tf.reshape(b, (-1, b.shape[2], b.shape[3]))
    cc = keras.backend.batch_dot(aa, bb)
    #the output is a scalar value when applied to two vectors, and a matrix when applied to two matrices
    return tf.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))
