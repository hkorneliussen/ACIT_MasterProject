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
https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion/text_encoder.py'''

'''
Importing dependencies
'''

#importing dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.experimental import numpy as tfnp

'''
Defining the textencoder

The textencoder class is a keras.model that is used to produce an embedded representation of a text input. 
Its input tokens and positions are passed thorugh an CLIPEmbedding layer, which is used to embed the tokens and positions into a continous vector space. 
The resulting embedded representation is then passed through multiple CLIPEncoder layers' to produce the final embedded representation. 
The final representation is then passed through a LayerNormalization layer to normalize the activations. 
If download_weights is true, the weights of the model will be downloaded and loaded. 
'''

#Defining text encoder class 
class TextEncoder(keras.Model):
    #the class takes two inputs; tokens and positions
    #it outputs an embedded representation of the tokens
    def __init__(self, max_length, vocab_size=49408, name=None, download_weights=True):
        tokens = keras.layers.Input(shape=(max_length,), dtype="int32", name="tokens")
        positions = keras.layers.Input(
            shape=(max_length,), dtype="int32", name="positions"
        )
        #passing the inputs through an embedding layer
        x = CLIPEmbedding(vocab_size, 768, max_length)([tokens, positions])
        #passing it through multiple CLIPEncoder layers to produce the final embedded representation
        for _ in range(12):
            x = CLIPEncoderLayer(768, 12, activation=quick_gelu)(x)
        embedded = keras.layers.LayerNormalization(epsilon=1e-5)(x)
        super().__init__([tokens, positions], embedded, name=name)
        #If download weights is true, the weights for the model will be downloaded from a URL and loaded into the model
        
        if download_weights:
            text_encoder_weights_fpath = 'models/kcv_encoder.h5'
            self.load_weights(text_encoder_weights_fpath)
            
'''
Defining helper functions and layers
'''
            
#Defining the Gaussian Error Linear Unit (GELU) activation function           
def quick_gelu(x):
    #GELU is a smooth approximation of the ReLU activation
    return x * tf.sigmoid(x * 1.702)
    
'''
Token embeddings are used to represent the meaning of the words in the input sequence. They capture the semantic information of the words. 
Position embeddings are used to represent the position of the words in the input sequence. The idea is that the same word can have different meanings in different positions in a sentence, 
and that the position of the word can provide additional information to the model. Position embeddings are added to the token embeddings to provide the model with information about the position of the words. 
'''
    
#Defining a custom layer in keras called CLIPEmbedding
#it takes two inputs; tokens and positions, and concatenates their embeddings to produce a final embedding representation
class CLIPEmbedding(keras.layers.Layer):
    def __init__(self, input_dim=49408, output_dim=768, max_length=77, **kwargs):
        super().__init__(**kwargs)
        #creating a token_embedding using an embedding layer
        self.token_embedding = keras.layers.Embedding(input_dim, output_dim)
        #creating a position_embedding using another embedding laer
        self.position_embedding = keras.layers.Embedding(max_length, output_dim)

    #Defining the foward pass thorugh the layer
    def call(self, inputs):
        tokens, positions = inputs
        tokens = self.token_embedding(tokens)
        positions = self.position_embedding(positions)
        #Final embedding representation s obtained by element-wise adding the token and position embeddings
        return tokens + positions
        

#defiing a single layer, CLIPEncoderLayer, in a multi-layer transformer-based architecture       
class CLIPEncoderLayer(keras.layers.Layer):
    #the CLIPEncoderLayer consists of several components
    def __init__(self, embed_dim, num_heads, activation=None, **kwargs):
        super().__init__(**kwargs)
        #Normalization layer that normalizes the input tensor to have mean 0 and variance 1
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        #Layer that performs self-attention on the input tensor
        self.clip_attn = CLIPAttention(embed_dim, num_heads, causal=True)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        #2 fully-connected (dense) layers that perform linear transformations on the input tensor 
        self.fc1 = keras.layers.Dense(embed_dim * 4)
        self.fc2 = keras.layers.Dense(embed_dim)
        self.activation = activation

    #defining the forward pass of the layer
    def call(self, inputs):
        residual = inputs
        #passing the inputs thourgh the layer_norm1 layer
        x = self.layer_norm1(inputs)
        #passing the inputs through the clip-attn layer
        x = self.clip_attn(x)
        #adding the results of the self-attention to the residual connection
        x = residual + x
        residual = x
        #passing the input tensor thorugh layer_norm2 layer
        x = self.layer_norm2(x)
        #passing it through 2 dense layers and an activation function
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        #the results is added to the residual connection and returned as the output of the layer
        return x + residual
        
'''
The CLIPAttention layer is a custom implementation of the attention mechanism in deep learning models. It takes inputs and rturns the attention-weighted representation of the inputs. 
The attention mechanism calculates a weight for each input and combines the inputs based on their weights. In this specific implementation, the attention mechanism is used for multi-head self-attention. 
This means that the layer takes the inputs, projects them into multiple attention heads, and uses each head to attent to different parts of the inputs. The results from each head are then concatenated and transformed to produce the final output
'''
        
#Defining a custom layer called CLIPAttention
class CLIPAttention(keras.layers.Layer):
    #setting up default values and creates some dense layers that will be used in the attention computation
    def __init__(self, embed_dim=768, num_heads=12, causal=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = keras.layers.Dense(self.embed_dim)
        self.k_proj = keras.layers.Dense(self.embed_dim)
        self.v_proj = keras.layers.Dense(self.embed_dim)
        self.out_proj = keras.layers.Dense(self.embed_dim)

    #taking the inputs and reutnring a reshaped version of the inputs
    def reshape_states(self, x, sequence_length, batch_size):
        x = tf.reshape(x, (batch_size, sequence_length, self.num_heads, self.head_dim))
        return tf.transpose(x, (0, 2, 1, 3))  # bs, heads, sequence_length, head_dim

    #implementing the attention computation
    def call(self, inputs, attention_mask=None):
        #if an attention is not provided, it creates a triangular attention mask that sets attention scores for positions to the right of the target sequence to negative infinity
        if attention_mask is None and self.causal:
            length = tf.shape(inputs)[1]
            attention_mask = tfnp.triu(
                tf.ones((1, 1, length, length), dtype=self.compute_dtype) * -tfnp.inf,
                k=1,
            )
        
        #the attention is computed by projecting the inputs into queries, keys and values, reshaping these states and computing the dot product of the query states and the transpose of the key states. 
        _, tgt_len, embed_dim = inputs.shape
        query_states = self.q_proj(inputs) * self.scale
        key_states = self.reshape_states(self.k_proj(inputs), tgt_len, -1)
        value_states = self.reshape_states(self.v_proj(inputs), tgt_len, -1)

        proj_shape = (-1, tgt_len, self.head_dim)
        query_states = self.reshape_states(query_states, tgt_len, -1)
        query_states = tf.reshape(query_states, proj_shape)
        key_states = tf.reshape(key_states, proj_shape)

        src_len = tgt_len
        value_states = tf.reshape(value_states, proj_shape)
        attn_weights = query_states @ tf.transpose(key_states, (0, 2, 1))

        attn_weights = tf.reshape(attn_weights, (-1, self.num_heads, tgt_len, src_len))
        attn_weights = attn_weights + attention_mask
        attn_weights = tf.reshape(attn_weights, (-1, tgt_len, src_len))

        #softmax is applied to the produced attention scores, which are used to compute the weighted sum of the value states
        attn_weights = tf.nn.softmax(attn_weights)
        attn_output = attn_weights @ value_states

        attn_output = tf.reshape(
            attn_output, (-1, self.num_heads, tgt_len, self.head_dim)
        )
        attn_output = tf.transpose(attn_output, (0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (-1, tgt_len, embed_dim))
        #the final attention output is returned after passing it thorugh the out_proj dense layer
        return self.out_proj(attn_output)
