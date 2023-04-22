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

"""Keras implementation of StableDiffusion.
Credits:
- Original implementation:
  https://github.com/CompVis/stable-diffusion
- Initial TF/Keras port:
  https://github.com/divamgupta/stable-diffusion-tensorflow

This implementation is based on: 
https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion/stable_diffusion.py"""

#Setting log level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#importing custom dependencies
from clip_tokenizer import SimpleTokenizer
from text_encoder import TextEncoder
from diffusion_model import DiffusionModel
from constants import _ALPHAS_CUMPROD
from constants import _UNCONDITIONAL_TOKENS
from decoder import Decoder

#import other dependencies
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

'''
Defining hyperparameters
'''

img_height = 512
img_width = 512
seed = None
MAX_PROMPT_LENGTH = 77

'''
Loading models
'''

#loading text encoder
def text_encoder(jit_compile=False):
  #returns the text encoder with pretrained weights
  text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
  if jit_compile:
    text_encoder.compile(jit_compile=True)
  return text_encoder

text_encoder = text_encoder()

#loading diffusion model
def diffusion_model(jit_compile=False):
  #returns diffusion model with pretrained weights
  diffusion_model = DiffusionModel(
      img_height, img_width, MAX_PROMPT_LENGTH
  )
  if jit_compile:
    diffusion_model.compile(jit_compile=True)
  return diffusion_model

diffusion_model = diffusion_model()

#Loading decoder
def decoder(jit_compile=False):
  #returns the diffusion image decoder with pretrained weights
  decoder = Decoder(img_height, img_width)
  if jit_compile:
    decoder.compile(jit_compile=True)
  return decoder

decoder = decoder()

#loading tokenizer
def tokenizer():
  #returns the tokenizer usd for text inputs
  tokenizer = SimpleTokenizer()
  return tokenizer

tokenizer = tokenizer()

'''
Defining functions
'''

#This function extends a tensor by repeating it to fit the shape of the given batch size
def expand_tensor(text_embedding, batch_size):
  #removing all dimensions with size 1 from the tensor, so that the tensor has the minimal number of dimensions
  text_embedding = tf.squeeze(text_embedding)
  #The rank of a tensor is the number of dimensions it has
  if text_embedding.shape.rank == 2:
    #if rank equals 2, the tensor is extended by repeating it to fit the batch_size. 
    text_embedding = tf.repeat(
        tf.expand_dims(text_embedding, axis=0), batch_size, axis=0
    )
  #the extended tensor is returned as the output of the function
  return text_embedding
  
  
#This function generates a tensor of random values with specified shape
def get_initial_diffusion_noise(batch_size, seed):
  if seed is not None:
    return tf.random.stateless_normal(
        (batch_size, img_height//8, img_width//8, 4),
        seed = [seed, seed],
    )
  else:
    return tf.random.normal(
        (batch_size, img_height//8, img_width//8, 4)
    )

def get_pos_ids():
  return tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
   
def get_initial_alphas(timesteps):
  #list that contains elements from the ALPHAS_ComPROD list, with indices specified by the timesteps argument
  alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
  #list that starts with the value 1, followed by all elements of the alpha list except the last on
  alphas_prev = [1.0] + alphas[:-1]

  return alphas, alphas_prev
  
#This function generates timestep embedding for a single timestep 
def get_timestep_embedding(timestep, batch_size, dim=320, max_period=10000):
  #computing the half of the embedding dimension
  half = dim // 2
  #calculating a range of frquencies
  freqs = tf.math.exp(
      -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
  )
  #multiplying timestep with freqs to get args
  args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
  #calculating cos and sin of args, and concatenate tem along the first dimension to form the embedding tensor
  embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
  #reshaping embedding to a tensor with shape [1, -1]
  embedding = tf.reshape(embedding, [1, -1])
  #repeating the 'embedding' tensor 'batch_size' times along the first dimension, returns a embedding tensor of shape [batch_size, dim]
  return tf.repeat(embedding, batch_size, axis=0)
  
def get_unconditional_context():
  unconditional_tokens = tf.convert_to_tensor(
      [_UNCONDITIONAL_TOKENS], dtype=tf.int32
  )
  unconditional_context = text_encoder.predict_on_batch(
      [unconditional_tokens, get_pos_ids()]
  )

  return unconditional_context
  
#Defining function to encode text into a latent text encoding
def encode_text(prompt):
  #tokenizing the input prompt using the SimpleTokenizer function 
  inputs = tokenizer.encode(prompt)

  #adding a special token 49407 tot he end of the tokenized prompt until it reaches
  #the length of MAX_PROMPT_LENGTH
  phrase = inputs + [49407] * (MAX_PROMPT_LENGTH - len(inputs))
  #converting the tokenized prompt into a tensor of dtype tf.int32
  phrase = tf.convert_to_tensor([phrase], dtype=tf.int32)

  #using textEncoder model to generate latent text encoding of the prompt, by passing
  #the tokenized prompt and position IDs as inputs 
  
  context = text_encoder.predict_on_batch([phrase, get_pos_ids()])
    
  #returning the latent text encoding of the prompt
  return context
  
'''
Generating images
'''

def gen_image(prompt, diffusion_noise,
     batch_size,
     num_steps,
     unconditional_guidance_scale,
     seed=seed):
    
    #Encoding text prompt
    encoded_text = encode_text(prompt)
    context = expand_tensor(encoded_text, batch_size)

    unconditional_context = tf.repeat(
      get_unconditional_context(), batch_size, axis=0)

    #defining initial diffusion noise/latent vector
    if diffusion_noise is not None:
        diffusion_noise = tf.squeeze(diffusion_noise)
        if diffusion_noise.shape.rank == 3:
            diffusion_noise = tf.repeat(
            tf.expand_dims(diffusion_noise, axis=0), batch_size,axis=0)
        latent = diffusion_noise
    else:
        latent = get_initial_diffusion_noise(batch_size, seed)
    
    tmp_latent = latent

    timesteps = tf.range(1, 1000, 1000 // num_steps)
    alphas, alphas_prev = get_initial_alphas(timesteps)
    progbar = keras.utils.Progbar(len(timesteps))
    iteration = 0

    #diffusion process
    for index, timestep in list(enumerate(timesteps))[::-1]:
        latent_prev = latent  # Set aside the previous latent vector
        t_emb = get_timestep_embedding(timestep, batch_size)
        unconditional_latent = diffusion_model.predict_on_batch([latent, t_emb, unconditional_context])

        latent = diffusion_model.predict_on_batch([latent, t_emb, context])
        latent = unconditional_latent + unconditional_guidance_scale * (latent - unconditional_latent)
        
        a_t, a_prev = alphas[index], alphas_prev[index]
        pred_x0 = (latent_prev - math.sqrt(1 - a_t) * latent) / math.sqrt(a_t)
        latent = latent * math.sqrt(1.0 - a_prev) + math.sqrt(a_prev) * pred_x0
        iteration += 1
        progbar.update(iteration)
  
    #creating decoded vector
    decoded = decoder.predict_on_batch(latent)
    decoded = ((decoded + 1) / 2) * 255
       
    return np.clip(decoded, 0, 255).astype("uint8"), tmp_latent, prompt

