'''
importing dependencies              
''' 
import os
import urllib.request
import random
import time
from datetime import datetime
from datetime import date
from time import mktime
import numpy as np   
from transformers import pipeline

import openai
import random
import re

'''
Defining helper functions
'''

#Defining tournament selection function
def tournament_selection(some_posts, k=3):
  """
  perform tournament selection with k (3 by default) opponents
  """
  #Randomly selection one individual, champion by default
  champion_index = random.randint(0, len(some_posts) - 1)


  for _ in range(0, k):  # k opponents to randomly selected individual
  # Perform tournament
    opponent_index = random.randint(0, len(some_posts) - 1)
    opponent_fitness = some_posts[opponent_index]['fitness']
    champion_fitness = some_posts[champion_index]['fitness']
    if opponent_fitness > champion_fitness:
      # The best wins and is stored
      champion_index = opponent_index
  
  champion = some_posts[champion_index]

  return champion
  
  
#Defining function to perform crossover
def crossover(some_post1_genotype, some_post2_genotype, crossover_probability):

  # Evaluate recombination
  rand_c_num = random.uniform(0,1)

  if rand_c_num < crossover_probability:
    some_post1_genotype = np.array(some_post1_genotype)
    some_post1_genotype_list = some_post1_genotype.tolist()

    crossover_point = random.randint(0,len(some_post1_genotype_list[0]))
    

    part_1 = some_post1_genotype_list[0][:crossover_point]

    some_post2_genotype = np.array(some_post2_genotype)
    some_post2_genotype_list = some_post2_genotype.tolist()

    part_2 = some_post2_genotype_list[0][crossover_point:]

    some_post_child1_genotype = part_1 + part_2 
    some_post_child2_genotype =  part_2 + part_1

    some_post_child1_genotype = np.asarray(some_post_child1_genotype)
    some_post_child2_genotype = np.asarray(some_post_child2_genotype)

    some_post_child1_genotype = np.reshape(some_post_child1_genotype, (1, 64, 64, 4))
    some_post_child2_genotype = np.reshape(some_post_child1_genotype, (1, 64, 64, 4))


  else:
    # Children are copies of parents by default
    some_post_child1_genotype, some_post_child2_genotype = some_post1_genotype, some_post2_genotype

  
  list_of_children = [np.array(some_post_child1_genotype), np.array(some_post_child2_genotype)]

  
  return list_of_children
  
def mutation(genotype, mutation_probability):
  
  genotype = genotype.tolist()

  j = 0
  k = 0
  i = 0
  counter = 0
 
  for j in range(len(genotype[0])):  # Loop through genotype
    for i in range(len(genotype[0])):
      for k in range(3):
        # Evaluate mutation
        rand_m_n = random.uniform(0,1)
        if rand_m_n < mutation_probability:
          # Replace item with a random value drawn from 0 mean and 1 standard deviation distribution
          genotype[0][j][i][k] = np.random.normal()
          counter+=1

      
  genotype = np.asarray(genotype)


  return genotype
  
  
# Generate random image captions
def generate_caption(prompt):
    model_engine = "text-davinci-002"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    caption = response.choices[0].text.strip()
    return caption
    
    
def gen_captions(num, key):
    prompts = []
    #openai.api_key = "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"
    openai.api_key = key
    
    themes_file = open('prompt_bank/themes.txt', 'r')
    themes = themes_file.read()
    themes = themes.split("\n")
    random.shuffle(themes)
    
    model_engine = "text-davinci-002"
    
    search_list = ['guy', 'friends']
    
    for theme in themes[:num]:
        condition=True
        while(condition):
            prompt = 'Generate a descriptive image caption with specified style for ' + theme + ':'
            caption = generate_caption(prompt)
            if re.compile('|'.join(search_list), re.IGNORECASE).search(caption):
                condition=True
            else:
                condition=False
                ret = caption + ' style: high resolution, 4k'

                prompts.append(ret)
       
        
    return prompts
    
def summarize(sentence):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(sentence, max_length=30, min_length=5, do_sample=False)[0]['summary_text']
    
    return summary
    

