'''
Importing dependencies
'''

#Costum dependencies
from utils import tournament_selection, crossover, mutation, gen_captions, summarize
from prompt_generator import prompt_generator 
from image_generator import gen_image
from hashtag_generator import generate_hashtags_from_prompt

#Other dependencies
import os.path
from os import path
import os
import random
import imageio as iio
import tensorflow as tf
from PIL import Image
import pickle
from operator import itemgetter, attrgetter
from datetime import date
import re
import shutil
from matplotlib import pyplot as plt
import numpy as np
from time import sleep
import numpy as np
from IPython import display

'''
Defining helper function
'''

#Defining function to update initial population folder 
def update_init_population(update_initial_population, img_height, img_width, batch_size, num_steps, unconditional_guidance_scale, key):
    #folder to store images
    population_folder = 'generated_images/population_0'
    
    #cleaning up folder if necessary
    if os.path.exists(population_folder):
        pass  
    else:
        os.makedirs(population_folder)

    if len(os.listdir(population_folder)) == 0 or update_initial_population:
        print("Creating initial population")

        if len(os.listdir(population_folder)) != 0:
            for img_name in os.listdir(population_folder):
                file_name = os.path.join(population_folder, img_name)
                os.remove(file_name)
                
        #dict to contain initial population
        posts_dict = []
        available_images = []
        iteration=0
        
        i=0
        
        #generating prompts
        prompts = gen_captions(10, key)
    
        #generating images
        for prompt in prompts:
            diffusion_noise = tf.random.normal((batch_size, img_height//8, img_width//8, 4))
            #generating image   
            try:
                print(prompt)
                img, latent, prompt = gen_image(prompt, diffusion_noise, batch_size, num_steps, unconditional_guidance_scale)
            #if unable to create image, summarize the prompts
            except:
                prompt = summarize(prompt)
                print(prompt)
                img, latent, prompt = gen_image(prompt, diffusion_noise, batch_size, num_steps, unconditional_guidance_scale)
            
            
            
            
            img, latent, prompt = gen_image(prompt, diffusion_noise, batch_size, num_steps, unconditional_guidance_scale)
            Image.fromarray(img[0]).save(f"{population_folder}/{i}.png") 
            hashtag = generate_hashtags_from_prompt(prompt)
            
            posts_dict.append({'num' : i, 'fitness' : 0, 'latent': latent, 'prompt': prompt, 'hashtag': hashtag})      
            i+=1
            
        print('Initial population created')
    
        #storing population
        with open(f'logs/posts_dict_{iteration}.txt', 'wb') as f:
            pickle.dump(posts_dict, f)
        
    else:
        print("Initial population exists. No update selected")
        
        
'''
Defining main function
'''

def main(iteration, update, clear, num, gs, ngs, key): 
    #Defining parameters
    img_height = 512
    img_width = 512
    batch_size = 1        
    num_steps = num        
    unconditional_guidance_scale =gs
    new_scale = ngs
    update_initial_population=update       
    mutation_probability = 0.1
    crossover_probability = 0.8
    style_change=0.5
    iteration = int(iteration)    
    
    #Defining dict to store average fitness score
    history_avg_score = []

    go = True
    print(f'starting on iteration: {iteration}')
    
    #if "clear" parameter is set, all generated images/populations is deleted
    if clear: 
        #clearing folders in"generated images"
        for root, dirs, files in os.walk('generated_images/'):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        
        #Clearing log folder
        for root, dirs, files in os.walk('logs/'):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
                
        #Clearing return folder
        for root, dirs, files in os.walk('return/'):    
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))    
    
    #if this is the first iteration, the initial population must be created
    if iteration==0:    
      population_folder = f'generated_images/population_{iteration}'
      if os.path.exists(population_folder):
        if len(os.listdir(population_folder)) < 10:
            update_initial_population=True
      #updating initial population
      update_init_population(update_initial_population, img_height, img_width, batch_size, num_steps, new_scale, key)
    
    while go:
        
        #get current population
        with open(f'logs/posts_dict_{iteration}.txt', 'rb') as handle:
            posts_dict = pickle.loads(handle.read())
            
        population_folder = f'generated_images/population_{iteration}'
        if not os.path.exists(population_folder):
            os.makedirs(population_folder)

        #creating folder for next iteration
        next_iteration = iteration + 1
        next_population_folder = f'generated_images/population_{next_iteration}'
        if not os.path.exists(next_population_folder):
            os.makedirs(next_population_folder)

        '''
        Getting fitness/feedback for current population
        '''
        b=0
        for image_path in os.listdir(population_folder):
            #read image from population folder
            input_path = os.path.join(population_folder, image_path)
            img = iio.imread(input_path)
            prompt = posts_dict[b]['prompt']

            #generate hashtag for image
            hashtag = generate_hashtags_from_prompt(prompt)

            #plot image
            plt.figure(figsize=(10, 10))
            plt.imshow(np.squeeze(img).astype("uint8"))
            plt.title(hashtag)
            plt.axis("off")
  
            display.display(plt.gcf())
            display.clear_output(wait=True)
            sleep(0.5)

            #get score from user
            response = input('would you interact with this post?')
            if response.lower() == 'y' or response.lower() == 'yes':
                posts_dict[b]['fitness'] = 1
            elif response.lower() == 'n' or response.lower() == 'no':
                posts_dict[b]['fitness'] = -1
            else:
                print('Illegal answer. Score is unchanged.')
                pass

            b+=1
            
        #updating dict file
        with open(f'logs/posts_dict_{iteration}.txt', 'wb') as f:
            pickle.dump(posts_dict, f)
      
        #calculating average fitness of population
        avg_score = 0
        for post in posts_dict:
          avg_score += post['fitness']
        avg_score /= len(posts_dict)
        
        print('Iteration', iteration, ', avg. population score = ', round(avg_score, 2))
        
        timestamp = date.today()
        timestamp = timestamp.strftime("%Y-%m-%d")
        log_file = f'logs/log_{timestamp}.txt'
        
        #writing current scores to log file
        with open(log_file, 'a') as f:
          f.write(f'\niteration: {iteration}')
          f.write(f'\navg score: {avg_score}')
          f.write('\n######################################') 
            
        '''
        creating new population
        '''
        print("Creating new population")
                  
        #shuffle population
        random.shuffle(posts_dict)
            
        #initializing variables
        posts_evolve = []
        posts_change_style = [] 
        new_population = []
            
        #sorting population
        sorted_dict = sorted(posts_dict, key=itemgetter('fitness'))    

        #all posts will join in the genetic algorithm
        posts_evolve = posts_dict
        #the last 2 individuals will change style
        posts_change_style.append(sorted_dict[8:10])
            
        '''
        changing style
        '''
        print("changing style on two of the most popular images")
            
        #Getting styles 
        styles_file = open('prompt_bank/styles.txt', "r")
        styles = styles_file.read()
        styles = styles.split("\n")
        random.shuffle(styles)
            
        new_prompts = []    
        prompts = []

        #Creating new prompts
        for i in range(len(posts_change_style[0])):
            word = 'style:'
            prompt = posts_change_style[0][i]['prompt']
            if word in prompt:
                index = prompt.find(word)
                prompt = prompt[:index]
            prompt = prompt + ' style: ' + styles[i]
            posts_change_style[0][i]['prompt'] = prompt
            
        #generating new images with changed style
        a = 0
        for i in range(len(posts_change_style[0])):
            prompt = posts_change_style[0][i]['prompt']
            fitness = posts_change_style[0][i]['fitness']
            diffusion_noise = tf.random.normal((batch_size, img_height//8, img_width//8, 4))
            img, latent, prompt = gen_image(prompt, diffusion_noise, batch_size, num_steps, new_scale)
            hashtag = generate_hashtags_from_prompt(prompt)
            new_population.append({'num' : a, 'fitness':fitness, 'latent': latent, 'prompt': prompt, 'hashtag': hashtag})
            Image.fromarray(img[0]).save(f"{next_population_folder}/{a}.png") 
            a+=1     
            
        '''
        Evolving images
        '''
        
        print("Evolving images based on fitness scores, creaing 4 new images")
        
        a=2
        style_change=0.5
        s=3
        selected_parents = []
            
        random.shuffle(posts_evolve)

        #select parents using tournament selection
        for _ in range(4):     
            selected_parent = tournament_selection(posts_evolve)            
            selected_parents.append(selected_parent)
              
        random.shuffle(selected_parents)   
        
        archive = []
            
        #creating offspring/children    
        for i in range(len(selected_parents))[:4]:                        
            some_post1 = selected_parents[i]   

            try:
                some_post2 = selected_parents[i + 1]
     
            except IndexError:
                some_post2 = selected_parents[i - 1] 

            #using crossover to combine two parents two form a children
            list_of_children_genome = crossover(some_post1['latent'], some_post2['latent'], crossover_probability)
                
            #ensuring that the same parent-combination is not used twice    
            archive_prompt_1 = some_post1['prompt'] + some_post2['prompt']
            archive_prompt_2 = some_post2['prompt'] + some_post1['prompt']
            
            #Ensuring that new parent combinations are used 
            do = True
            while do:
                if archive_prompt_1 in archive or archive_prompt_2 in archive or some_post1['prompt']==some_post2['prompt']:
                    random.shuffle(posts_evolve)
                    
                    selected_parent_1 = tournament_selection(posts_evolve)
                    selected_parent_2 = tournament_selection(posts_evolve)
                               
                    some_post1 = selected_parent_1
                    some_post2 = selected_parent_2    

                    list_of_children_genome = crossover(some_post1['latent'], some_post2['latent'], crossover_probability)
                    
                    archive_prompt_1 = some_post1['prompt'] + some_post2['prompt']
                    archive_prompt_2 = some_post2['prompt'] + some_post1['prompt']
                                  
                else:
                    do = False               
            
            #the resulting children will change style with some propability
            rand_style_num = random.uniform(0,1)
                
            prompt1 = some_post1['prompt']
           
            if rand_style_num < style_change:
                changed = True
                word = 'style:'
                if word in prompt1: 
                    index = prompt1.find(word)
                    prompt1 = prompt1[:index]
                prompt1 = prompt1 + ' style: ' + styles[s]
                s+=1
               
            #changing style of the children's prompt with given probability   
            rand_style_num = random.uniform(0,1)
                
            prompt2 = some_post2['prompt']
            if rand_style_num < style_change and not changed:
      
                word = 'style:'
                if word in prompt2: 
                    index = prompt2.find(word)
                    prompt = prompt2[:index]
                prompt2 = prompt2 + ' style: ' + styles[s]
                s+=1
    
            c_prompt = prompt1 + ' combinded with: ' + prompt2
            archive_prompt_1 = some_post1['prompt'] + some_post2['prompt']
            archive_prompt_2 = some_post2['prompt'] + some_post1['prompt']
            archive.append(archive_prompt_1)
            archive.append(archive_prompt_2)

            #performing mutation on the new children with given probability
            rand_index = random.randint(0,1)          
            child_genome = list_of_children_genome[rand_index]                 
            child_genome = mutation(child_genome, mutation_probability)
            
            #creating image
            diffusion_noise = child_genome
            prompt = c_prompt
            prompt_len = (len(prompt))
            try:
                img, latent, prompt = gen_image(prompt, diffusion_noise, batch_size, num_steps, unconditional_guidance_scale)
            #if unable to create image, summarize the prompts
            except:
                prompt1 = summarize(prompt1)
                prompt2 = summarize(prompt2)
                    
                prompt = prompt1 + ' combinded with: ' + prompt2 
                rand_style_num = random.uniform(0,1)
                if rand_style_num < style_change:
                    prompt = prompt + 'style: '+ styles[s]
                    s+=1
                    
                img, latent, prompt = gen_image(prompt, diffusion_noise, batch_size, num_steps, unconditional_guidance_scale)
            
            #create a new prompt for the created image
            prompt = prompt_generator(img)
            #create corresponding hashtag
            hashtag = generate_hashtags_from_prompt(prompt)

            #appending the new children to the new population
            new_population.append({'num' : a, 'fitness':0, 'latent': diffusion_noise, 'prompt': prompt, 'hashtag': hashtag})
    
            #storing the image
            Image.fromarray(img[0]).save(f"{next_population_folder}/{a}.png") 
            a+=1         
           
        '''
        Adding 4 new images
        '''
        
        print("adding 4 new images to the new population")
        
        #creating new prompts
        prompts = gen_captions(4, key)               
        for pr in prompts:
            #creating images and corresponding hashtags
            diffusion_noise = tf.random.normal((batch_size, img_height//8, img_width//8, 4))
            img, latent, prompt = gen_image(pr, diffusion_noise, batch_size, num_steps, new_scale)
            hashtag = generate_hashtags_from_prompt(prompt)
            Image.fromarray(img[0]).save(f"{next_population_folder}/{a}.png")    
            #adding images to the new population
            new_population.append({'num' : a, 'fitness' : 0, 'latent': latent, 'prompt': prompt, 'hashtag': hashtag})                

            a+=1   
                
        #storing new popluation dict
        posts_dict = new_population 

        with open(f'logs/posts_dict_{next_iteration}.txt', 'wb') as f:
            pickle.dump(posts_dict, f)

        iteration = next_iteration
        
        #Giving user choice of continue or end the framework
        choice = input('continue? (y or n): ')
        if choice.lower() == 'n':
            print(f'Ending. Reached iteration: {iteration-1}')
            go = False
        elif choice.lower() == 'y':
            go = True
            print(f'starting ieration nr: {iteration}')
        else:
            print(f'illegal answer. Ending. Reached iteration: {iteration-1} ')
        
