print("version: 13")                                                                         

import os
import random
from prompt_generator import prompt_generator 
import tensorflow as tf
from image_generator import gen_image
import imageio as iio
from PIL import Image
import pickle    
import time
import glob
from datetime import datetime
from datetime import date
from time import mktime
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
from IPython import display
from operator import itemgetter, attrgetter
import shutil
from utils import gen_captions
from utils import summarize

from utils import update_pop_images, tournament_selection, crossover, mutation
print("importing hashtag generator")
from hashtag_generator import generate_hashtags_from_prompt
print("done with importing hashtag generator")        

#Defining function to update initial population folder 
def update_init_population(update_initial_population, img_height, img_width, batch_size, num_steps, unconditional_guidance_scale):
    population_folder = 'evaluations/generated_images/population_0'
    pop_images_path = 'prompt_bank/pop_images'

    if os.path.exists(population_folder):
        pass
    else:  
        os.makedirs(population_folder)


    if len(os.listdir(population_folder)) == 0 or update_initial_population:
        print("no files in the population directory, creating initial population")

        if len(os.listdir(population_folder)) != 0:
            print("clearing population folder")
            for img_name in os.listdir(population_folder):
                file_name = os.path.join(population_folder, img_name)
                os.remove(file_name)
                
        #dict to contain initial population
        posts_dict = []
        available_images = []
        iteration=0
        '''
        #selecting 10 random images
        for image_path in os.listdir(pop_images_path):
            input_path = os.path.join(pop_images_path, image_path)
            available_images.append(input_path)

        selected_images =  random.sample(available_images, 10)
        prompts = []
        
        
        
        
        #generating prompts for selected images
        for image_file in selected_images:
            img = iio.imread(image_file)
            prompt = prompt_generator(img)
            prompts.append(prompt)
        
        '''
        i=0
        
        #generating prompts
        prompts = gen_captions(10)
        
        
        #generating images
        for prompt in prompts:
            print(prompt)
            diffusion_noise = tf.random.normal((batch_size, img_height//8, img_width//8, 4))
            img, latent, prompt = gen_image(prompt, diffusion_noise, batch_size, num_steps, unconditional_guidance_scale)
            Image.fromarray(img[0]).save(f"{population_folder}/{i}.png") 
            hashtag = generate_hashtags_from_prompt(prompt)
            
            posts_dict.append({'num' : i, 'fitness' : 0, 'latent': latent, 'prompt': prompt, 'hashtag': hashtag})
             
         
            i+=1
            
        print('done creating')
    
        with open(f'evaluations/logs/posts_dict_{iteration}.txt', 'wb') as f:
            pickle.dump(posts_dict, f)
        
    else:
        print("files in init population. Not update selected")
    
    
print("V3")    
    
def main(iteration, update, clear, num, gs):  
    img_height = 512
    img_width = 512
    batch_size = 1        
    num_steps = num        
    unconditional_guidance_scale = gs
    update_initial_population=update       
    
    pop_images_path = 'prompt_bank/pop_images'
    
    mutation_probability = 0.1
    crossover_probability = 0.8      
    style_change=0.5
    iteration = int(iteration)    
    
    history_avg_score = []
    
    #updating pop_images folder
    #update_pop_images()
    
    if clear:  
    #clearing folders in"generated images"
        for root, dirs, files in os.walk('evaluations/generated_images/'):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        
        #Clearing log folder
        for root, dirs, files in os.walk('evaluations/logs/'):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    
    
        
       
    
    print(f'starting on iteration: {iteration}')
    
    if iteration==0:
        #updating initial population
        
        population_folder = f'evaluations/generated_images/population_{iteration}'
        
        update_init_population(update_initial_population, img_height, img_width, batch_size, num_steps, unconditional_guidance_scale)
        
        with open(f'evaluations/logs/posts_dict_{iteration}.txt', 'rb') as handle:
            posts_dict = pickle.loads(handle.read())
        
        
    
        #get fitness
        #getting fitness score for original population
        i=0
        for image_path in os.listdir(population_folder):
            print(i)
        #read image from population folder
            input_path = os.path.join(population_folder, image_path)
            img = iio.imread(input_path)
            prompt = posts_dict[i]['prompt']
            print(prompt)
            hashtag = posts_dict[i]['hashtag']

            #generate hashtag for image
            #hashtag = generate_hashtags_from_prompt(prompt)

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
                posts_dict[i]['fitness'] = 1
            elif response.lower() == 'n' or response.lower() == 'no':
                posts_dict[i]['fitness'] = -1
            else:
                print('Illegal answer. Score is unchanged.')
                pass

            i+=1
            
        #updating dict file
        with open(f'evaluations/logs/posts_dict_{iteration}.txt', 'wb') as f:
            pickle.dump(posts_dict, f)
            
        #calculating average fitness of population
        avg_score = 0
        for post in posts_dict:
            avg_score += post['fitness']
        avg_score /= len(posts_dict)
        
        print('Iteration', iteration, ', avg. population score = ', round(avg_score, 2))
        
        timestamp = date.today()
        timestamp = timestamp.strftime("%Y-%m-%d")
        log_file = f'evaluations/logs/log_{timestamp}.txt'
        
        with open(log_file, 'a') as f:
            f.write(f'\niteration: {iteration}')
            f.write(f'\navg score: {avg_score}')
            f.write('\n######################################') 
         
        iteration+=1  
        
    
    go = True  
    
    while go:      
        #current population
        population_folder = f'evaluations/generated_images/population_{iteration}'
        
        #last population
        last_population_folder = f'evaluations/generated_images/population_{iteration-1}'
        

            
        if os.path.exists(population_folder):
            pass
        else:
            os.makedirs(population_folder)
        
    
          
        
        if len(os.listdir(population_folder)) == 0:
            have_population=False
            #get last population
            with open(f'evaluations/logs/posts_dict_{iteration-1}.txt', 'rb') as handle:
                posts_dict = pickle.loads(handle.read())
        else:
            have_population=True
            with open(f'evaluations/logs/posts_dict_{iteration}.txt', 'rb') as handle:
                posts_dict = pickle.loads(handle.read())
        
    
        #generate new population
        if have_population==False:
            
            
            
            
            #shuffle population
            random.shuffle(posts_dict)
            
            #initializing variables
            posts_evolve = []
            posts_change_style = [] 
            new_population = []
            
            #sorting population
            sorted_dict = sorted(posts_dict, key=itemgetter('fitness'))
    
            #the following 4 indivudals will evolve
            #posts_evolve.append(sorted_dict[4:8])
            posts_evolve.append(sorted_dict[4:10])

            #the last 2 individuals will change style
            posts_change_style.append(sorted_dict[8:10])
            
            '''
            changing style
            '''
            
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
                print(prompt)
                fitness = posts_change_style[0][i]['fitness']
                diffusion_noise = tf.random.normal((batch_size, img_height//8, img_width//8, 4))
                img, latent, prompt = gen_image(prompt, diffusion_noise, batch_size, num_steps, unconditional_guidance_scale)
                hashtag = generate_hashtags_from_prompt(prompt)
                new_population.append({'num' : a, 'fitness':fitness, 'latent': latent, 'prompt': prompt, 'hashtag': hashtag})
                Image.fromarray(img[0]).save(f"{population_folder}/{a}.png") 
                a+=1
          
            '''
            Evolving old:
            
            s=3
            selected_parents = []
            
            for _ in range(len(posts_evolve[0])):
                selected_parent = tournament_selection(posts_evolve[0])
                selected_parents.append(selected_parent)
            
            for i in range(0, len(posts_evolve[0]), 2):

                some_post1 = selected_parents[i]

                try:
                    some_post2 = selected_parents[i + 1]
     
                except IndexError:
                    some_post2 = selected_parents[i - 1]

                list_of_children_genome = crossover(some_post1['latent'], some_post2['latent'], crossover_probability)
                c_prompt = some_post1['prompt'] + ' combinded with: ' + some_post2['prompt']
  

  
                for child_genome in list_of_children_genome:
                 # Mutation

                    child_genome = mutation(child_genome, mutation_probability)
                    diffusion_noise = child_genome
                    prompt = c_prompt

                    rand_style_num = random.uniform(0,1)
    
                    if rand_style_num < style_change:
      
                        word = 'style:'
                        if word in prompt: 
                            index = prompt.find(word)
                            prompt = prompt[:index]
                        prompt = prompt + ' style: ' + styles[s]
                        s+=1

                    print(prompt)

                    img, latent, prompt = gen_image(prompt, diffusion_noise, batch_size, num_steps, unconditional_guidance_scale)

                    prompt = prompt_generator(img)

                    new_population.append({'num' : a, 'fitness':0, 'latent': diffusion_noise, 'prompt': prompt})
    
                    Image.fromarray(img[0]).save(f"{population_folder}/{a}.png") 
                    a+=1
            
            '''
            
            '''
            evolving new:
            '''
             
            
            a=2
            style_change=0.5
            s=3
            selected_parents = []
            
            random.shuffle(posts_evolve)

            for _ in range(len(posts_evolve[0])):
                selected_parent = tournament_selection(posts_evolve[0])
                selected_parents.append(selected_parent)
              
            random.shuffle(selected_parents)                 
            
            for i in range(len(selected_parents))[:4]:
                        
                print(i)
                some_post1 = selected_parents[i]
                #print(some_post1['prompt'])       

                try:
                    some_post2 = selected_parents[i + 1]
     
                except IndexError:
                    some_post2 = selected_parents[i - 1]

                #print(some_post2['prompt'])        

                list_of_children_genome = crossover(some_post1['latent'], some_post2['latent'], crossover_probability)
                
                rand_style_num = random.uniform(0,1)
                
                prompt1 = some_post1['prompt']
                if rand_style_num < style_change:
      
                    word = 'style:'
                    if word in prompt1: 
                        index = prompt1.find(word)
                        prompt1 = prompt1[:index]
                    prompt1 = prompt1 + ' style: ' + styles[s]
                    s+=1
                
                print(prompt1)
                rand_style_num = random.uniform(0,1)
                
                prompt2 = some_post2['prompt']
                if rand_style_num < style_change:
      
                    word = 'style:'
                    if word in prompt2: 
                        index = prompt2.find(word)
                        prompt = prompt2[:index]
                    prompt2 = prompt2 + ' style: ' + styles[s]
                    s+=1
                
                
                print(prompt2)
                #c_prompt = some_post1['prompt'] + ' combinded with: ' + some_post2['prompt']
                
                c_prompt = prompt1 + ' combinded with: ' + prompt2
  
                rand_index = random.randint(0,1)          
                child_genome = list_of_children_genome[rand_index]                 
                child_genome = mutation(child_genome, mutation_probability)
                diffusion_noise = child_genome
                prompt = c_prompt
                print(prompt)
                prompt_len = (len(prompt))
                print(prompt_len)
                

                try:
                    img, latent, prompt = gen_image(prompt, diffusion_noise, batch_size, num_steps, unconditional_guidance_scale)
                except:
                    prompt1 = summarize(prompt1)
                    prompt2 = summarize(prompt2)
                    
                    prompt = prompt1 + ' combinded with: ' + prompt2 
                    rand_style_num = random.uniform(0,1)
                    if rand_style_num < style_change:
                        prompt = prompt + 'style: '+ styles[s]
                        s+=1
                    print(prompt)
                    
                    img, latent, prompt = gen_image(prompt, diffusion_noise, batch_size, num_steps, unconditional_guidance_scale)

                prompt = prompt_generator(img)
                hashtag = generate_hashtags_from_prompt(prompt)

                new_population.append({'num' : a, 'fitness':0, 'latent': diffusion_noise, 'prompt': prompt, 'hashtag': hashtag})
    
                Image.fromarray(img[0]).save(f"{population_folder}/{a}.png") 
                a+=1         
           
            '''
            aDDING 2 NEW
            '''
            #updating pop images
            #update_pop_images()
            
            '''
            prompts = []  
            available_images = []      
            pop_images_path = 'prompt_bank/pop_images'  

            for image_path in os.listdir(pop_images_path):
               
                input_path = os.path.join(pop_images_path, image_path)
                available_images.append(input_path)

            #selected_images =  random.sample(available_images, 4)

                
            random.shuffle(available_images)
        

            for image_file in available_images[:4]:
                print(image_file)
                try:
                    img = iio.imread(image_file)
                    prompt = prompt_generator(img)
                    prompts.append(prompt)
                except:
                    print('error, trying new one')
                    pass
           
            print(f'number of images selected: {len(prompts)}')        
            '''
            prompts = gen_captions(4)
                
            for pr in prompts:

                print(pr)
                diffusion_noise = tf.random.normal((batch_size, img_height//8, img_width//8, 4))
                img, latent, prompt = gen_image(pr, diffusion_noise, batch_size, num_steps, unconditional_guidance_scale)
                hashtag = generate_hashtags_from_prompt(prompt)
                Image.fromarray(img[0]).save(f"{population_folder}/{a}.png")    
            
                new_population.append({'num' : a, 'fitness' : 0, 'latent': latent, 'prompt': prompt, 'hashtag': hashtag})
                

                a+=1   
                
                
            posts_dict = new_population 
            
            with open(f'evaluations/logs/posts_dict_{iteration}.txt', 'wb') as f:
                pickle.dump(posts_dict, f)
            
        #Get fitness

        
    
        i=0
        #getting fitness score for current population
        for image_path in os.listdir(population_folder):
            print(i)
        #read image from population folder
            input_path = os.path.join(population_folder, image_path)
            img = iio.imread(input_path)
            prompt = posts_dict[i]['prompt']
            print(prompt)
            hashtag = posts_dict[i]['hashtag']

            #generate hashtag for image
            #hashtag = generate_hashtags_from_prompt(prompt)

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
                posts_dict[i]['fitness'] = 1
            elif response.lower() == 'n' or response.lower() == 'no':
                posts_dict[i]['fitness'] = -1
            else:
                print('Illegal answer. Score is unchanged.')
                pass

            i+=1
            
      
            
        #calculating average fitness of population
        avg_score = 0
        for post in posts_dict:
            avg_score += post['fitness']
        avg_score /= len(posts_dict)
        
        print('Iteration', iteration, ', avg. population score = ', round(avg_score, 2))
        
        timestamp = date.today()
        timestamp = timestamp.strftime("%Y-%m-%d")
        log_file = f'evaluations/logs/log_{timestamp}.txt'
        
        with open(log_file, 'a') as f:
            f.write(f'\niteration: {iteration}')
            f.write(f'\navg score: {avg_score}')
            f.write('\n######################################') 
                    
        
        
        with open(f'evaluations/logs/posts_dict_{iteration}.txt', 'wb') as f:
            pickle.dump(posts_dict, f)
            
        iteration += 1
      
        choice = input('continue? (y or n): ')
        if choice.lower() == 'n':
            print(f'Ending. Reached iteration: {iteration-1}')
            go = False
        elif choice.lower() == 'y':
            go = True
            print(f'starting ieration nr: {iteration}')
        else:
            print(f'illegal answer. Ending. Reached iteration: {iteration-1} ')
        
      
    
    
        
    
    
    