print('version: 2')    

#dependencies
from utils import update_pop_images, tournament_selection, crossover, mutation
from prompt_generator import prompt_generator 
from image_generator import gen_image
from hashtag_generator import generate_hashtags_from_prompt

from utils import gen_captions, summarize

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


#Defining function to update initial population folder 
def update_init_population(update_initial_population, img_height, img_width, batch_size, num_steps, unconditional_guidance_scale):
    population_folder = 'generated_images/population_0'
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
        
        #prompts = gen_captions(10)
        
        prompts = [
    'a cute cat. high resolution. 4k',
    'a cute dog. high resolution. 4k',
    'a cute and fluffy rabbit. high resolution. 4k',
    'a polar bear in a winter landscape. high resolution. 4k', 
    'a leopard hunting for its next prey. high resolution. 4k',
    'a cute bear in a deep forrest. high resolution. 4k',
    'a horse on the field. high resolution. 4k', 
    'a majestestic lion. high resolution. 4k',
    'a cute baby elephant in its natural habbitat. high resolution. 4k', 
    'a turtle swimming in a blue and clear ocean. high resolution. 4k',
    'fish swimming in a clear ocean surrounded by corrals. high resolution. 4k',
    'dolphins jumping in the ocean. high resolution. 4k',
    'a cute fox in a green forrest with blue eyes. high resolution. 4k',
    'a pack of wolfs howling at the moon. high resolution. 4k',
    'a giraffe eating leavs from a tree on the savannah. high resolution. 4k',
    'a chameleon in the rain forrest. high resolution. 4k',
    'a cute koala in the tree. high resolution. 4k',
    'a cute baby sheep on a sunny field of grass. high resolution. 4k',
    'a colorful flamingo in its natural habbitat. high resolution. 4k',
    'a beautiful swan. high resolution. 4k',
    'a beautiful colorful peacock. high resolution. 4k',
    'earth after human extinction, a new beginning, nature taking back the planet. high resolution. 4k',
    'a starry night. high resolution. 4k', 
    'northen lights with mountains in the background. high resolution. 4k', 
    'a beautiful mountain. high resolution. 4k',
    'a deep forrest. high resolution. 4k',
    'A cat wearing a top hat and monocle, sitting at a desk with a quill pen. high resolution. 4k',
    'A peaceful lake surrounded by mountains. high resolution. 4k',
    'A vast and majestic desert landscape. high resolution. 4k',
    'A tropical rainforest with exotic plants and animals. high resolution. 4k',
    'A waterfall cascading down a cliff face. high resolution. 4k',
    'A serene and picturesque meadow. high resolution. 4k',
    'A snowy winter wonderland with towering evergreens. high resolution. 4k',
    'A serene ocean sunset with brilliant colors. high resolution. 4k',
    'A tranquil autumn forest with colorful leaves falling. high resolution. 4k',
    'A serene and quiet countryside with rolling hills and farms. high resolution. 4k',   
    'A magical garden with colorful flowers and water features. high resolution. 4k',
    'a teapot in space. high resolution. 4k',
    'a phone booth underwater. high resolution. 4k',
    'A bicycle on a desert island. high resolution. 4k',
    'A bookshelf in a spaceship. high resolution. 4k',
    'A coffee mug on top of a mountain. high resolution. 4k',
    'A garden gnome in the middle of a busy city street. high resolution. 4k',    
    'A laptop in a treehouse. high resolution. 4k',
    'A hat on the ocean floor. high resolution. 4k',
    'A pencil sharpener in the Arctic tundra. high resolution. 4k',
    'A traffic light in a jungle. high resolution. 4k',
    'A mailbox in the middle of a desert. high resolution. 4k',   
    'A rubber duck in a bathtub on top of a skyscraper. high resolution. 4k',
    'A beautiful sunset over the ocean. high resolution. 4k',
    'A cute puppy playing in the park. high resolution. 4k',
    'A majestic mountain range covered in snow. high resolution. 4k',
    'A busy street in a big city at night. high resolution. 4k',   
    'A colorful hot air balloon festival. high resolution. 4k',
    'A stunning waterfall in the middle of a forest. high resoluton. 4k',
    'A beautiful flower garden in full bloom. high resolution. 4k',
    'A futuristic city skyline at sunset. high resolution. 4k',
    'A vintage car driving down a winding road. high resolution. 4k',
    'A serene lake surrounded by mountains and trees. high resolution. 4k',
    'A dramatic thunderstorm. high resolution. 4k',
    'A cozy fireplace in a rustic cabin. high resolution. 4k',
    'A colorful street art mural in a vibrant city neighborhood. high resolution. 4k',
    'A snowy landscape. high resolution. 4k',
    'A tropical rainforest. high resolution. 4k',
    'A starry night sky. high resolution. 4k',
    'A beautiful botanical garden. high resolution. 4k',
    'A beautiful national park. high resolution. 4k',
    'A scenic hiking trail. high resolution. 4k',
    'A breathtaking view from a mountain top. high resolution. 4k',
    'A colorful underwater coral reef. high resolution. 4k',
    'A group of llamas trekking through the mountains. high resolution. 4k',
    'A herd of elephants bathing in a river. high resolution. 4k',
    'A family of meerkats playing on a sun-drenched rock. high resolution. 4k',
    'A penguin colony waddling across an icy tundra. high resolution. 4k',
    'A group of flamingos gathered in a shallow lagoon. high resolution. 4k',
    'A playful pack of wolves howling at the moon. high resolution. 4k',
    'A majestic eagle soaring over a vast wilderness. high resolution. 4k',
    'A cute family of hedgehogs snuggled up in a cozy burrow. high resolution. 4k',
    'A group of monkeys swinging through the jungle canopy. high resolution. 4k',
    'A pair of mischievous otters frolicking in a stream. high resolution. 4k',
    'A stunning rainbow over a peaceful countryside. high resolution. 4k',
    'A charming family of ducks paddling in a tranquil pond. high resolution. 4k',
    'A group of sea turtles swimming in crystal clear waters. high resolution. 4k',   
    'A friendly pod of dolphins jumping and playing in the ocean. high resolution. 4k',
    'A gorgeous rainbow lorikeet perched on a colorful flower. high resolution. 4k',
    'A curious red panda peeking out from behind a tree. high resolution. 4k',
    'A group of kangaroos bounding across a golden savannah. high resolution. 4k',
    'A majestic lion relaxing in the shade of a tree. high resolution. 4k',
    'A family of bears fishing in a cool mountain stream. high resolution. 4k',
    'A playful family of river otters sliding down a waterfall. high resolution. 4k',
    'A field of sunflowers with smiling faces. high resolution. 4k',
    'A forest of mushrooms and fairies. high resolution. 4k',
    'A hot air balloon safari over the savannah. high resolution. 4k',
    'A magical ice castle surrounded by snow-covered mountains. high resolution. 4k',
    'A floating island in the clouds. high resolution. 4k',
    'A beach with palm trees and penguins. high resolution. 4k',
    'A garden maze with oversized plants and animals. high resolution. 4k',
    'A forest with giant mushrooms and snails. high resolution. 4k',
    'A winter wonderland with gingerbread houses and candy canes. high resolution. 4k',
    'A peaceful river with a cityscape made of legos. high resolution. 4k',
    'A futuristic city with flying cars and robots. high resolution. 4k',
    'A jungle with a treehouse village and waterfalls. high resolution. 4k',
    'A secret garden with hidden tunnels and enchanted creatures. high resolution. 4k',
    'A panda riding a unicycle through a field of strawberries. high resolution. 4k',
    'A spaceship made of cupcakes flying through a galaxy of donuts. high resolution. 4k',
    'A mermaid knitting a scarf in a coral reef. high resolution. 4k',
    'A robot dancing in a disco ball forest. high resolution. 4k',
    'A giant octopus playing the guitar on top of a mountain. high resolution. 4k',
    'A group of aliens having a tea party on the moon. high resolution. 4k',
    'A giant snail racing against a hare through a city street. high resolution. 4k',
    'A group of ghosts having a party in a haunted mansion. high resolution. 4k',
    'A chicken with a top hat and monocle walking through a garden of giant mushrooms. high resolution. 4k',
    'A ninja training in a field of sunflowers. high resolution. 4k',   
    'A surreal landscape with floating islands and upside-down trees. high resolution. 4k',
    'A colorful underwater world with talking seashells and dancing mermaids. high resolution. 4k',
    'A futuristic city with flying cars and neon skyscrapers. high resolution. 4k',
    'A magical forest with glowing mushrooms and friendly dragons. high resolution. 4k',
    'An enchanted garden with oversized flowers and a tea party hosted by talking animalshigh resolution. 4k',
    'A retro arcade with pixelated characters and glowing machines. high resolution. 4k',
    'A beautiful sunset over the ocean. high resolution. 4k',
    'A cozy cabin in the woods. high resolution. 4k',
    'A colorful bouquet of flowers. high resolution. 4k',
    'A majestic mountain range. high resolution. 4k',
    'A hot air balloon in the middle of a snowy forest. high resolution. 4k',
    'A piano on a floating platform in the middle of the ocean. high resolution. 4k',
    'A vintage car parked in the middle of a wheat field. high resolution. 4k',
    'A teapot and teacup on a giant mushroom in a magical forest. high resolution. 4k',
    'A giant pencil writing on a city skyline. high resolution. 4k',
    'A giant paper airplane flying over a mountain range. high resolution. 4k',
    'A rowboat in a sea of flowers. high resolution. 4k',
    'A grand piano on a cliff overlooking the ocean. high resolution. 4k',
    'A typewriter in a field of wildflowers. high resolution. 4k',
    'A vintage bicycle in the middle of a desert. high resolution. 4k',
    'A pair of sunglasses on a cactus in a canyon. high resolution. 4k',
    'A retro camera in a modern cityscape. high resolution. 4k',
    'A deck of cards on a snow-covered mountain peak. high resolution. 4k',
    'A polaroid camera on a beach with palm trees and turquoise waters. high resolution. 4k',
    'A top hat on a street lamp in a bustling city. high resolution. 4k',
    'A flamingo standing on a snow-covered mountain. high resolution. 4k',
    'A group of penguins swimming in a tropical ocean. high resolution. 4k',
    'A giraffe standing on a city street. high resolution. 4k',
    'A tiger lounging in a field of sunflowers. high resolution. 4k',
    'A panda eating bamboo on a city rooftop. high resolution. 4k',
    'A group of monkeys swinging from a city skyscraper. high resolution. 4k',
    'A kangaroo jumping over a city skyline. high resolution. 4k',
    'A group of dolphins swimming in a desert oasis. high resolution. 4k',
    'A lion walking on a snowy mountain ridge. high resolution. 4k',
    'A koala sitting on a street sign in a busy city. high resolution. 4k',
    'A group of elephants walking through a snow-covered forest. high resolution. 4k',
    'A baby elephant playing in a city fountain. high resolution. 4k',
    'A group of bunnies hopping through a city park. high resolution. 4k',
    'A kitten playing with a butterfly in a desert oasis. high resolution. 4k',
    'A beautiful colorfull butterfly. high resolution. 4k',
    'A panda eating ice cream on a city sidewalk. high resolution. 4k',
    'A group of hedgehogs exploring a snowy forest. high resolution. 4k',
    'A raccoon peeking out from a city sewer grate. high resolution. 4k',
    'A koala sleeping in a city hammock. high resolution. 4k',
    'A group of ducklings swimming in a city pond. high resolution. 4k',
    'A baby fox playing with a city street vendor. high resolution. 4k',
    'A squirrel eating a city hot dog. high resolution. 4k',
    'A group of meerkats popping out from a city storm drain. high resolution. 4k',
    'A baby bear playing with a city traffic cone. high resolution. 4k',
    'A group of otters playing on a city beach. high resolution. 4k',
    'A baby deer napping in a city park. high resolution. 4k',
    'A fox in a city alleyway with graffiti art. high resolution. 4k',
    'A group of ducks swimming in a city fountain with skyscrapers in the background. high resolution. 4k',
    'A raccoon stealing a slice of pizza on a city street. high resolution. 4k',
    'A baby deer peeking out from a city storm drain. high resolution. 4k',
    'A squirrel with a city skyline in the background. high resolution. 4k',
    'A baby fox hiding in a city park bush. high resolution. 4k',
    'A group of bunnies playing in a city flower bed. high resolution. 4k',
    'A baby elephant exploring a city alleyway. high resolution. 4k',
    'A surrealist painting of a city skyline with floating elephants. high resolution. 4k',
    'A sculpture garden filled with giant insects and animals. high resolution. 4k',
    'A mural of a whimsical forest with colorful animals and plants. high resolution. 4k',
    'A street artist painting a city wall with a giant octopus. high resolution. 4k',
    'A gallery filled with paintings of animals in surreal settings. high resolution. 4k',
    'A street performer dressed as a giant bird playing music in a city park. high resolution. 4k',
    'A city bridge decorated with colorful mosaics of animals and plants. high resolution. 4k',
    'A giant metal sculpture of a fantastical creature in a city plaza. high resolution. 4k',
    'A city park with sculptures of oversized fruit and vegetables. high resolution. 4k',
    'A neon art installation featuring animals in motion. high resolution. 4k',
    'A city museum with a collection of ancient animal-inspired artifacts. high resolution. 4k',
    'A city street lined with giant flower sculptures. high resolution. 4k',
    'An art installation of a giant glowing jellyfish in a city aquarium. high resolution. 4k',
    'A reimagining of "Starry Night" by Vincent van Gogh with a city skyline. high resolution. 4k',
    'A reinterpretation of "Girl with a Pearl Earring" by Johannes Vermeer with a modern twist. high resolution. 4k',
    'An abstract interpretation of "The Persistence of Memory" by Salvador Dali with a futuristic twist. high resolution. 4k',
    'A pop art version of "Campbells Soup Cans" by Andy Warhol with different flavors. high resolution. 4k',
    'A whimsical version of "The Great Wave off Kanagawa" by Hokusai with animals surfing the waves. high resolution. 4k',
    'A surreal interpretation of "The Birth of Venus" by Sandro Botticelli with a celestial twist. high resolution. 4k',
    'A modern version of "The Scream" by Edvard Munch with a city skyline in the background. high resolution. 4k',
    'A digital art version of "The Garden of Earthly Delights" by Hieronymus Bosch with interactive elements. high resolution. 4k',
    'A street art interpretation of "The Kiss" by Gustav Klimt with a city wall as the canvas. high resolution. 4k',
    'A contemporary version of "Water Lilies" by Claude Monet with a minimalist approach. high resolution. 4k',
    'A graffiti version of "Mona Lisa" by Leonardo da Vinci with a bold twist. high resolution. 4k',
    'An animated version of "The Starry Night" by Vincent van Gogh with moving elements. high resolution. 4k',
    'A comic book version of "The Last Supper" by Leonardo da Vinci with a modern cast of characters. high resolution. 4k',
    'A digital collage version of "Les Demoiselles dAvignon" by Pablo Picasso with different textures and elements. high resolution. 4k',
    'A street art version of "The Son of Man" by Rene Magritte with a city wall as the canvas and graffiti elements. high resolution. 4k',
    'A playful puppy running through a field of flowers. high resolution. 4k',
    'A majestic cat perched on a tree branch, overlooking the city. high resolution. 4k',
    'A dog dressed in a superhero costume, ready to save the day. high resolution. 4k',
    'A cozy scene of a cat curled up by the fireplace, with a mug of hot cocoa nearby. high resolution. 4k',
    'A mischievous kitten tangled up in a ball of yarn. high resolution. 4k',
    'A dog enjoying a day at the beach, playing fetch with a frisbee. high resolution. 4k',
    'A cat in a bow tie, ready for a fancy night out. high resolution. 4k',
    'A dog and cat cuddled up together, taking a nap in the sun. high resolution. 4k',
    'A dog in a raincoat and boots, ready for a walk in the rain. high resolution. 4k',
    'A cat sitting in a teacup, enjoying a cup of tea. high resolution. 4k',
    'A dog dressed up as a cowboy, ready to ride into the sunset. high resolution. 4k',
    'A cat lounging in a hammock, enjoying a lazy day in the sun. high resolution. 4k',
    'A dog and cat sharing a plate of spaghetti, a la Lady and the Tramp. high resolution. 4k',
    'A fluffy cat with a flower crown, posing for a photoshoot in a garden. high resolution. 4k',
    'An abandoned castle in the middle of a dense forest. high resolution. 4k',
    'A misty morning in the mountains, with the sun rising over the peaks. high resolution. 4k',
    'A majestic bald eagle soaring high above a pristine lake. high resolution. 4k',
    'A rustic cabin nestled in the woods, surrounded by towering trees and a babbling brook. high resolution. 4k',
    'A field of colorful wildflowers stretching out as far as the eye can see. high resolution. 4k',
    'A dramatic rocky coastline, with crashing waves and rugged cliffs. high resolution. 4k',
    'A quiet forest trail, winding through tall trees and mossy boulders. high resolution. 4k',
    'A cascading waterfall hidden deep in the forest, with a rainbow of mist in the air. high resolution. 4k',
    'A panoramic view of the night sky, with the Milky Way stretching overhead. high resolution. 4k',
    'A towering mountain peak, surrounded by a sea of clouds. high resolution. 4k',
    'A tranquil meadow with grazing horses and a backdrop of mountains in the distance. high resolution. 4k',
    'A colorful autumn forest, with vibrant reds, oranges, and yellows on the trees. high resolution. 4k',
    'A pristine alpine lake, reflecting the snow-capped peaks above. high resolution. 4k',
    'A sun-dappled forest glade, with a gentle stream running through it. high resolution. 4k',
    'A rugged desert landscape, with towering sand dunes and stunning rock formations. high resolution. 4k',
    'A vintage typewriter on a deserted beach. high resolution. 4k',
    'A grand piano in the middle of a forest clearing. high resolution. 4k',
    'A sparkling chandelier in an abandoned warehouse. high resolution. 4k',
    'A rusty bicycle in the middle of a dried-up lake bed. high resolution. 4k',
    'A stack of old books on a sunken ship. high resolution. 4k',
    'A grandfather clock in an overgrown garden. high resolution. 4k',
    'A row of colorful umbrellas on a barren hillside. high resolution. 4k',
    'A vintage camera in the middle of a bustling city street. high resolution. 4k',
    'A collection of antique keys in an abandoned mansion. high resolution. 4k',
    'A crystal vase on a deserted mountaintop. high resolution. 4k',
    'The Mona Lisa with a modern twist. high resolution. 4k',
    'An abstract painting of a stormy sea. high resolution. 4k',
    'A surrealistic portrait of a melting clock. high resolution. 4k',
    'A minimalist interpretation of a city skyline. high resolution. 4k',
    'An impressionist painting of a field of sunflowers. high resolution. 4k',
    'A cubist still life of a fruit bowl. high resolution. 4k',
    'A pointillist landscape of a mountain range. high resolution. 4k',
    'A portrait of a pet in the style of a classic portrait. high resolution. 4k',
    'A nebula with a hidden surprise. high resolution. 4k',
    'A black hole that defies all laws of physics. high resolution. 4k',
    'A supernova exploding in brilliant colors. high resolution. 4k',
    'A comet with a long, sparkling tail. high resolution. 4k',
    'A planet with rings that shine like diamonds. high resolution. 4k',
    'A galaxy with a unique shape. high resolution. 4k',
    'A meteor shower lighting up the night sky. high resolution. 4k',
    'An asteroid belt with hidden treasures. high resolution. 4k',
    'A constellation with a secret meaning. high resolution. 4k',
    'A wormhole leading to another dimension. high resolution. 4k',
    'A beaturiful colorful bird on a tree branch. high resolution. 4k',
    'A solar system with multiple suns. high resolution. 4k',
    'A pulsar emitting rhythmic bursts of energy. high resolution. 4k',
    'A quasar with an unfathomable energy output. high resolution. 4k',
    'The northern lights dancing over a mountain range. high resolution. 4k',
    'A fjord with crystal-clear water reflecting the surrounding peaks. high resolution. 4k',
    'A cabin in the woods surrounded by snow-covered trees. high resolution. 4k',
    'A rocky coastline with crashing waves and a lighthouse. high resolution. 4k',
    'A waterfall cascading down a steep cliff. high resolution. 4k',
    'A herd of reindeer grazing on a snowy plain. high resolution. 4k',
    'A glacier carving through a mountain valley. high resolution. 4k',
    'A night sky filled with stars and the aurora borealis. high resolution. 4k',
    'A doggo taking a nap on a pile of laundry. high resolution. 4k',
    'A kitty sleeping in a cardboard box that is clearly too small for her. high resolution. 4k',
    'A dog with a guilty look and a chewed-up slipper in his mouth. high resolution. 4k',
    'A cat perched on a windowsill, looking down on the world below. high resolution. 4k',
    'A pup with a goofy grin and his tongue hanging out. high resolution. 4k',
    'A kitty playing with a ball of yarn, looking like she is plotting her next move. high resolution. 4k',
    'A doggo wearing a hilarious costume, looking both embarrassed and adorable. high resolution. 4k',
    'A cat with a disapproving stare, judging her humans every move. high resolution. 4k',
    'A dog looking out the car window, his tongue flapping in the wind. high resolution. 4k',
    'A kitty lounging on a sun-soaked windowsill, living her best life. high resolution. 4k',
    'A pup trying to catch his tail, failing miserably but having a blast. high resolution. 4k',
    'A cat caught mid-meow, looking like she is singing an operatic aria. high resolution. 4k',
    'A dog snuggling with his favorite stuffed animal, looking like a toddler with a blankie. high resolution. 4k',
    'A kitty perched on top of a bookshelf, looking like she is ruling the world. high resolution. 4k',
    'A pup wearing a pair of glasses, looking both nerdy and adorable. high resolution. 4k',
    'a beautiful rainbow. high resolution. 4k',
    'A penguin in a top hat and monocle riding a unicycle. high resolution. 4k',
    'A banana wearing sunglasses and holding a boombox. high resolution. 4k',
    'A llama riding a skateboard and doing tricks. high resolution. 4k',
    'A robot drinking coffee and reading a newspaper. high resolution. 4k',
    'A sloth dressed as a superhero, but moving in slow motion. high resolution. 4k',
    'A turtle riding a bicycle and wearing a helmet. high resolution. 4k',
    'A dragon flying over a city and taking a selfie. high resolution. 4k',
    'A koala driving a race car and waving to the crowd. high resolution. 4k',
    'A squirrel playing the guitar and singing a love song to a nut. high resolution. 4k',
    'A potato wearing glasses and holding a book, trying to look smart. high resolution. 4k',
    'A flamingo in a business suit attending a meeting. high resolution. 4k',
    'A koala holding a microphone and performing stand-up comedy. high resolution. 4k',
    'A pineapple wearing a sombrero and playing a guitar. high resolution. 4k',
    'A giraffe wearing a tutu and ballet shoes. high resolution. 4k',
    'A donut wearing a cape and fighting crime. high resolution. 4k',
    'A robot playing chess against a plant. high resolution. 4k',
    'A hamster dressed up as a cowboy, riding a miniature horse. high resolution. 4k',
    'A monkey wearing a party hat and blowing out candles on a birthday cake. high resolution. 4k',
    'A frog wearing a bow tie and a monocle, sitting at a grand piano. high resolution. 4k',
    'A walrus dressed in a tuxedo, walking down a red carpet. high resolution. 4k',
    'A robot bartender serving drinks to a group of aliens. high resolution. 4k',
    'A platypus in a superhero costume, ready to save the day. high resolution. 4k',
    'A secret garden with a mysterious door. high resolution. 4k',
    'A deserted mansion with a dark past. high resolution. 4k',
    'A cactus with a cowboy hat riding a unicycle. high resolution. 4k',
    'A pineapple playing the guitar on a beach. high resolution. 4k',
    'A zebra reading a book in a library filled with other animals. high resolution. 4k',
    'A koala drinking tea with a group of flamingos. high resolution. 4k',
    'A robot doing yoga on a mountaintop. high resolution. 4k',
    'A slice of pizza running a marathon. high resolution. 4k',
    'A giraffe wearing a top hat and monocle, smoking a pipe. high resolution. 4k',
    'A unicorn riding a bicycle in a city street. high resolution. 4k',
    'A fish playing the piano in an underwater concert hall. high resolution. 4k',
    'A snail driving a race car with a trail of slime behind it. high resolution. 4k',
    'A panda dressed as a superhero, flying through the sky on a magic carpet. high resolution. 4k',
    'A sheep riding a rollercoaster while eating cotton candy. high resolution. 4k',
    'A shark playing the guitar at a rock concert, surrounded by a group of cheering fish fans. high resolution. 4k',
    'A penguin riding a skateboard while drinking a cup of hot chocolate. high resolution. 4k',
    'A frog wearing a crown and sitting on a throne made of leaves. high resolution. 4k',
    'A bunny riding a unicorn through a rainbow-colored forest.  high resolution. 4k',
    'A platypus dressed in a business suit, sitting at a desk and typing on a computer. high resolution. 4k',
    'A monkey dressed as a pirate, searching for treasure on a deserted island. high resolution. 4k',
    'A walrus doing a handstand on an ice floe, wearing a tutu and a tiara. high resolution. 4k',
    'A giant teapot pouring a river of tea into a cup. high resolution. 4k',
    'A rainbow bridge leading to a castle made of clouds. high resolution. 4k',
    'A fishbowl with a miniature city built inside. high resolution. 4k',
    'A mountain range at sunset with a flock of birds flying across the sky. high resolution. 4k',
    'A field of wildflowers with a rainbow stretching across the horizon. high resolution. 4k',
    'A starry night sky with a comet streaking across it. high resolution. 4k',
    'A waterfall cascading down into a crystal-clear pool. high resolution. 4k',
    'A forest with trees that have leaves made of different colors of stained glass. high resolution. 4k',
    'A lake with a floating island made of glowing crystals. high resolution. 4k',
    'A garden with flowers that have bioluminescent petals. high resolution. 4k',
    'A beach with a lighthouse shining its beam across the water. high resolution. 4k',
    'A desert landscape with sand dunes shaped like waves. high resolution. 4k',
    'An aurora borealis over a snowy mountain range. high resolution. 4k', 
    'A close-up of a snowflake with intricate patterns and details. high resolution. 4k', 
    'A close-up of a butterfly wing with vibrant colors and delicate textures. high resolution. 4k', 
    'A close-up of a crystal with sharp edges and transparent clarity. high resolution. 4k', 
    'A close-up of a peacock feather with vibrant hues and intricate details. high resolution. 4k', 
    'A close-up of a dewdrop on a blade of grass with reflections of the surroundings. high resolution. 4k', 
    'A close-up of a firework exploding in the sky with sparks and colorful lights. high resolution. 4k', 
    'A close-up of a seashell with spiral patterns and glossy texture. high resolution. 4k', 
    'A close-up of a tree bark with rough textures and patterns. high resolution. 4k', 
    'A close-up of a flower petal with intricate details and gradient colors. high resolution. 4k', 
    'A close-up of a cactus spines with needle-like details and textures. high resolution. 4k', 
    'A close-up of a hummingbird feathers with iridescent colors and delicate textures. high resolution. 4k', 
    'A close-up of a coral reef with vibrant colors and intricate details. high resolution. 4k', 
    'A close-up of a butterfly antenna with delicate textures and patterns. high resolution. 4k', 
    'A close-up of a mushroom with unique shapes and patterns. high resolution. 4k', 
    'A close-up of a dragonfly wing with transparent details and vibrant colors. high resolution. 4k', 
    'A close-up of a pebble with smooth curves and unique textures. high resolution. 4k', 
    'A close-up of a fern leaf with intricate patterns and shades of green. high resolution. 4k', 
    'A close-up of a dragonfruit with bright colors and interesting textures. high resolution. 4k', 
    'A close-up of a snow-capped mountain peak with jagged edges and icy textures. high resolution. 4k', 
    'A droplet of dew clings to a blade of grass. style: high resolution, 4k'
]


      
        random.shuffle(prompts)
    
        #generating images
        for prompt in prompts[10]:
            print(prompt)
            diffusion_noise = tf.random.normal((batch_size, img_height//8, img_width//8, 4))
            img, latent, prompt = gen_image(prompt, diffusion_noise, batch_size, num_steps, unconditional_guidance_scale)
            Image.fromarray(img[0]).save(f"{population_folder}/{i}.png") 
            hashtag = generate_hashtags_from_prompt(prompt)
            
            posts_dict.append({'num' : i, 'fitness' : 0, 'latent': latent, 'prompt': prompt, 'hashtag': hashtag})
             
         
            i+=1
            
        print('done creating')
    
        with open(f'logs/posts_dict_{iteration}.txt', 'wb') as f:
            pickle.dump(posts_dict, f)
        
    else:
        print("files in init population. Not update selected")
        
        
#parameters
def main(iteration, update, clear, num, gs, ngs): 
    img_height = 512
    img_width = 512
    batch_size = 1        
    num_steps = num        
    unconditional_guidance_scale =gs
    new_scale = ngs
    update_initial_population=update       
    
    pop_images_path = 'prompt_bank/pop_images'
    
    mutation_probability = 0.1
    crossover_probability = 0.8
    style_change=0.5
    iteration = int(iteration)    
    
    history_avg_score = []

    go = True
    
    #updating pop_images folder
    #update_pop_images() 

    print(f'starting on iteration: {iteration}')
    
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

    
    

    if iteration==0:
        
      #updating initial population
      update_init_population(update_initial_population, img_height, img_width, batch_size, num_steps, new_scale)
      
      #has return in iteration 0 
      if path.exists('return/posts_dict_0_return.txt'):
          print('have return')
          go = True
      
      else:
        print('population 0 ready.. Get fitness for iteration 0..')
        go = False

    
    while go:
        
        #get current population
        with open(f'logs/posts_dict_{iteration}.txt', 'rb') as handle:
            posts_dict = pickle.loads(handle.read())
        
        population_folder = f'generated_images/population_{iteration}'
        if not os.path.exists(population_folder):
            os.makedirs(population_folder)

        next_iteration = iteration + 1
        next_population_folder = f'generated_images/population_{next_iteration}'
        if not os.path.exists(next_population_folder):
            os.makedirs(next_population_folder)

  

        fitness_dict_name = f'return/posts_dict_{iteration}_return.txt'
        #Get dict containing fitness from instagram
        #get current population
        try:
          with open(fitness_dict_name, 'rb') as handle:
            fitness_dict = pickle.loads(handle.read())
        except:
            print(f'missing return dict for iteration {iteration}')
            go = False

        #updating fitness
        for i in range(len(posts_dict)):
          posts_dict[i]['fitness'] = fitness_dict[i]['fitness']


        #calculating average fitness of population
        avg_score = 0
        for post in posts_dict:
          avg_score += post['fitness']
        avg_score /= len(posts_dict)
        
        print('Iteration', iteration, ', avg. population score = ', round(avg_score, 2))
        
        timestamp = date.today()
        timestamp = timestamp.strftime("%Y-%m-%d")
        log_file = f'logs/log_{timestamp}.txt'
        
        with open(log_file, 'a') as f:
          f.write(f'\niteration: {iteration}')
          f.write(f'\navg score: {avg_score}')
          f.write('\n######################################') 
            
        '''
        creating new population

        '''
        
        
        '''
        
        prompts_new = ['A high-resolution (4K) image of a Gothic cathedral with intricate stonework and towering spires, shrouded in mist and surrounded by autumn leaves.',


'A high-resolution (4K) photograph of a bustling city street at night, with bright neon signs, reflections in rain-soaked pavement, and a sense of energy and movement.',
'A surreal digital painting of a dreamlike landscape, with floating islands, strange creatures, and vibrant colors.',
'A detailed pencil drawing of a majestic eagle in flight, with precise shading and intricate feather patterns.',
'A minimalist geometric design with clean lines and bold colors, inspired by the Art Deco style of the 1920s.',
'A high-resolution (4K) photograph of a panoramic mountain vista at sunset, with warm orange and pink hues and a sense of vastness and grandeur.',

'An abstract expressionist painting with bold brushstrokes and vibrant colors, inspired by the work of Jackson Pollock or Mark Rothko.',
'A high-resolution (4K) photograph of a misty forest with sunrays shining through the trees, with a sense of calmness and tranquility.',
'A digital painting of a futuristic cityscape, with sleek buildings and flying cars in a neon-lit night sky, inspired by the cyberpunk genre.',
'A watercolor portrait of a majestic lion, with a dramatic color palette and expressive brushstrokes capturing the animal fierce spirit.',
'A high-resolution (4K) photograph of a serene lake at dawn, with a misty fog over the water and a colorful sky.',

'A surreal collage of different elements, such as a bird, a cityscape, and a clock, with unexpected combinations and textures.',

'An impressionistic painting of a field of lavender, with soft brushstrokes and a dreamy atmosphere, inspired by the work of Monet.',
'A digital art piece with an abstract and geometric design, with bold colors and shapes, and a futuristic feel.',

 'A serene and peaceful watercolor painting of a lake at sunset, with warm colors and a dreamy atmosphere, capturing the beauty of nature.', 
 'A minimalist and abstract digital art piece with geometric shapes and vibrant colors, inspired by the art of the Bauhaus movement.', 
 'A detailed charcoal drawing of a mountain landscape, with intricate textures and a sense of majesty and grandeur.', 
 'A surreal and imaginative digital collage of different elements, such as a whale in the sky, a castle in the clouds, and a tree with books as leaves.', 
 'A high-resolution (4K) photograph of a snow-covered mountain peak at sunrise, with a golden light and a sense of peace and stillness.', 
'An impressionistic painting of a bustling city street at night, with bright lights and a sense of energy and movement.',
 'A hyper-realistic oil painting of a bouquet of flowers, with intricate details and textures that make each petal and leaf come to life.', 
 'A digital art piece with a retro-futuristic design, with bold lines, bright colors, and a sci-fi feel.', 
'An impressionistic oil painting of a seascape, with vibrant colors and loose brushstrokes that capture the energy of the waves.', 
'A detailed graphite drawing of a squirrel holding a tiny umbrella, with a whimsical and humorous touch.', 
'A hyper-realistic acrylic painting of a chameleon changing colors to blend in with its surroundings, with intricate details and a sense of camouflage.', 
'A hyper-realistic oil painting of a hummingbird hovering in mid-air, with intricate details and textures that capture the bird beauty and agility.',
'A charming digital illustration of a pug wearing a tiny bow tie, with a whimsical and humorous touch.',
'A hyper-realistic graphite drawing of a dog wearing a pair of glasses, with intricate details and a sense of intelligence.',

'A digital art piece with a whimsical and creative design, featuring a group of frogs wearing superhero costumes and saving the day.',

'A hyper-realistic oil painting of a koala wearing a bow tie and holding a cup of tea, with a sense of sophistication and elegance.',


'A serene watercolor painting of a group of birds perched on a cherry blossom tree, with delicate details and a sense of tranquility and beauty.',
'A hyper-realistic graphite drawing of a wolf howling at the full moon, with intricate details and a sense of wildness and freedom.',
'A digital art piece with a creative and imaginative design, featuring a group of butterflies flying over a field of sunflowers, with a sense of joy and playfulness.',
'A breathtaking oil painting of a flock of flamingos flying over a crystal clear lake, with a sense of elegance and beauty.',

'A hyper-realistic graphite drawing of a bald eagle perched on a tree branch, with intricate details and a sense of power and strength.',

'A digital illustration of a group of foxes playing in a snowy forest, with a sense of playfulness and curiosity.',
'A mesmerizing watercolor painting of a butterfly emerging from its cocoon, with delicate details and a sense of transformation and beauty.',

'A fluffy white kitten napping on a fluffy pink cloud, with a starry night sky in the background.',

'A playful red fox frolicking in a field of golden wheat, with a majestic mountain range looming in the distance.',

'A mesmerizing scene of a planet orbiting a distant star, with swirling nebulas and a glowing aurora in the background.',

'A stunning view of a galaxy with millions of stars, with a glowing blue planet in the foreground.',

'An awe-inspiring 4K time-lapse of a starry night sky, with the Milky Way galaxy shining brightly and shooting stars streaking across the darkness.',

'A surreal 4K time-lapse of a thunderstorm, with lightning bolts illuminating the dark clouds and rain pouring down in sheets.',

'A dreamlike 4K shot of a magical forest, with sunlight streaming through the trees and mist hovering above the ground, creating an enchanted atmosphere.',

'A stunning 4K shot of a rainbow, arching across a dramatic landscape with storm clouds and rays of sunshine, creating a magical atmosphere.',
'An incredible 4K time-lapse of a sunset, with warm colors filling the sky and casting a golden light on the landscape below.',
'A breathtaking 4K aerial view of a turquoise lake, surrounded by majestic snow-capped mountains and a forest of evergreen trees.',

'A magnificent 4K time-lapse of a full moon rising over a scenic landscape, with its soft glow illuminating the beauty of the night.',
'An awe-inspiring 4K aerial view of a city skyline at night, with colorful lights and reflections creating a vibrant and dynamic scene.',
'A breathtaking 4K shot of a mountain lake, with crystal-clear water and snow-capped peaks reflecting in its surface, surrounded by evergreen trees.',
'A dreamlike 4K time-lapse of a field of stars, with constellations and shooting stars moving across the night sky in a mesmerizing',
'A beautiful oil painting of a lush forest in the autumn, with vibrant colors and a peaceful atmosphere.',
'A hauntingly beautiful photograph of an abandoned train station, with the decay and rust adding to the eerie atmosphere of the scene.',
'A digital art piece featuring a futuristic robot exploring an alien landscape, with vibrant colors and a sense of adventure.',
'A surreal digital art piece featuring a giant rabbit in a field of flowers, with a dreamlike feel and vibrant colors.',


'A group of adorable puppies playing in a 4K field of flowers, with butterflies fluttering around them.',

'A beautifully detailed 3D rendering of a chameleon resting on a branch, with intricate scales and textures that come to life in high definition.',

'A captivating and otherworldly painting of a group of whales swimming in a sea of stars, with a dreamy and surreal atmosphere that invites the viewer to escape into a magical world.',

'A beautiful and serene photograph of a hummingbird perched on a flower, captured in high definition with stunning detail and clarity.',

'A stunning 4K time-lapse of a field of sunflowers at sunrise, with vibrant colors and a mesmerizing movement that captures the beauty of nature in motion.',

'A dramatic and breathtaking photograph of a lightning storm over a city skyline, captured in high definition with vivid colors and stunning detail.',
'A charming and whimsical illustration of a group of owls gathered in a tree, with colorful hats and scarves and a cozy and inviting atmosphere.',


'A mesmerizing and hypnotic animation of a kaleidoscope of colors and shapes, with a sense of movement and fluidity that captures the eye.',


'A vibrant and colorful illustration of a parrot',

'A weathered, wooden rowboat floating on a misty, tranquil lake, surrounded by towering trees and mountains.',


'A rustic bicycle leaning against a tree in a field of sunflowers, with the sun setting behind it and casting a golden glow.',
'A vintage camera sitting on a windowsill, overlooking a bustling cityscape at night, with the neon lights and skyscrapers reflecting in its lens.',

'A hilarious photograph of a tiny pug dressed in a full suit and tie, sitting at a desk and pretending to work on a computer, with a serious expression on its face.',

'A playful photograph of a happy dog wearing a superhero cape and mask, flying through the air with a cityscape in the background.'


    'Pop Art Style: This cat is a walking work of art with its vibrant colors and bold stripes.',
    'Watercolor Style: A majestic elephant splashes through a colorful watercolor landscape.',
    'Minimalist Style: A lone penguin stands on a pristine ice floe, silhouetted against a glowing sunset.',
    'Retro Style: This dapper llama sports a polka-dot bow tie and shades straight out of the 80s.',
    'Cartoon Style: A mischievous raccoon peers out from behind a tree, eyes twinkling with mischief.'
    'Graffiti Style: This graffiti-style art piece features a gang of hip-hop hamsters, complete with bling and boomboxes.', 
    'a very cute cat', 
    'a very cute dog'
    
     'a cute fox in a green forrest with blue eyes. high resolution. 4k',
 'a teapot in space. high resolution. 4k',
'a phone booth underwater. high resolution. 4k',
'A coffee mug on top of a mountain. high resolution. 4k'
 'A mailbox in the middle of a desert. high resolution. 4k',
'A piano on a floating platform in the middle of the ocean. high resolution. 4k',
'A squirrel with a city skyline in the background. high resolution. 4k',
'A baby fox hiding in a city park bush. high resolution. 4k',
'A squirrel playing the guitar and singing a love song to a nut. high resolution. 4k',
'A frog wearing a bow tie and a monocle, sitting at a grand piano. high resolution. 4k',
'A platypus in a superhero costume, ready to save the day. high resolution. 4k',

' Cubism Style: This dalmatian is a masterpiece of geometric shapes and bright colors, like a living Picasso painting.',
'Gothic Style: A dark and mysterious raven perches on a crumbling stone tower, its feathers glinting in the moonlight.',
'Steampunk Style: A sleek and futuristic cheetah prowls through a gritty, industrial cityscape, its metallic joints gleaming in the neon glow.',
'Art Nouveau Style: A majestic peacock spreads its ornate feathers in a graceful display of art and nature, like a living sculpture.',
'Rococo Style: This fancy feline lounges on a plush velvet couch, surrounded by gilded decorations and opulent chandeliers, like a true aristocrat.',
'Street Art Style: A rebellious graffiti artist rabbit uses the city as its canvas, leaving a trail of vibrant colors and witty slogans in its wake.',
'Cyberpunk Style: A sleek and deadly panther stalks through a neon-lit alleyway, its augmented eyes scanning the darkness for prey.',

'Neo-Impressionist Style: A tranquil scene of a serene swan floating on a glittering lake, rendered in shimmering dots of vibrant color.',
'Postmodern Style: A meta masterpiece featuring a painting of a painting of a painting of a chicken, each one more abstract and absurd than the last.',
'Gothic Revival Style: A majestic black stallion gallops through a stormy landscape, its muscular form and flowing mane evoking the dark romance of classic literature.',
'Hyperrealist Style: A photorealistic rendering of a pudgy pugs wrinkled face, capturing every fold and crease in exquisite detail.',
'Afrofuturism Style: A futuristic elephant with glowing tusks and intricate tribal tattoos roams through a neon-lit cityscape, embodying the power and mysticism of African culture.',
'Cubist Pop Art Style: This funky flamingo is a playful combination of bright colors and jumbled shapes, like a living puzzle with a sense of humor.'


]
        '''
        
        prompts_new = [
'A high-resolution (4K) photograph of a panoramic mountain vista at sunset, with warm orange and pink hues and a sense of vastness and grandeur.',

'A high-resolution (4K) photograph of a misty forest with sunrays shining through the trees, with a sense of calmness and tranquility.',
'A digital painting of a futuristic cityscape, with sleek buildings and flying cars in a neon-lit night sky, inspired by the cyberpunk genre.',

'A high-resolution (4K) photograph of a serene lake at dawn, with a misty fog over the water and a colorful sky.',

'A surreal collage of different elements, such as a bird, a cityscape, and a clock, with unexpected combinations and textures.',

'An impressionistic painting of a field of lavender, with soft brushstrokes and a dreamy atmosphere, inspired by the work of Monet.',
'A digital art piece with an abstract and geometric design, with bold colors and shapes, and a futuristic feel.',

 'A serene and peaceful watercolor painting of a lake at sunset, with warm colors and a dreamy atmosphere, capturing the beauty of nature.', 
  
 'A detailed charcoal drawing of a mountain landscape, with intricate textures and a sense of majesty and grandeur.', 
 'A surreal and imaginative digital collage of different elements, such as a whale in the sky, a castle in the clouds, and a tree with books as leaves.', 
 'A high-resolution (4K) photograph of a snow-covered mountain peak at sunrise, with a golden light and a sense of peace and stillness.', 
'A detailed 4k high resolution image of a squirrel holding a tiny umbrella, with a whimsical and humorous touch.', 
'A hyper-realistic acrylic painting of a chameleon changing colors to blend in with its surroundings, with intricate details and a sense of camouflage.', 
'A hyper-realistic oil painting of a hummingbird hovering in mid-air, with intricate details and textures that capture the bird beauty and agility.',
'A charming 4k high resolution image of a pug wearing a tiny bow tie, with a whimsical and humorous touch.',
'A high resolution 4k image of a cute cat wearing cool sunglasses',
'A hyper-realistic image of a dog wearing a pair of glasses, with intricate details and a sense of intelligence.',

'A hyper-realistic oil painting of a koala wearing a bow tie and holding a cup of tea, with a sense of sophistication and elegance.',


'A serene watercolor painting of a group of birds perched on a cherry blossom tree, with delicate details and a sense of tranquility and beauty.',
'A hyper-realistic 4k image of a wolf howling at the full moon, with intricate details and a sense of wildness and freedom.',
'An art piece with a creative and imaginative design, featuring a group of butterflies flying over a field of sunflowers, with a sense of joy and playfulness.',
'A breathtaking oil painting of a flock of flamingos flying over a crystal clear lake, with a sense of elegance and beauty.',

'A 4k high resolution image of a fox with green eyes in a snowy forest, with a sense of playfulness and curiosity.',
'A mesmerizing watercolor painting of a butterfly emerging from its cocoon, with delicate details and a sense of transformation and beauty.',

'A fluffy white kitten napping on a fluffy pink cloud, with a starry night sky in the background.',

'A playful red fox frolicking in a field of golden wheat, with a majestic mountain range looming in the distance.',

'A mesmerizing scene of a planet orbiting a distant star, with swirling nebulas and a glowing aurora in the background.',

'A stunning view of a galaxy with millions of stars, with a glowing blue planet in the foreground.',

'An awe-inspiring 4K time-lapse of a starry night sky, with the Milky Way galaxy shining brightly and shooting stars streaking across the darkness.',

'A dreamlike 4K shot of a magical forest, with sunlight streaming through the trees and mist hovering above the ground, creating an enchanted atmosphere.',

'A stunning 4K shot of a rainbow, arching across a dramatic landscape with storm clouds and rays of sunshine, creating a magical atmosphere.',
'An incredible 4K time-lapse of a sunset, with warm colors filling the sky and casting a golden light on the landscape below.',
'A breathtaking 4K aerial view of a turquoise lake, surrounded by majestic snow-capped mountains and a forest of evergreen trees.',

'A magnificent 4K time-lapse of a full moon rising over a scenic landscape, with its soft glow illuminating the beauty of the night.',
'An awe-inspiring 4K aerial view of a city skyline at night, with colorful lights and reflections creating a vibrant and dynamic scene.',
'A breathtaking 4K shot of a mountain lake, with crystal-clear water and snow-capped peaks reflecting in its surface, surrounded by evergreen trees.',
'A dreamlike 4K time-lapse of a field of stars, with constellations and shooting stars moving across the night sky in a mesmerizing',
'A beautiful oil painting of a lush forest in the autumn, with vibrant colors and a peaceful atmosphere.',

'A surreal digital art piece featuring a giant rabbit in a field of flowers, with a dreamlike feel and vibrant colors.',

'An adorable puppy playing in a 4K field of flowers, with butterflies fluttering around him.',

'A beautifully detailed 3D rendering of a chameleon resting on a branch, with intricate scales and textures that come to life in high definition.',

'A captivating and otherworldly painting of a group of whales swimming in a sea of stars, with a dreamy and surreal atmosphere that invites the viewer to escape into a magical world.',



'A stunning 4K time-lapse of a field of sunflowers at sunrise, with vibrant colors and a mesmerizing movement that captures the beauty of nature in motion.',

'A charming and high quality illustration of an owl gathered in a tree, with colorful hat and scarf and a cozy and inviting atmosphere.',


'A vibrant and colorful illustration of a parrot with a hat. high resolution. 4k.',

'A weathered, wooden rowboat floating on a misty, tranquil lake, surrounded by towering trees and mountains.',


'A rustic bicycle leaning against a tree in a field of sunflowers, with the sun setting behind it and casting a golden glow.',
'A vintage camera sitting on a windowsill, overlooking a bustling cityscape at night, with the neon lights and skyscrapers reflecting in its lens.',

'A hilarious photograph of a tiny pug dressed in a full suit and tie, sitting at a desk and pretending to work on a computer, with a serious expression on its face.',

'A playful photograph of a happy dog wearing a superhero cape and mask, flying through the air with a cityscape in the background.'


    'Pop Art Style: This cat is a walking work of art with its vibrant colors and bold stripes.',
    'Watercolor Style: A majestic elephant splashes through a colorful watercolor landscape.',
    'Minimalist Style: A lone penguin stands on a pristine ice floe, silhouetted against a glowing sunset.',
    'Retro Style: This dapper llama sports a polka-dot bow tie and shades straight out of the 80s.',
    'Cartoon Style: A mischievous raccoon peers out from behind a tree, eyes twinkling with mischief.'
    'Graffiti Style: This graffiti-style art piece features a gang of hip-hop hamsters, complete with bling and boomboxes.', 
    'a very cute cat', 
    'a very cute dog'
    
     'a cute fox in a green forrest with blue eyes. high resolution. 4k',
 'a teapot in space. high resolution. 4k',
'a phone booth underwater. high resolution. 4k',
'A coffee mug on top of a mountain. high resolution. 4k'
 'A mailbox in the middle of a desert. high resolution. 4k',
'A piano on a floating platform in the middle of the ocean. high resolution. 4k',
'A squirrel with a city skyline in the background. high resolution. 4k',
'A squirrel playing the guitar and singing a love song to a nut. high resolution. 4k',
'A frog wearing a bow tie and a monocle, sitting at a grand piano. high resolution. 4k',
'A platypus in a superhero costume, ready to save the day. high resolution. 4k',

' Cubism Style: This dalmatian is a masterpiece of geometric shapes and bright colors, like a living Picasso painting.',
'Gothic Style: A dark and mysterious raven perches on a crumbling stone tower, its feathers glinting in the moonlight.',
'Steampunk Style: A sleek and futuristic cheetah prowls through a gritty, industrial cityscape, its metallic joints gleaming in the neon glow.',
'Art Nouveau Style: A majestic peacock spreads its ornate feathers in a graceful display of art and nature, like a living sculpture.',
'Rococo Style: This fancy feline lounges on a plush velvet couch, surrounded by gilded decorations and opulent chandeliers, like a true aristocrat.',
'Street Art Style: A rebellious graffiti artist rabbit uses the city as its canvas, leaving a trail of vibrant colors and witty slogans in its wake.',
'Cyberpunk Style: A sleek and deadly panther stalks through a neon-lit alleyway, its augmented eyes scanning the darkness for prey.',

'Neo-Impressionist Style: A tranquil scene of a serene swan floating on a glittering lake, rendered in shimmering dots of vibrant color.',
'Postmodern Style: A meta masterpiece featuring a painting of a painting of a painting of a chicken, each one more abstract and absurd than the last.',
'Gothic Revival Style: A majestic black stallion gallops through a stormy landscape, its muscular form and flowing mane evoking the dark romance of classic literature.',
'Hyperrealist Style: A photorealistic rendering of a pudgy pugs wrinkled face, capturing every fold and crease in exquisite detail.',
'Afrofuturism Style: A futuristic elephant with glowing tusks and intricate tribal tattoos roams through a neon-lit cityscape, embodying the power and mysticism of African culture.',
'Cubist Pop Art Style: This funky flamingo is a playful combination of bright colors and jumbled shapes, like a living puzzle with a sense of humor.'


]
 
    
        
        random.shuffle(prompts_new)
        selected_prompts = prompts_new[:4]
        for pr in selected_prompts:
            print(pr)
        
        
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
        #posts_evolve.append(sorted_dict[4:10])
        posts_evolve = posts_dict
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
            #diffusion_noise = posts_change_style[0][i]['latent']
            img, latent, prompt = gen_image(prompt, diffusion_noise, batch_size, num_steps, new_scale)
            hashtag = generate_hashtags_from_prompt(prompt)
            new_population.append({'num' : a, 'fitness':fitness, 'latent': latent, 'prompt': prompt, 'hashtag': hashtag})
            Image.fromarray(img[0]).save(f"{next_population_folder}/{a}.png") 
            a+=1            

        a=2
        style_change=0.5
        s=3
        selected_parents = []
            
        random.shuffle(posts_evolve)

        #for _ in range(len(posts_evolve[0])):
        for _ in range(4):          
            selected_parent = tournament_selection(posts_evolve)      
            
            selected_parents.append(selected_parent)
              
        random.shuffle(selected_parents)   
        
        archive = []
            
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
                
            archive_prompt_1 = some_post1['prompt'] + some_post2['prompt']
            archive_prompt_2 = some_post2['prompt'] + some_post1['prompt']
            
            do = True
            while do:
                print("doing..")
                if archive_prompt_1 in archive or archive_prompt_2 in archive or some_post1['prompt']==some_post2['prompt']:
                    print("TRIGGERED")
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
            archive_prompt_1 = some_post1['prompt'] + some_post2['prompt']
            archive_prompt_2 = some_post2['prompt'] + some_post1['prompt']
            archive.append(archive_prompt_1)
            archive.append(archive_prompt_2)
            
            print(archive)
  
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
    
            Image.fromarray(img[0]).save(f"{next_population_folder}/{a}.png") 
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
        
        random.shuffle(available_images)
        #selected_images =  random.sample(available_images, 4)

        for image_file in available_images[:4]:
            print(image_file)
            try:
                img = iio.imread(image_file)
                prompt = prompt_generator(img)
                prompts.append(prompt)
            except:
                print('error, trying new one')
                pass
           
        print(f'number of images selected: {len(prompts)}')         '''
        #prompts = gen_captions(4)
        
        
        
        for pr in selected_prompts:

            print(pr)
            diffusion_noise = tf.random.normal((batch_size, img_height//8, img_width//8, 4))
            img, latent, prompt = gen_image(pr, diffusion_noise, batch_size, num_steps, new_scale)
            hashtag = generate_hashtags_from_prompt(prompt)
            Image.fromarray(img[0]).save(f"{next_population_folder}/{a}.png")    
            
            new_population.append({'num' : a, 'fitness' : 0, 'latent': latent, 'prompt': prompt, 'hashtag': hashtag})
                

            a+=1   
                
        #storing new popluation dict
        posts_dict = new_population 

        with open(f'logs/posts_dict_{next_iteration}.txt', 'wb') as f:
            pickle.dump(posts_dict, f)

        print(f'done with iteration: {iteration}')
        print(f'get fitness and start next iteration: {next_iteration}')
           

        go = False
        