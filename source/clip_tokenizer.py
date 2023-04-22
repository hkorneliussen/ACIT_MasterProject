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

"""This code is based on: 
https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion/clip_tokenizer.py"""

#importing dependencies
import gzip
import html
from functools import lru_cache

import regex as re
from tensorflow import keras


@lru_cache()
#Defining function that creates a dictionary mapping between byte values and unicode characters
def bytes_to_unicode():
    
    #defining a list "bs" of values that corresponds to ASCII characters and some latin characters
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    #defining a list cs with the same values as bs
    cs = bs[:]
    n = 0
    #iterating over all possible 8-bit values (0-255)
    for b in range(2**8):
        #adding any missing value to both bs and cs
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
            
    #transforming cs into a list of unicode characters        
    cs = [chr(n) for n in cs]
    #creating a dictionry by paring the values in bs with the values in cs
    return dict(zip(bs, cs))
    
#Defining function to extract all consectuive character pairs from an input word string    
def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    #returning the pairs as a set of tuples
    return pairs

#Defining function that clean up HTML-encoded text    
def basic_clean(text):
    text = html.unescape(html.unescape(text))
    return text.strip()
    
#function that removes extra whitespaces from the input text
def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
    
'''
The class simpleTokenizer provides an implementation of a tokenizer that splits text into tokens (sub-words) using a process called Byte Pair Encoding (BPE). The class takes a BPE vocabulary as input and uses it to encode text into a sequence of tokens and decode it back into text. 
'''
    
#Definign a python class named SimpleTokenizer    
class SimpleTokenizer:
    #The optional argument bpe_path is the file path to the BPE vocabulary used for tokenization
    def __init__(self, bpe_path=None):
        #if bpe_path isn't provided, the method downloads the default BPE vocabulary from a URL
        bpe_path = 'models/bpe_simple_vocab_16e6.txt.gz'
        '''
        bpe_path = bpe_path or keras.utils.get_file(
            "bpe_simple_vocab_16e6.txt.gz",
            "https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz?raw=true",
            file_hash="924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
        )
        '''
        #byte_encoder and byte_decoder are dictionaries that store the mappings between bytes and unicode characters
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        #gzip is used to open the BPE vocabulary file and reading its content
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        #the content is split into lines and stored in the merges variable
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        #The vocab attribute is created by concatenating a list of all unicode characters, a list of the same characters with a special toeken '/w' append to each and the BPE merges
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.vocab = vocab
        #the encoder and decoder attributes are dictionaries that store the mappings between tokens and their indices
        self.encoder = self._create_encoder(self.vocab)
        self.decoder = self._create_decoder(self.encoder)
        #the bpe_ranks attribute is a dictionary that maps each BPE merge to its rank in the 'merges' list
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        
        #special_tokens is a dictionary that stores special tokens and their corresponding values
        self.special_tokens = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        #the cache attribute is a dictionary that caches the results of tokenizing text. T
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        #the path attribute is used for matcing text during tokenization
        self.pat = self._create_pat()

    '''
    the create_encoder function takes a list 'vocab' as input and returns a dictionary that maps each item in 'vocab' to a unique integer
    '''
    def _create_encoder(self, vocab):
        return dict(zip(vocab, range(len(vocab))))
        
    '''
    create_decoder returns a decoding of the vocabulary represented by the encoder. 
    '''
    def _create_decoder(self, encoder):
        return {v: k for k, v in encoder.items()}
        
    #create_pat returns a compiled regular expression pattern    
    def _create_pat(self):
        #the pattern is constructed by concatenatng the escaped keys from the special_tokens separated by the "|" symbol
        return re.compile(
            "|".join([re.escape(key) for key in self.special_tokens.keys()])
            + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
     
    #defining function that returns the integer encoding of the end-of-text token    
    @property
    def end_of_text(self):
        return self.encoder["<|endoftext|>"]
        
    #defining function that returns the integer encoding of the start-of-text token    
    @property
    def start_of_text(self):
        return self.encoder["<|startoftext|>"]
        
    #defining function to add new tokens to the existing vocabulary of the tokenizer    
    def add_tokens(self, tokens):
        #checking if tokens is a sting
        if isinstance(tokens, str):
            #if it is, convert it to a single-element list
            tokens = [tokens]
        tokens_added = 0
        #iterating over the tokens and adding them to the vocal if its not already in it
        for token in tokens:
            if token in self.vocab:
                continue
            tokens_added += 1
            self.vocab.append(token)
            self.special_tokens[token] = token
            self.cache[token] = token
        #updating the encoder and ecoder dictionaries
        self.encoder = self._create_encoder(self.vocab)
        self.decoder = self._create_decoder(self.encoder)
        #updating the pat (pattern) instance variable
        self.pat = self._create_pat()
        #returns the number of tokens added to vocab
        return tokens_added
    
    '''    
    function that implements the BPE algorithm for subword tokenization. The input 'token' is a string that represent a single word. The function returns a BPE-encoded representation of the word
    
    The BPE algorithm is a data compression technique for text. The function takes a single token as input (sequence of chaacters) and returns a compressed representation of the token. The algorithm works by iteratively merging the most frquent character pairs in the token until a stopping condition is reached, such as reaching a maximum number of merges. This results in a compressed representation of the token, where common sub-word patterns are represented by a single unit (called a byte-pair).  
    '''
    def bpe(self, token):
        #checking if the input token is in the cache
        #the cache is used to store previously computed BPE representations to speed up the encoding process
        if token in self.cache:
            return self.cache[token]
        #transforming the token into a tuple of characters, where each character is treated as a single subword
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        #passing the tuple into get_pairs, which generates a set of all possible bigrams (pairs of consective subwords) in the tuple
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        #in the while loop, the most frequent biagrams are merged until no more frequent bigrams can be found
        while True:
            #the biagram with highest frequency is identified by sorting the set of bigrams using the frequency values stored in self.bpe_ranks
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        #returning the BPE-encoded representation of the word
        return word
        
    #Function that takes a string of text as input and returns an encoded list of integers, which are indices of the BPE tokens in the vocabulary    
    def encode(self, text):
        bpe_tokens = []
        #performing text pre-processing steps
        text = whitespace_clean(basic_clean(text)).lower()
        #for each token, perform byte encoding of the token and apply BPE on it
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        #Return the encoded list of integers
        return [self.start_of_text] + bpe_tokens + [self.end_of_text]
        

    #function that takes as input a list of tokens and return the original text representation of the token sequence
    def decode(self, tokens):
        #joining the tokens in the input list to a single string
        text = "".join([self.decoder[token] for token in tokens])
        #decodes the string obtained to a UTF-8 string by converting each character in the string to its corresponding byte representation using byte_encoder mapping, and then decoding the resulting byte string into a Unicode string
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            #replace all "/w" with a space character
            .replace("</w>", " ")
        )
        #returning the resulting text string
        return text
