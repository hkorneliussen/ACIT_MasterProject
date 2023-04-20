'''
source: https://towardsdatascience.com/keyword-extraction-process-in-python-with-natural-language-processing-nlp-d769a9069d5c
'''

print("importing dependenceis")
#importing dependencies
import spacy
from collections import Counter
from string import punctuation
import re               

print("loading spacy model")
#loading spacy model
nlp = spacy.load('hashtags')


'''
Keyword extraction function
'''
def get_hotwords(text):
  result = []
  #list of the part of speech tag we want to extract 
  #can also extract other parts, e.g. verbs
  pos_tag = ['PROPN', 'ADJ', 'NOUN']
  #converting input text to lowercase and tokenize it via the spacy model
  #the resulting doc object contains token objects 
  doc = nlp(text.lower())
  for token in doc:
    #if the tokenized text is part of stopwords or punctuation, ignore this token
    if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
      continue
    #store the result part of speech tag of the tokenized text
    if(token.pos_ in pos_tag):
      result.append(token.text)
  #return the result as a list of strings
  return result
       
def generate_hashtags_from_prompt(prompt):
    output = set(get_hotwords(prompt))
    hashtags = [('#' + x) for x in output]
    hashtag = ' '.join(hashtags)
    
    search_list = ['style']
    if re.compile('|'.join(search_list), re.IGNORECASE).search(hashtag):
        hashtag = hashtag.replace('#style', '')
        
    search_list = ['resolution']
    if re.compile('|'.join(search_list), re.IGNORECASE).search(hashtag):
        hashtag = hashtag.replace('#resolution', '')
        
    search_list = ['high']
    if re.compile('|'.join(search_list), re.IGNORECASE).search(hashtag):
        hashtag = hashtag.replace('#high', '')
        
    
    return hashtag   