#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 20:33:26 2018

@author: yuyingjie
"""

import re
import numpy as np
import pandas as pd
from fastText import load_model

from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.corpus import stopwords

# the number of words we look at each example. Could experiment with this.
window_length=200

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

tokenizer=TweetTokenizer()
lemmatizer = WordNetLemmatizer()
eng_stopwords = set(stopwords.words("english"))

APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}

def clean(comment):
    comment = comment.lower()
    # replace ips
    comment = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', comment)
    # Isolate punctuation
    comment = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', comment)
    # remove username
    comment = re.sub('\[\[.*\]','',comment)
        # Remove some special characters
    comment = re.sub(r'([\;\:\|•«\n])', ' ', comment)
    # Replace numbers and symbols with language
    comment = comment.replace('&', ' and ')
    comment = comment.replace('@', ' at ')
    comment = comment.replace('0', ' zero ')
    comment = comment.replace('1', ' one ')
    comment = comment.replace('2', ' two ')
    comment = comment.replace('3', ' three ')
    comment = comment.replace('4', ' four ')
    comment = comment.replace('5', ' five ')
    comment = comment.replace('6', ' six ')
    comment = comment.replace('7', ' seven ')
    comment = comment.replace('8', ' eight ')
    comment = comment.replace('9', ' nine ')
        
#    words= tokenizer.tokenize(comment)
#    
#    words = [APPO[word] if word in APPO else word for word in words]
#    
#    words = [lemmatizer.lemmatize(word,"v") for word in words]
#    
#    words = [word for word in words if not word in eng_stopwords]
#    
#    clean_sent = " ".join(words)
    
    return comment
    
train['comment_text'] = train['comment_text'].fillna('unknown')
test['comment_text'] = train['comment_text'].fillna('unknown')

#%%
ft_model = load_model('ft_model.bin')
classes = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]
n_features = ft_model.get_dimension() #300

def text_to_vector(text):
    '''turn a given string to a sequence of word vectors'''
    text = clean(text)
    words = text.split()
    window = words[-window_length:]
    
    x = np.zeros((window_length, n_features))
    
    for i, word in enumerate(window):
        x[i,:] = ft_model.get_word_vector(word).astype('float32')
    return x

def df_to_data(df):
    '''convert a given dataframe to a dataset of inputs fot the NN'''
    x= np.zeros((len(df), window_length, n_features), dtype='float32')
    for i, comment in enumerate(df['comment_text']):
        x[i,:] = text_to_vector(comment)
    return x

#%%
# using a generator so that we dont have to keep the whole thing im memory
'''
The idea is that instead of converting the whole training set to one large array,
 we can write a function that just spits out one batch of data at a time, infinitely. 
 Keras can automaticaly spin up a separate thread for this method (note though that "threads" 
 in Python are ridiculous and do not give any speedup whatsoever). 
 This means that we have to write some more code and training will be slightly slower, 
 but we need only a fraction of the memory and we can add some cool randomization 
 to each batch later on (see ideas section below).
'''



