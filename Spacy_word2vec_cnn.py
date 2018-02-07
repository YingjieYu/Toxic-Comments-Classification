#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:10:49 2018

@author: yuyingjie

 SpaCy is an industrial-strength natural language processing python library (spacy.io). 
 The built-in word vectors are 300-dimensional vectors trained on the Common Crawl corpus using the GloVe algorithm. 
 Because there are a lot of text examples in the training set, 
 I use spaCy's pipeline feature to process 500 samples at a time, 
 grab their word vectors, and format them into numpy arrays.

"""
#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from IPython.display import display
import base64
import string
import re
from collections import Counter
from time import time
# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from nltk.corpus import stopwords
from sklearn.metrics import log_loss


stopwords = stopwords.words('english')
sns.set_context('notebook')

import spacy
import en_core_web_sm
import en_core_web_lg
nlp = en_core_web_sm.load()
nlp_lg =en_core_web_lg.load()
train = pd.read_csv("train.csv")
y_train = train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
test = pd.read_csv('test.csv')
#%%
# clean data before feed into spacy
punctuation = string.punctuation

def cleanup_text(docs,logging=False, nlp=nlp_lg):
    '''
    cleanup text by removing personal pronouns, stopwords, and puncuation
    '''
    texts=[]
    counter=1
    for doc in docs:
        if counter %  1000 ==0 and logging:
            print("Processd %d out of %d documents." % (counter, len(docs)))
        counter +=1
        doc = nlp(doc,disable=['parser','ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_!= '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuation]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)

#%% check to most frequent words in each catogories
#toxic_text = [text for text in train[train['toxic']==1]['comment_text']]
#toxic_text_clean = cleanup_text(toxic_text,logging=True)
##Remove 's from all text because spaCy doesn't remove this contraction when lemmatizing words for some reason.
#toxic_text_clean = ' '.join(toxic_text_clean).split()
#toxic_text_clean = [word for word in toxic_text_clean if word !='\'s']
#toxic_counts = Counter(toxic_text_clean)
#
## Plot top 25 most frequently occuring words for Edgar Allen Poe
#toxic_common_words = [word[0] for word in toxic_counts.most_common(25)]
#toxic_common_counts = [word[1] for word in toxic_counts.most_common(25)]
#
## Use spooky background
#plt.style.use('dark_background')
#plt.figure(figsize=(15, 12))
#
#sns.barplot(x=toxic_common_words, y=toxic_common_counts)
#plt.title('Most Common Words in Toxic')
#plt.show()
    
#%% clean up train and test comments
train_cleaned = cleanup_text(train['comment_text'], logging=True)
print('Cleaned up training data shape: ', train_cleaned.shape)

train_cleaned_lg = cleanup_text(train['comment_text'], logging=True,nlp=nlp_lg)
print('Cleaned up training data shape: ', train_cleaned_lg.shape)

# Testing cell. Use for testing word vector dimensionality. Should be (384,)
# But when using pipe function word vectors are (128,) for some reason...
#te2 = nlp(train_cleaned_lg[0])
#print(te2.vector.shape)
#for doc in nlp.pipe(train_cleaned_lg[:5]):
#     print(doc.vector.shape)

test_cleaned = cleanup_text(test['comment_text'], logging=True)
test_cleaned_lg = cleanup_text(test['comment_text'], logging=True,nlp=nlp_lg)

#%% parse documents
start = time()
train_vec_lg=[]
for doc in nlp_lg.pipe(train_cleaned_lg,batch_size = 500):
    if doc.has_vector:
        train_vec_lg.append(doc.vector)
    # If doc doesn't have a vector, then fill it with zeros.    
    else:
        train_vec_lg.append(np.zeros((128,),dtype='float32'))
train_vec_lg=np.array(train_vec_lg)

end = time()
print('totoal time passed parsing ducument:{} seconds'.format(end-start))
print('Size of vector embeddings: ', train_vec_lg.shape[1])
print('Shape of vectors embeddings matrix: ', train_vec_lg.shape)

test_vec_lg = []
for doc in nlp_lg.pipe(test_cleaned_lg,batch_size = 500):
    if doc.has_vector:
        test_vec_lg.append(doc.vector)
    # If doc doesn't have a vector, then fill it with zeros.    
    else:
        test_vec_lg.append(np.zeros((300,),dtype='float32'))
test_vec_lg=np.array(test_vec_lg)
        


#%% Alternate Approach using Word2Vec
'''
Word2Vec also benefits from having stopwords because they give context to the sentences. 
Therefore we will instead modify our above function "cleanup_text" and create a new function called "cleanup_text_word2vec" 
to not remove stopwords as well as include personal pronouns.
'''
all_text = np.concatenate((train['comment_text'],test['comment_text']),axis=0)
all_text = pd.DataFrame(all_text, columns = ['text'])
len(all_text)
def clean_text_word2vec(docs, logging=True, nlp = nlp):
    sentences=[]
    counter = 1
    for doc in docs:
        if counter %1000==0 and logging:
            print('Processed %d out of %d documents' % (counter, len(docs)))
        # Disable tagger so that lemma_ of personal pronouns (I, me, etc) don't getted marked as "-PRON-"
        doc = nlp(doc, disable = ['tagger'])
        doc = ' '.join([tok.lemma_.lower() for tok in doc])
        # Split into sentences based on punctuation
        doc = re.split("[\.?!;] ", doc)
        # Remove commas, periods, and other punctuation (mostly commas)
        doc = [re.sub("[\.,;:!?]", "", sent) for sent in doc]
        # Split into words
        doc = [sent.split() for sent in doc]
        sentences += doc
        counter += 1
    return sentences

all_cleaned_word2vec = clean_text_word2vec(all_text['text'], logging=True)

'''
Here we define the parameters for Word2Vec:

size: Word vector dimensionality size is 300
window: Maximum distance between center word and predicted word in a sentence
min_count: Ignore all words that appear with less frequency than this
workers: Use this many workers to train model. Leads to faster training on multi-core machines
sg: Define archetecture. 1 for skip-gram, 0 for continouous bag of words (CBOW).
CBOW is faster but skip-gram gives better performance
'''
        
from gensim.models.word2vec import Word2Vec

text_dim=300
wordvec_model = Word2Vec(all_cleaned_word2vec, size =text_dim, window =5,min_count=3,workers=4,sg=1)
print("%d unique words represented by %d dimensional vectors" % (len(wordvec_model.wv.vocab), text_dim))
        
def create_average_vec(doc):
    '''
    create word vector given a cleaned priece of text,
    rememver to use train_cleaned
    '''
    average = np.zeros((text_dim,), dtype='float32')
    num_words = 0.
    for word in doc.split():
        if word in wordvec_model.wv.vocab:
            average = np.add(average, wordvec_model[word])
            num_words += 1.
    if num_words != 0.:
        average = np.divide(average, num_words)
    return average

# count the number of empty strings in train_cleaned and remove them
#count = 0
#index_to_remove = []
#for i in range(len(train_cleaned)):
#    if train_cleaned[i]=='':
#        print('index:',i)
#        index_to_remove.append(i)
#        count+=1
#print(count)
#train_cleaned = train_cleaned.drop(index_to_remove,axis = 0)
#train_cleaned.shape  
  
# create word vectors
train_vec_w2v = np.zeros((train_cleaned.shape[0], text_dim), dtype='float32')
for i in range(len(train_cleaned)):
    train_vec_w2v[i] = create_average_vec(train_cleaned[i])
print("Train word vector shape:", train_vec_w2v.shape)

test_vec_w2v = np.zeros((test_cleaned.shape[0], text_dim), dtype='float32')
for i in range(len(test_cleaned)):
    test_vec_w2v[i] = create_average_vec(test_cleaned[i])
print("Test word vector shape:", test_vec_w2v.shape)        
#%% deep learning with Keras
'''
The architecture below is a multilayer perceptron with four hidden layers, 
relu activation function, and he normal kernel initialization. 
We also include four dropout layers to avoid overfitting.

for multi label classification:

Don't use softmax.

Use sigmoid for activation of your output layer.

Use binary_crossentropy for loss function.

Use predict for evaluation.
'''
y_train=np.array(y_train)
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM, Embedding, Bidirectional, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import SGD

def build_model(architecture = 'mlp'):
    model = Sequential()
    if architecture == 'mlp':
        
        # Densely Connected Neural Network (Multi-Layer Perceptron)
        model.add(Dense(512, activation='relu',input_dim=300,kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(y_train.shape[1], activation='sigmoid'))
    if architecture == 'cnn':
        # cnn uses convolutions over the input layer to compute the output.
        inputs = Input(shape=(300,1))
        x = Conv1D(64, 3, strides=1, padding='same', activation='relu')(inputs)
        #Cuts the size of the output in half, maxing over every 2 inputs
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, 3, strides=1, padding='same', activation='relu')(x)
        x = GlobalMaxPooling1D()(x) 
        outputs = Dense(y_train.shape[1], activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs, name='CNN')
        
    elif architecture == 'lstm':
        # LSTM network
        inputs = Input(shape=(300,1))

        x = Bidirectional(LSTM(64, return_sequences=True),
                          merge_mode='concat')(inputs)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        outputs = Dense(3, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='LSTM')
    return model

model_mlp = build_model('mlp')
model_cnn =build_model('cnn')
#model_mlp.summary()
#model_cnn.summary()

# If the model is a CNN then expand the dimensions of the training data
if model_cnn.name == "CNN" or model_lstm.name == 'LSTM':
    X_train = np.expand_dims(train_vec_w2v, axis=2)
    X_test = np.expand_dims(test_vec_w2v, axis=2)
    print('Text train shape: ', X_test.shape)
    print('Text test shape: ', X_test.shape)


# Compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_cnn.compile(optimizer=sgd, loss='binary_crossentropy')

model_cnn.fit(X_train,y_train, epochs=8, batch_size=128,verbose=1)
preds = model_cnn.predict(X_test)
print(preds.shape)

subm = pd.read_csv('sample_submission.csv')

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('sub_w2v_cnn.csv', index=False)
