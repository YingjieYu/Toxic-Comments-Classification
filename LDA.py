#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:33:29 2018

@author: yuyingjie

create features based on lda model 
get a dataframe of topic probability

"""
import pandas as pd
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

import gensim
from gensim import corpora

train = pd.read_csv("train.csv")

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_complete = train['comment_text']
doc_clean = [clean(doc).split() for doc in doc_complete]    
# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=20)
#ldamodel..save('lda.model')
ldamodel = gensim.models.ldamodel.LdaModel.load("lda.model")

doc_topics = ldamodel.get_document_topics(doc_term_matrix)
doc_topics=[dict(i) for i in doc_topics]
doc_topics = pd.DataFrame(doc_topics)
doc_topics.fillna(value=0, inplace=True)
doc_topics.to_csv('doc_topics.csv')
