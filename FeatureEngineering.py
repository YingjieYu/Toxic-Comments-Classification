#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct features:
Features which are a directly due to words/content.We would be exploring the following techniques

Word frequency features
Count features
Bigrams
Trigrams
Vector distance mapping of words (Eg: Word2Vec)
Sentiment scores


Indirect features:
Some more experimental features.

count of sentences
count of words
count of unique words
count of letters
count of punctuations
count of uppercase words/letters
count of stop words
Avg length of each word


Leaky features:
From the example, we know that the comments contain identifier information (eg: IP, username,etc.). We can create features out of them but, it will certainly lead to overfitting to this specific Wikipedia use-case.

toxic IP scores
toxic users
Note: Creating the indirect and leaky features first. There are two reasons for this,

Count features(Direct features) are useful only if they are created from a clean corpus
Also the indirect features help compensate for the loss of information when cleaning the dataset
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
color = sns.color_palette()

import re
import gc
import time
import warnings
from collections import defaultdict
from tqdm import tqdm

# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.corpus import stopwords


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from scipy.sparse import coo_matrix, hstack,csr_matrix



df = pd.concat((train,test))
df = df['comment_text'].fillna('unknown',inplace=True)
# creat a clean feature
rowsums=train.iloc[:,2:].sum(axis=1)
train['clean'] =(rowsums==0) 
train['clean'].sum()

def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

#%% create indirect features

#word count
df['count_word'] = df['comment_text'].apply(lambda x: len(str(x).split()))
#'unique word count
df['count_unique_word'] = df['comment_text'].apply(lambda x: len(set(str(x).split())))


#df_train = df.iloc[0:len(train),]
#df_test = df.iloc[len(train):,]
## join tags
#train_tags= train.iloc[:,2:]
#df_train = pd.concat([df_train, train_tags], axis = 1)

#plt.figure(figsize=(12,6))
#sns.violinplot(y = 'count_word',x='clean',data = df_train,split=True,inner="quart")
#plt.xlabel('Clean?', fontsize =12)
#plt.ylabel('#of words', fontsize = 12)
#plt.title('number of words in each comments',fontsize = 15)
#plt.show()

# spams
df['word_unique_percent'] = df['count_unique_word']*100/df['count_word']
df['spammers'] = df['word_unique_percent'] < 30

#%% corpus cleaning
corpus = df.comment_text
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
    ''' return cleaned word list'''
    comment = comment.lower()
    comment = re.sub('\\n','',comment) # remove \n
    comment = re.sub('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}','',comment) # remove id and user
    comment = re.sub('\[\[.*\]','',comment) #remove usernames
    
    words= tokenizer.tokenize(comment)
    
    words = [APPO[word] if word in APPO else word for word in words]
    
    words = [lemmatizer.lemmatize(word,"v") for word in words]
    
    words = [word for word in words if not word in eng_stopwords]
    
    clean_sent = " ".join(words)
    return clean_sent
    
clean_corpus = corpus.apply(lambda x: clean(x)) # takes about 3 minites


tfidf_clean = TfidfVectorizer(min_df=200,  max_features=None, 
            strip_accents='unicode', analyzer='word',ngram_range=(1,3),
            use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tfidf_clean_matrix = tfidf_clean.fit_transform(clean_corpus) 
tfidf_features = np.array(tfidf_clean.get_feature_names())

train_tfidf =  tfidf_clean.transform(clean_corpus.iloc[:train.shape[0]])
test_tfidf = tfidf_clean.transform(clean_corpus.iloc[train.shape[0]:])
save_sparse_csr('train_tfidf', train_tfidf)
save_sparse_csr('test_tfidf', test_tfidf)


cv = CountVectorizer()
df_countvec=cv.fit_transform(clean_corpus)
train_countvec = df_countvec[:train.shape[0]]
test_countvec= df_countvec[train.shape[0]:]
save_sparse_csr('train_countvec', train_countvec)
save_sparse_csr('test_countvec', test_countvec)


#######################
#with open("glove.840B.300d.txt", "rb") as lines:
#    w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
#           for line in lines}

# load the GloVe vectors in a dictionary

w2v = {}
f = open('glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    # Catch the exception where there are strings in the Glove text file.
    try:
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs
    except ValueError:
        pass
f.close()

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
                
w2v_tfidf=TfidfEmbeddingVectorizer(w2v)
df_w2v_tfidf =w2v_tfidf.fit(clean_corpus)
df_w2v_tfidf =df_w2v_tfidf.transform(clean_corpus)

df_w2v_tfidf_matrix = np.asmatrix(df_w2v_tfidf)
#df_w2v_tfidf_matrix.dump("df_w2v_tfidf_matrix.dat")
#df_w2v_tfidf_matrix = numpy.load("df_w2v_tfidf_matrix.dat")

train_w2v_tfidf = df_w2v_tfidf[:train.shape[0]]
test_w2v_tfidf = df_w2v_tfidf[train.shape[0]:]


#df_w2v_tfidf__ = pd.DataFrame(df_w2v_tfidf_['value'].values.tolist())


# this function creates a normalized vector for the whole sentence
def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(w2v[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

df_glove = [sent2vec(x) for x in tqdm(clean_corpus)]
df_glove = np.array(df_glove)

#%%




#%% visualization of tiidf and token importance
def top_tfidf_feats(row, features, top_n = 25):
    top_ids = np.argsort(row)[::-1][:top_n]
    top_features = [(features[i],row[i]) for i in top_ids]
    df = pd.DataFrame(top_features)
    df.columns = ['feature','tfidf']
    return df

def top_feats_in_doc(X_matrix, features,row_id,top_n = 25):
    '''top tfidf features in specific document (matrix row)'''
    row = np.squeeze(X_matrix[row_id].toarray())
    return top_tfidf_feats(row,features,top_n)

def top_mean_feats(Xtr, features, grp_ids, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    
    D = Xtr[grp_ids].toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

train_labels = train.iloc[:,2:8]
def top_feats_by_class(Xtr, features, min_tfidf=0.1, top_n=20):
    df_importance = []
    for column in train_labels.columns:
        ids = train_labels[train_labels[column] == 1].index
        df_= top_mean_feats(Xtr, features, ids,min_tfidf=min_tfidf, top_n=top_n)
        df_.column = column
        df_importance.append(df_)
    return df_importance

tfidf_top_n_per_class=top_feats_by_class(train_unigrams,tfidf_features)

plt.figure(figsize=(16,22))
plt.suptitle("TF_IDF Top words per class(unigrams)",fontsize=20)
gridspec.GridSpec(4,2)
plt.subplot2grid((4,2),(0,0))
sns.barplot(tfidf_top_n_per_class[0].feature.iloc[0:9],tfidf_top_n_per_class[0].tfidf.iloc[0:9],color=color[0])
plt.title("class : Toxic",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)

plt.subplot2grid((4,2),(0,1))
sns.barplot(tfidf_top_n_per_class[1].feature.iloc[0:9],tfidf_top_n_per_class[1].tfidf.iloc[0:9],color=color[1])
plt.title("class : Severe toxic",fontsize=15)
plt.ylabel('TF-IDF score', fontsize=12)

#plt.subplot2grid((4,2),(3,0),colspan=2)
#sns.barplot(tfidf_top_n_per_lass[6].feature.iloc[0:19],tfidf_top_n_per_lass[6].tfidf.iloc[0:19])
#plt.title("class : Clean",fontsize=15)
#plt.xlabel('Word', fontsize=12)
#plt.ylabel('TF-IDF score', fontsize=12)

