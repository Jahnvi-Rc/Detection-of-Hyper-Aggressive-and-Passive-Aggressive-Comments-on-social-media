#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:53:51 2019

@author: jahnvirc
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', -1)
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)
from sklearn.model_selection import train_test_split


from sklearn.feature_extraction.text import CountVectorizer
np.random.seed(37)

df = pd.read_csv('Data/dataframe.csv',engine='python')


cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(df.clean_text.values.astype(str)).toarray()
y = df.iloc[:,7]

X_train_cv,X_test_cv,y_train_cv,y_test_cv = train_test_split(X,y,test_size=0.25,random_state =0)

y_train_cv = y_train_cv.astype(str)
y_test_cv = y_test_cv.astype(str)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_cv = sc.fit_transform(X_train_cv)
X_test_cv = sc.transform(X_test_cv)

from sklearn.preprocessing import LabelEncoder
encoded_y = LabelEncoder()
y_train_cv = encoded_y.fit_transform(y_train_cv)
y_test_cv = encoded_y.fit_transform(y_test_cv)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_cv = scaler.fit_transform(X_train_cv)
X_test_cv = scaler.fit_transform(X_test_cv)




np.savetxt("categorized_data/X_train_cv.csv", X_train_cv, delimiter=",")
np.savetxt("categorized_data/X_test_cv.csv", X_test_cv, delimiter=",")
np.savetxt("categorized_data/y_train_cv.csv", y_train_cv, delimiter=",")
np.savetxt("categorized_data/y_test_cv.csv", y_test_cv, delimiter=",")


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

X = cv.fit_transform(df.clean_text.values.astype(str)).toarray()
y = df.iloc[:,7]

X_train_tfidf,X_test_tfidf,y_train_tfidf,y_test_tfidf = train_test_split(X,y,test_size=0.25,random_state =0)

y_train_tfidf = y_train_tfidf.astype(str)
y_test_tfidf = y_test_tfidf.astype(str)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_tfidf = sc.fit_transform(X_train_tfidf)
X_test_tfidf = sc.transform(X_test_tfidf)

from sklearn.preprocessing import LabelEncoder
encoded_y = LabelEncoder()
y_train_tfidf = encoded_y.fit_transform(y_train_tfidf)
y_test_tfidf = encoded_y.fit_transform(y_test_tfidf)


X_train_tfidf = scaler.fit_transform(X_train_tfidf)
X_test_tfidf = scaler.fit_transform(X_test_tfidf)

np.savetxt("categorized_data/X_train_tfidf.csv", X_train_tfidf, delimiter=",")
np.savetxt("categorized_data/X_test_tfidf.csv", X_test_tfidf, delimiter=",")
np.savetxt("categorized_data/y_train_tfidf.csv", y_train_tfidf, delimiter=",")
np.savetxt("categorized_data/y_test_tfidf.csv", y_test_tfidf, delimiter=",")


import gensim
from nltk.tokenize import word_tokenize
SIZE = 25

X_train = pd.read_csv('Data/X_train.csv',engine='python')
X_test = pd.read_csv('Data/X_test.csv',engine='python')


X_train['clean_text_wordlist'] = X_train.clean_text.astype(str).apply(lambda x : word_tokenize(x))
X_test['clean_text_wordlist'] = X_test.clean_text.astype(str).apply(lambda x : word_tokenize(x))

model = gensim.models.Word2Vec(X_train.clean_text_wordlist
                 , min_count=1
                 , size=SIZE
                 , window=3
                 , workers=4)
#print(model.most_similar('plane', topn=3))

def compute_avg_w2v_vector(w2v_dict, tweet):
    list_of_word_vectors = [w2v_dict[w] for w in tweet if w in w2v_dict.vocab.keys()]
    
    if len(list_of_word_vectors) == 0:
        result = [0.0]*SIZE
    else:
        result = np.sum(list_of_word_vectors, axis=0) / len(list_of_word_vectors)
        
    return result
X_train_w2v = X_train['clean_text_wordlist'].apply(lambda x: compute_avg_w2v_vector(model.wv, x))
X_test_w2v = X_test['clean_text_wordlist'].apply(lambda x: compute_avg_w2v_vector(model.wv, x))
X_train_w2v = pd.DataFrame(X_train_w2v.values.tolist(), index= X_train.index)
X_test_w2v = pd.DataFrame(X_test_w2v.values.tolist(), index= X_test.index)
X_train_w2v = pd.concat([X_train_w2v, X_train.drop(['clean_text', 'clean_text_wordlist'], axis=1)], axis=1)
X_test_w2v = pd.concat([X_test_w2v, X_test.drop(['clean_text', 'clean_text_wordlist'], axis=1)], axis=1)



np.savetxt("categorized_data/X_train_w2v.csv", X_train_w2v, delimiter=",")
np.savetxt("categorized_data/X_test_w2v.csv", X_test_w2v, delimiter=",")
