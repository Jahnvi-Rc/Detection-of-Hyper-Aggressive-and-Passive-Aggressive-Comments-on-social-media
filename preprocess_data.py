#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:21:00 2019

@author: jahnvirc
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', -1)
import re
import string
import emoji
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
np.random.seed(37)

df = pd.read_csv('Data/formspring labelled data.csv',engine='python')
df = df.drop_duplicates(subset='TEXT', keep='first',inplace=False)
data = df.iloc[:,[3,4,6,7,8]]
data = data.infer_objects() 
class TextCounts(BaseEstimator, TransformerMixin):
    def transform(self, X):
        count_words = X.apply(lambda x: len(re.findall(r'\w+', str(x)))) 
        count_mentions = X.apply(lambda x: len(re.findall(r'@\w+', str(x))))
        count_hashtags = X.apply(lambda x: len(re.findall(r'#\w+', str(x))))
        count_capital_words = X.apply(lambda x: len(re.findall(r'\b[A-Z]{2,}\b', str(x))))
        count_excl_quest_marks = X.apply(lambda x: len(re.findall(r'!|\?', str(x))))
        count_urls = X.apply(lambda x: len(re.findall(r'http.?://[^\s]+[\s]?', str(x))))
        count_emojis = X.apply(lambda x: emoji.demojize(str(x))).apply(lambda x: len(re.findall(r':[a-z_&]+:', str(x))))        
        df_test = pd.DataFrame({'count_words': count_words
                           , 'count_mentions': count_mentions
                           , 'count_hashtags': count_hashtags
                           , 'count_capital_words': count_capital_words
                           , 'count_excl_quest_marks': count_excl_quest_marks
                           , 'count_urls': count_urls
                           , 'count_emojis': count_emojis
                          })
        
        return df_test
tc = TextCounts()
df_eda = tc.transform(data.TEXT)
df_eda['ANSWER'] = data.ANSWER
class CleanText(BaseEstimator, TransformerMixin):
   
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)
    
    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
    
    def emoji_oneword(self, input_text):
        return input_text.replace('_','')
    
    def remove_punctuation(self, input_text):
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)

    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    def remove_br(self, input_text):
        return re.sub('br', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
    
    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def transform(self, X):
        clean_X = X.apply(self.remove_br).apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)
        return clean_X

ct = CleanText()
sr_clean = ct.transform(data.TEXT)   
df_model = df_eda
df_model['clean_text'] = sr_clean
print(df_model.ANSWER.value_counts())

from sklearn.utils import resample
df_majority = df_model[df_model.ANSWER=='No']
df_minority = df_model[df_model.ANSWER=='Yes']
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=7000,    
                                 random_state=123) 
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled.ANSWER.value_counts()

df_upsampled.to_csv('Data/dataframe.csv', encoding='utf-8', index=False)
X_train, X_test, y_train, y_test = train_test_split(df_upsampled.drop('ANSWER', axis=1), df_upsampled.ANSWER, test_size=0.25, random_state=0)


X_train.to_csv('Data/X_train.csv', encoding='utf-8', index=False)
X_test.to_csv('Data/X_test.csv', encoding='utf-8', index=False)
y_train.to_csv('Data/y_train.csv', encoding='utf-8', index=False)
y_test.to_csv('Data/y_test.csv', encoding='utf-8', index=False)