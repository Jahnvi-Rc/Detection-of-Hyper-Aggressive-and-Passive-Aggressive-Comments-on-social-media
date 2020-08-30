#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:15:45 2019

@author: jahnvirc
"""


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from sklearn.naive_bayes import MultinomialNB

index = ['Count Vectorizer','TF-IDF','Word2Vector']
columns = ['Random Forest','Logistic Regression','Naive Bayes']
output = pd.DataFrame(index=index,columns=columns)

X_train_cv = pd.read_csv('categorized_data/X_train_cv.csv',engine='python')
y_train_cv = pd.read_csv('categorized_data/y_train_cv.csv',engine='python')
X_train_tfidf = pd.read_csv('categorized_data/X_train_tfidf.csv',engine='python')
y_train_tfidf = pd.read_csv('categorized_data/y_train_tfidf.csv',engine='python')
X_train_w2v = pd.read_csv('categorized_data/X_train_tfidf.csv',engine='python')
y_train_w2v = pd.read_csv('Data/y_train.csv',engine='python')
X_test_cv = pd.read_csv('categorized_data/X_test_cv.csv',engine='python')
y_test_cv = pd.read_csv('categorized_data/y_test_cv.csv',engine='python')
X_test_tfidf = pd.read_csv('categorized_data/X_test_tfidf.csv',engine='python')
y_test_tfidf = pd.read_csv('categorized_data/y_test_tfidf.csv',engine='python')
X_test_w2v = pd.read_csv('categorized_data/X_test_tfidf.csv',engine='python')
y_test_w2v = pd.read_csv('Data/y_test.csv',engine='python')



#1. Logistic Regression
clf = LogisticRegression(random_state=10, solver='lbfgs',penalty = 'l2',C=1)
clf.fit(X_train_cv, y_train_cv)
test_accuracy = clf.score(X_test_cv, y_test_cv)
output.set_value('Count Vectorizer','Logistic Regression',test_accuracy*100);

#2. Random Forest
clf = RandomForestClassifier(n_estimators=10,max_features=1, bootstrap=False,criterion = 'entropy',random_state = 0)
clf.fit(X_train_cv, y_train_cv)
test_accuracy = clf.score(X_test_cv, y_test_cv)
output.set_value('Count Vectorizer','Random Forest',test_accuracy*100);

#3 Naive Bayes
clf = MultinomialNB(alpha=0.25) 
clf.fit(X_train_cv, y_train_cv)
test_accuracy = clf.score(X_test_cv, y_test_cv)
output.set_value('Count Vectorizer','Naive Bayes',test_accuracy*100);


#1. Logistic Regression
clf = LogisticRegression(random_state=10, solver='lbfgs',penalty = 'l2',C=1)
clf.fit(X_train_tfidf, y_train_tfidf)
test_accuracy = clf.score(X_test_tfidf, y_test_tfidf)
output.set_value('TF-IDF','Logistic Regression',test_accuracy*100);

#2. Random Forest
clf = RandomForestClassifier(n_estimators=10,max_features=1, bootstrap=False,criterion = 'entropy',random_state = 0)
clf.fit(X_train_tfidf, y_train_tfidf)
test_accuracy = clf.score(X_test_tfidf, y_test_tfidf)
output.set_value('TF-IDF','Random Forest',test_accuracy*100);

#3 Naive Bayes
clf = MultinomialNB(alpha=0.25) 
clf.fit(X_train_tfidf, y_train_tfidf)
test_accuracy = clf.score(X_test_tfidf, y_test_tfidf)
output.set_value('TF-IDF','Naive Bayes',test_accuracy*100);


#1. Logistic Regression
clf = LogisticRegression(random_state=10, solver='lbfgs',penalty = 'l2',C=1)
clf.fit(X_train_w2v, y_train_w2v)
test_accuracy = clf.score(X_test_w2v, y_test_w2v)
output.set_value('Word2Vector','Logistic Regression',test_accuracy*100);

#2. Random Forest
clf = RandomForestClassifier(n_estimators=10,max_features=1, bootstrap=False,criterion = 'entropy',random_state = 0)
clf.fit(X_train_w2v, y_train_w2v)
test_accuracy = clf.score(X_test_w2v, y_test_w2v)
output.set_value('Word2Vector','Random Forest',test_accuracy*100);

#3 Naive Bayes
clf = MultinomialNB(alpha=0.5) 
clf.fit(X_train_w2v, y_train_w2v)
test_accuracy = clf.score(X_test_w2v, y_test_w2v)
output.set_value('Word2Vector','Naive Bayes',test_accuracy*100);

print("Accuracy Table for Various Techniques and Models\n")
display(output)
output.to_csv('Outputs/Output.csv', encoding='utf-8', index=False)
