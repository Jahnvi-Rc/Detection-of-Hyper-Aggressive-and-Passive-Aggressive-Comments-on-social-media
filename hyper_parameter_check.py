#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:10:33 2019

@author: jahnvirc
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

clf_rf = RandomForestClassifier(n_estimators=6,bootstrap=True,random_state = 0)

parameters = [{'n_estimators':[1,2,3,4,5,6,7,8,9,10],
              'max_features':[1,3,5],
              'bootstrap': [True, False],
              'criterion': ["gini", "entropy"]
              }]
grid_search = GridSearchCV(estimator = clf_rf,param_grid= parameters,scoring='accuracy',cv=5,n_jobs=-1)


X_train_cv = pd.read_csv('categorized_data/X_train_cv.csv',engine='python')
y_train_cv = pd.read_csv('categorized_data/y_train_cv.csv',engine='python')
y_train_cv = y_train_cv.values
X_train_cv = X_train_cv.values
y_train_cv = y_train_cv.ravel()
grid_search = grid_search.fit(X_train_cv,y_train_cv)
best_parm_cv = grid_search.best_params_

X_train_tfidf = pd.read_csv('categorized_data/X_train_tfidf.csv',engine='python')
y_train_tfidf = pd.read_csv('categorized_data/y_train_tfidf.csv',engine='python')
y_train_tfidf = y_train_tfidf.values
X_train_tfidf = X_train_tfidf.values
y_train_tfidf = y_train_tfidf.ravel()
grid_search = grid_search.fit(X_train_tfidf,y_train_tfidf)
best_parm_tfidf = grid_search.best_params_

X_train_w2v = pd.read_csv('categorized_data/X_train_tfidf.csv',engine='python')
y_train_w2v = pd.read_csv('Data/y_train.csv',engine='python')
y_train_w2v = y_train_w2v.values
X_train_w2v = X_train_w2v.values
y_train_w2v = y_train_w2v.ravel()
grid_search = grid_search.fit(X_train_w2v,y_train_w2v)
best_parm_w2v = grid_search.best_params_

index = ['n_estimators','max_features','bootstrap','criterion']
columns = ['Random Forest CV','Random Forest TFIDF','Random Forest W2V']
vals_rf = pd.DataFrame(index=index,columns=columns)
vals_rf.set_value('n_estimators','Random Forest CV',best_parm_cv['n_estimators']);
vals_rf.set_value('max_features','Random Forest CV',best_parm_cv['max_features']);
vals_rf.set_value('bootstrap','Random Forest CV',best_parm_cv['bootstrap']);
vals_rf.set_value('criterion','Random Forest CV',best_parm_cv['criterion']);
vals_rf.set_value('n_estimators','Random Forest TFIDF',best_parm_tfidf['n_estimators']);
vals_rf.set_value('max_features','Random Forest TFIDF',best_parm_tfidf['max_features']);
vals_rf.set_value('bootstrap','Random Forest TFIDF',best_parm_tfidf['bootstrap']);
vals_rf.set_value('criterion','Random Forest TFIDF',best_parm_tfidf['criterion']);
vals_rf.set_value('n_estimators','Random Forest W2V',best_parm_w2v['n_estimators']);
vals_rf.set_value('max_features','Random Forest W2V',best_parm_w2v['max_features']);
vals_rf.set_value('bootstrap','Random Forest W2V',best_parm_w2v['bootstrap']);
vals_rf.set_value('criterion','Random Forest W2V',best_parm_w2v['criterion']);
display(vals_rf)

clf_mnb = MultinomialNB()
parameters_mnb = [{'alpha': (0.25, 0.5, 0.75)
              }]
grid_search = GridSearchCV(estimator = clf_mnb,param_grid= parameters_mnb,scoring='accuracy',cv=10,n_jobs=-1)

grid_search = grid_search.fit(X_train_cv,y_train_cv)
best_parm_mnb_cv = grid_search.best_params_

grid_search = grid_search.fit(X_train_tfidf,y_train_tfidf)
best_parm_mnb_tfidf = grid_search.best_params_

grid_search = grid_search.fit(X_train_w2v,y_train_w2v)
best_parm_mnb_w2v = grid_search.best_params_

index = ['alpha']
columns = ['MultinomialNB CV','MultinomialNB TFIDF','MultinomialNB W2V']
vals_mnb = pd.DataFrame(index=index,columns=columns)
vals_mnb.set_value('alpha','MultinomialNB CV',best_parm_mnb_cv['alpha']);
vals_mnb.set_value('alpha','MultinomialNB TFIDF',best_parm_mnb_tfidf['alpha']);
vals_mnb.set_value('alpha','MultinomialNB W2V',best_parm_mnb_w2v['alpha']);
display(vals_mnb)


clf_logreg = LogisticRegression(random_state = 0)
parameters_logreg = [{'C': (0.25, 0.5, 1.0),
                      'penalty': ('l1', 'l2')
              }]
grid_search = GridSearchCV(estimator = clf_logreg,param_grid= parameters_logreg,scoring='accuracy',cv=10,n_jobs=-1)

grid_search = grid_search.fit(X_train_cv,y_train_cv)
best_parm_logreg_cv = grid_search.best_params_

grid_search = grid_search.fit(X_train_tfidf,y_train_tfidf)
best_parm_logreg_tfidf = grid_search.best_params_

grid_search = grid_search.fit(X_train_w2v,y_train_w2v)
best_parm_logreg_w2v = grid_search.best_params_

index = ['C','penalty']
columns = ['Logistic Regression CV','Logistic Regression TFIDF','Logistic Regression W2V']
vals_logreg = pd.DataFrame(index=index,columns=columns)
vals_logreg.set_value('C','Logistic Regression CV',best_parm_logreg_cv['C']);
vals_logreg.set_value('penalty','Logistic Regression CV',best_parm_logreg_cv['penalty']);
vals_logreg.set_value('C','Logistic Regression TFIDF',best_parm_logreg_tfidf['C']);
vals_logreg.set_value('penalty','Logistic Regression TFIDF',best_parm_logreg_tfidf['penalty']);
vals_logreg.set_value('C','Logistic Regression W2V',best_parm_logreg_w2v['C']);
vals_logreg.set_value('penalty','Logistic Regression W2V',best_parm_logreg_w2v['penalty']);
display(vals_logreg)

vals_rf.to_csv('Outputs/vals_rf.csv', encoding='utf-8', index=False)
vals_mnb.to_csv('Outputs/vals_mnb.csv', encoding='utf-8', index=False)
vals_logreg.to_csv('Outputs/vals_logreg.csv', encoding='utf-8', index=False)
