# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:43:29 2020

@author: raiss
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
import Orange
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare

def preprocess(base):

    base = base.drop(['school'], axis=1)
    
    base = base.replace(['LE3', 'GT3'], [0,1])
    
    items = ['sex', 'address','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher',
             'internet','romantic']
    
    for item in items:
        base = pd.concat([base,pd.get_dummies(base[item], prefix=item)],axis=1)
        base = base.drop([item],axis=1)
    
    base.loc[base['G3'] < 10, 'G3'] = 0
    base.loc[base['G3'] >= 10, 'G3'] = 1
    
    return base

def search(X,y,n_splits, random_state,algorithms, title):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state) # Deve considerar questões temporais
    gskf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state) # Poderia ser um hold out simples
    perf = accuracy_score # Poderia ser outra medida qualquer
    
    score = {}
    for algorithm in algorithms.keys():
        score[algorithm] = []
    
    for algorithm, (clf, parameters) in algorithms.items():
        for train, test in kf.split(X, y):
            prep = StandardScaler()
            prep.fit(X[train])
            best = GridSearchCV(clf, parameters, cv=gskf, scoring=make_scorer(perf))
            best.fit(prep.transform(X[train]), y[train])
            score[algorithm].append(perf(best.predict(prep.transform(X[test])), y[test]))
            
    df_metrics = pd.DataFrame.from_dict(score)
            
    df_score = pd.DataFrame.from_dict(score).rank(axis=1, ascending=False)
    
    stat, p = friedmanchisquare(*[grp for idx, grp in df_score.iteritems()])
    
    print('Statistics=%.3f, p=%f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
    	print('Same distributions (fail to reject H0)')
    else:
    	print('Different distributions (reject H0)')
        
    names = list(df_score.columns)   
    avranks = df_score.mean().values.tolist()
    print("avranks: ")
    print(df_score.mean())
    cd = Orange.evaluation.compute_CD(avranks, n_splits)
    print("cd:", cd)
    Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
    plt.title(title)
    #plt.show()
    plt.savefig('img/CLA' + title + '.png', bbox_inches='tight')
    
    return df_metrics

algorithms = {
    "MLP": (MLPClassifier(), { "activation": ("identity", "logistic", "tanh", "relu"), "solver": ("lbfgs", "adam"), "max_iter": [1000], "random_state": [1]}), 
    "SVM": (SVC(), {"C": [1, 10], "kernel": ("linear", "rbf"), "random_state": [2]}),
    "KNN": (KNeighborsClassifier(), { "n_neighbors": [1, 3, 5] }),
    "DT" : (DecisionTreeClassifier(), { "criterion": ("gini", "entropy"), "max_depth": [5, 10, 20], "random_state": [3]}),
    "RF" : (RandomForestClassifier(), { "criterion": ("gini", "entropy"), "max_depth": [5, 10, 20], "n_estimators": [30,50,100], "random_state": [4]})
}

base = pd.read_csv('dados/student-por.csv', sep = ";")

base = preprocess(base)

y = base['G3'].to_numpy()

n_splits = 10
random_state = 17

# [G1, G2]
print("[G1, G2]")
X = base.drop(columns = ['G3']).to_numpy()
df_G1_G2 = search(X,y,n_splits,random_state,algorithms,"[G1, G2]")
print(df_G1_G2)
print("Medias:")
print(df_G1_G2.mean())
print("Desvio Padrao: ")
print(df_G1_G2.std())
df_G1_G2.to_csv('output-classification/df_metrics_class_from_G1_G2.csv')

# [G1]
print("[G1]")
X = base.drop(columns = ['G2','G3']).to_numpy()
df_G1 = search(X,y,n_splits,random_state,algorithms,"[G1]")
print(df_G1)
print("Medias:")
print(df_G1.mean())
print("Desvio Padrao: ")
print(df_G1.std())
df_G1.to_csv('output-classification/df_metrics_class_from_G1.csv')

# [G2]
print("[G2]")
X = base.drop(columns = ['G1','G3']).to_numpy()
df_G2 = search(X,y,n_splits,random_state,algorithms,"[G2]")
print(df_G2)
print("Medias:")
print(df_G2.mean())
print("Desvio Padrao: ")
print(df_G2.std())
df_G2.to_csv('output-classification/df_metrics_class_from_G2.csv')


