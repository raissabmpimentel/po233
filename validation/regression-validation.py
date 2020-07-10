#!/usr/bin/env python
# coding: utf-8

# Imports
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ShuffleSplit, KFold
from sklearn.metrics import accuracy_score, make_scorer, mean_squared_error
import Orange
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare

# Function: Preprocessing
def preprocess(base):

    base = base.drop(['school'], axis=1)
    base = base.replace(['LE3', 'GT3'], [0,1])
    items = ['sex', 'address','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher',
             'internet','romantic']
    
    for item in items:
        base = pd.concat([base,pd.get_dummies(base[item], prefix=item)],axis=1)
        base = base.drop([item],axis=1)
        
    return base



# Function: Algorithm Search
def search(X, y, n_splits, random_state, algorithms, title):
    #kf = SortedStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state) # Deve considerar questões temporais
    #gskf = SortedStratifiedKFold(n_splits=3, shuffle=True, random_state=random_state) # Poderia ser um hold out simples
    #gskf = ShuffleSplit(n_splits=3, random_state=random_state)
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    gskf = KFold(n_splits=3, shuffle=True, random_state=random_state)
    perf = mean_squared_error # Poderia ser outra medida qualquer
    
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
    df_score = pd.DataFrame.from_dict(score).rank(axis=1, ascending=True)
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
        plt.savefig('img/REG' + title + '.png', bbox_inches='tight')
    
    return df_metrics

algorithms = {
    "MLP": (MLPRegressor(), { "activation": ["identity", "tanh", "relu"], "solver": ["lbfgs", "adam"], "max_iter": [1500], "random_state": [1]}),
    "SVM": (SVR(), {"C": [1.0], "kernel": ("linear", "rbf", "poly", "sigmoid")}),
    "KNN": (KNeighborsRegressor(), { "n_neighbors": [1, 3, 5] }),
    "DT" : (DecisionTreeRegressor(), { "criterion": ("mse", "friedman_mse", "mae"), "max_depth": [5, 10, 20], "random_state": [3]}),
    "RF" : (RandomForestRegressor(), { "criterion": ("mse", "mae"), "max_depth": [5, 10, 20], "n_estimators": [30,50,100], "random_state": [4]})
}


# Preprocessing
base = pd.read_csv('../dados/student-por.csv', sep = ";")
base = preprocess(base)

# k-fold numbers
n_splits = 10
random_state = 17


# --- G3 Regression ---
y = base['G3'].to_numpy()

# [G1, G2]
print("[G1, G2]")
X = base.drop(columns = ['G3']).to_numpy()
df_G3_from_G1_G2 = search(X,y,n_splits,random_state,algorithms,"[G1, G2]")
print(df_G3_from_G1_G2)
print("Medias:")
print(df_G3_from_G1_G2.mean())
print("Desvio Padrao: ")
print(df_G3_from_G1_G2.std())
df_G3_from_G1_G2.to_csv('../output-regression/df_metrics_G3_from_G1_G2.csv')

# [G1]
print("[G1]")
X = base.drop(columns = ['G3', 'G2']).to_numpy()
df_G3_from_G1 = search(X,y,n_splits,random_state,algorithms,"[G1]")
print(df_G3_from_G1)
print("Medias:")
print(df_G3_from_G1.mean())
print("Desvio Padrao: ")
print(df_G3_from_G1.std())
df_G3_from_G1.to_csv('../output-regression/df_metrics_G3_from_G1.csv')

# [G2]
print("[G2]")
X = base.drop(columns = ['G3', 'G1']).to_numpy()
df_G3_from_G2 = search(X,y,n_splits,random_state,algorithms,"[G2]")
print(df_G3_from_G2)
print("Medias:")
print(df_G3_from_G2.mean())
print("Desvio Padrao: ")
print(df_G3_from_G2.std())
df_G3_from_G2.to_csv('../output-regression/df_metrics_G3_from_G2.csv')

# []
print("[]")
X = base.drop(columns = ['G3', 'G1', 'G2']).to_numpy()
df_G3 = search(X,y,n_splits,random_state,algorithms,"G3 from []")
print(df_G3)
print("Medias:")
print(df_G3.mean())
print("Desvio Padrao: ")
print(df_G3.std())
df_G3.to_csv('../output-regression/df_metrics_G3.csv')

# # --- Absences Regression ---
# y = base['absences'].to_numpy()

# # [G1, G2, G3]
# print("[G1, G2, G3]")
# X = base.drop(columns = ['absences']).to_numpy()
# df_absences_from_G1_G2_G3 = search(X,y,n_splits,random_state,algorithms,"absences from [G1, G2, G3]")
# print(df_absences_from_G1_G2_G3)
# print("Medias:")
# print(df_absences_from_G1_G2_G3.mean())
# print("Desvio Padrao: ")
# print(df_absences_from_G1_G2_G3.std())
# df_absences_from_G1_G2_G3.to_csv('output-regression/df_metrics_absences_from_G1_G2_G3.csv')

# # [G1, G2]
# print("[G1, G2]")
# X = base.drop(columns = ['absences', 'G3']).to_numpy()
# df_absences_from_G1_G2 = search(X,y,n_splits,random_state,algorithms,"absences from [G1, G2]")
# print(df_absences_from_G1_G2)
# print("Medias:")
# print(df_absences_from_G1_G2.mean())
# print("Desvio Padrao: ")
# print(df_absences_from_G1_G2.std())
# df_absences_from_G1_G2.to_csv('output-regression/df_metrics_absences_from_G1_G2.csv')

# # [G1, G3]
# print("[G1, G3]")
# X = base.drop(columns = ['absences', 'G2']).to_numpy()
# df_absences_from_G1_G3 = search(X,y,n_splits,random_state,algorithms,"absences from [G1, G3]")
# print(df_absences_from_G1_G3)
# print("Medias:")
# print(df_absences_from_G1_G3.mean())
# print("Desvio Padrao: ")
# print(df_absences_from_G1_G3.std())
# df_absences_from_G1_G3.to_csv('output-regression/df_metrics_absences_from_G1_G3.csv')

# # [G2, G3]
# print("[G2, G3]")
# X = base.drop(columns = ['absences', 'G1']).to_numpy()
# df_absences_from_G2_G3 = search(X,y,n_splits,random_state,algorithms,"absences from [G2, G3]")
# print(df_absences_from_G2_G3)
# print("Medias:")
# print(df_absences_from_G2_G3.mean())
# print("Desvio Padrao: ")
# print(df_absences_from_G2_G3.std())
# df_absences_from_G2_G3.to_csv('output-regression/df_metrics_absences_from_G2_G3.csv')

# # [G1]
# print("[G1]")
# X = base.drop(columns = ['absences', 'G3', 'G2']).to_numpy()
# df_absences_from_G1 = search(X,y,n_splits,random_state,algorithms,"absences from [G1]")
# print(df_absences_from_G1)
# print("Medias:")
# print(df_absences_from_G1.mean())
# print("Desvio Padrao: ")
# print(df_absences_from_G1.std())
# df_absences_from_G1.to_csv('output-regression/df_metrics_absences_from_G1.csv')

# # [G2]
# print("[G2]")
# X = base.drop(columns = ['absences', 'G3', 'G1']).to_numpy()
# df_absences_from_G2 = search(X,y,n_splits,random_state,algorithms,"absences from [G2]")
# print(df_absences_from_G2)
# print("Medias:")
# print(df_absences_from_G2.mean())
# print("Desvio Padrao: ")
# print(df_absences_from_G2.std())
# df_absences_from_G2.to_csv('output-regression/df_metrics_absences_from_G2.csv')

# # [G3]
# print("[G3]")
# X = base.drop(columns = ['absences', 'G2', 'G1']).to_numpy()
# df_absences_from_G3 = search(X,y,n_splits,random_state,algorithms,"absences from [G3]")
# print(df_absences_from_G3)
# print("Medias:")
# print(df_absences_from_G3.mean())
# print("Desvio Padrao: ")
# print(df_absences_from_G3.std())
# df_absences_from_G3.to_csv('output-regression/df_metrics_absences_from_G3.csv')

# # []
# print("[]")
# X = base.drop(columns = ['absences', 'G2', 'G1', 'G3']).to_numpy()
# df_absences = search(X,y,n_splits,random_state,algorithms,"absences from []")
# print(df_absences)
# print("Medias:")
# print(df_absences.mean())
# print("Desvio Padrao: ")
# print(df_absences.std())
# df_absences.to_csv('output-regression/df_metrics_absences.csv')

