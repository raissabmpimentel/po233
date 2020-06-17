# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler

def treinamento(test_size, max_iter, solver, activation, random_state, verbose, tol):
    previsores_treinamento, previsores_teste, y_treinamento, y_teste = train_test_split(previsores, y, test_size=test_size, random_state=0)
    
    prep = StandardScaler()
            
    prep.fit(previsores_treinamento)
    
    regressor = MLPRegressor(verbose = verbose,
                                  max_iter=max_iter,
                                  solver = solver,
                                  activation=activation,
                                  tol=tol,
                                  random_state=random_state,
                                  )
    
    regressor.fit(prep.transform(previsores_treinamento), y_treinamento)
    previsoes = regressor.predict(prep.transform(previsores_teste))
    
    mse = mean_squared_error(y_teste, previsoes)
    score = regressor.score(prep.transform(previsores_teste), y_teste)
    return (mse, score)

#base = pd.read_csv('dados/pre_processed_data.csv')
#y = base['G3'].values

test_size = 0.2
max_iter = 1500
solver = 'adam'
activation = 'relu'
random_state = 4
verbose = False
tol = 1e-5

base = pd.read_csv('dados/student-por.csv', sep = ";")
base = base.drop(['school'], axis=1)
base = base.replace(['LE3', 'GT3'], [0,1])
items = ['sex', 'address','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher',
         'internet','romantic']

for item in items:
    base = pd.concat([base,pd.get_dummies(base[item], prefix=item)],axis=1)
    base = base.drop([item],axis=1)

y = base['G3'].values

# [G1, G2]

print(f"[G1, G2]:\n")
previsores = base.drop(columns=['G3']).values
precisao_1, score_1 = treinamento(test_size, max_iter, solver, activation, random_state, verbose, tol)

print(f"MSE: {precisao_1} Score: {score_1}\n")

# [G1]
print(f"[G1]:\n")
previsores = base.drop(columns=['G2','G3']).values
precisao_2, score_2 = treinamento(test_size, max_iter, solver, activation, random_state, verbose, tol)
print(f"MSE: {precisao_2} Score: {score_2}\n")

    
# [G2]
print(f"[G2]:\n")
previsores = base.drop(columns=['G1','G3']).values
precisao_3, score_3 = treinamento(test_size, max_iter, solver, activation, random_state, verbose, tol)
print(f"MSE: {precisao_3} Score: {score_3}\n")

# []
print(f"[]:\n")
previsores = base.drop(columns=['G1','G2','G3']).values
precisao_4, score_4 = treinamento(test_size, max_iter, solver, activation, random_state, verbose, tol)
print(f"MSE: {precisao_4} Score: {score_4}\n")