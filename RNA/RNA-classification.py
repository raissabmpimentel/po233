# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def treinamento(test_size, max_iter, solver, activation, random_state, verbose, tol):
    previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=test_size, random_state=0)
    
    porc_classe1 = np.where(classe_teste == 1)[0].shape[0]/classe_teste.shape[0]
    print(f"porc_classe1: {porc_classe1}\n")
    classificador = MLPClassifier(verbose = verbose,
                                  max_iter=max_iter,
                                  solver = solver,
                                  activation=activation,
                                  tol=tol,
                                  random_state=random_state)
    
    classificador.fit(previsores_treinamento, classe_treinamento)
    previsoes = classificador.predict(previsores_teste)
    
    precisao = accuracy_score(classe_teste, previsoes)
    return(precisao)

base = pd.read_csv('../dados/pre_processed_data.csv')
base_antes = pd.read_csv('dados/student-por.csv', sep=';')
base['G3'] = base_antes['G3']
base.loc[base['G3'] < 10, 'G3'] = 0
base.loc[base['G3'] >= 10, 'G3'] = 1
classe = base['G3'].values

test_size = 0.2
max_iter = 600
solver = 'adam'
activation = 'relu'
random_state = 4
verbose = False
tol = 1e-5

# [G1, G2]

print(f"[G1, G2]:\n")
previsores = base.drop(columns=['G3']).values
precisao_1 = treinamento(test_size, max_iter, solver, activation, random_state, verbose, tol)
print(f"Score: {precisao_1}\n")

# [G1]
print(f"[G1]:\n")
previsores = base.drop(columns=['G2','G3']).values
precisao_2 = treinamento(test_size, max_iter, solver, activation, random_state, verbose, tol)
print(f"Score: {precisao_2}\n")
    
# [G2]
print(f"[G2]:\n")
previsores = base.drop(columns=['G1','G3']).values
precisao_3 = treinamento(test_size, max_iter, solver, activation, random_state, verbose, tol)
print(f"Score: {precisao_3}\n")

# []
print(f"[]:\n")
previsores = base.drop(columns=['G1','G2','G3']).values
precisao_4 = treinamento(test_size, max_iter, solver, activation, random_state, verbose, tol)
print(f"[] - Score: {precisao_4}\n")