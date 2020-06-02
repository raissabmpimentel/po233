#!env python

# Import libs
import pandas as pd
import numpy as np

# Import data
data_filename = "dados/pre_processed_data.csv"
data = pd.read_csv(data_filename)

# Separating target attribute
target = data['absences']
data = data.drop(columns=['absences'])
original = data

# Scikit Learn MLP Regression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# My training
seed = 1

# {G1, G2, G3}
data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=seed, test_size=0.2)
regression = MLPRegressor(random_state=seed, early_stopping=True, max_iter=10000, solver='sgd').fit(data_train, target_train)
score = regression.score(data_test, target_test)
print(f"[G1, G2, G3] - Score: {score}")

# {G1, G2}
data = original.drop(columns=['G3'])
data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=seed, test_size=0.2)
regression = MLPRegressor(random_state=seed, early_stopping=True, max_iter=10000, solver='sgd').fit(data_train, target_train)
score = regression.score(data_test, target_test)
print(f"[G1, G2]     - Score: {score}")

# {G1, G3}
data = original.drop(columns=['G2'])
data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=seed, test_size=0.2)
regression = MLPRegressor(random_state=seed, early_stopping=True, max_iter=10000, solver='sgd').fit(data_train, target_train)
score = regression.score(data_test, target_test)
print(f"[G1, G3]     - Score: {score}")

# {G2, G3}
data = original.drop(columns=['G1'])
data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=seed, test_size=0.2)
regression = MLPRegressor(random_state=seed, early_stopping=True, max_iter=10000, solver='sgd').fit(data_train, target_train)
score = regression.score(data_test, target_test)
print(f"[G2, G3]     - Score: {score}")

# {G1}
data = original.drop(columns=['G2', 'G3'])
data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=seed, test_size=0.2)
regression = MLPRegressor(random_state=seed, early_stopping=True, max_iter=10000, solver='sgd').fit(data_train, target_train)
score = regression.score(data_test, target_test)
print(f"[G1]         - Score: {score}")

# {G2}
data = original.drop(columns=['G1', 'G3'])
data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=seed, test_size=0.2)
regression = MLPRegressor(random_state=seed, early_stopping=True, max_iter=10000, solver='sgd').fit(data_train, target_train)
score = regression.score(data_test, target_test)
print(f"[G2]         - Score: {score}")

# {G3}
data = original.drop(columns=['G1', 'G2'])
data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=seed, test_size=0.2)
regression = MLPRegressor(random_state=seed, early_stopping=True, max_iter=10000, solver='sgd').fit(data_train, target_train)
score = regression.score(data_test, target_test)
print(f"[G3]         - Score: {score}")

# {}
data = original.drop(columns=['G1', 'G2', 'G3'])
data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=seed, test_size=0.2)
regression = MLPRegressor(random_state=seed, early_stopping=True, max_iter=10000, solver='sgd').fit(data_train, target_train)
score = regression.score(data_test, target_test)
print(f"[]           - Score: {score}")