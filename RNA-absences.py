#!env python

# Import libs
import pandas as pd
import numpy as np

def RNA_Regression(dataV, targetV, seed, algorithm, max_iteractions, es):
    # Regression
    data_train, data_test, target_train, target_test = train_test_split(dataV, targetV, random_state=seed, test_size=0.1)
    regression = MLPRegressor(random_state=seed, early_stopping=es, max_iter=max_iteractions, solver=algorithm).fit(data_train, target_train)
    
    # Regression Score
    score = regression.score(data_test, target_test)
    
    # Square Mean Error
    target_pred = regression.predict(data_test)
    sqe = mean_squared_error(target_test, target_pred)
    return (sqe, score)

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
from sklearn.metrics import mean_squared_error

# My training
seed = 1
algorithm = 'adam' # 'adam', 'sgd', 'lbfgs'
max_iteractions = 10000
early_stopping = True

# {G1, G2, G3}
(sqe, score) = RNA_Regression(data, target, seed, algorithm, max_iteractions, early_stopping)
print(f"[G1, G2, G3] - SQE: {sqe:.5f}    |    Score: {score:.5f}")

# {G1, G2}
data = original.drop(columns=['G3'])
(sqe, score) = RNA_Regression(data, target, seed, algorithm, max_iteractions, early_stopping)
print(f"[G1, G2]     - SQE: {sqe:.5f}    |    Score: {score:.5f}")

# {G1, G3}
data = original.drop(columns=['G2'])
(sqe, score) = RNA_Regression(data, target, seed, algorithm, max_iteractions, early_stopping)
print(f"[G1, G3]     - SQE: {sqe:.5f}    |    Score: {score:.5f}")

# {G2, G3}
data = original.drop(columns=['G1'])
(sqe, score) = RNA_Regression(data, target, seed, algorithm, max_iteractions, early_stopping)
print(f"[G2, G3]     - SQE: {sqe:.5f}    |    Score: {score:.5f}")

# {G1}
data = original.drop(columns=['G2', 'G3'])
(sqe, score) = RNA_Regression(data, target, seed, algorithm, max_iteractions, early_stopping)
print(f"[G1]         - SQE: {sqe:.5f}    |    Score: {score:.5f}")

# {G2}
data = original.drop(columns=['G1', 'G3'])
(sqe, score) = RNA_Regression(data, target, seed, algorithm, max_iteractions, early_stopping)
print(f"[G2]         - SQE: {sqe:.5f}    |    Score: {score:.5f}")

# {G3}
data = original.drop(columns=['G1', 'G2'])
(sqe, score) = RNA_Regression(data, target, seed, algorithm, max_iteractions, early_stopping)
print(f"[G3]         - SQE: {sqe:.5f}    |    Score: {score:.5f}")

# {}
data = original.drop(columns=['G1', 'G2', 'G3'])
(sqe, score) = RNA_Regression(data, target, seed, algorithm, max_iteractions, early_stopping)
print(f"[]           - SQE: {sqe:.5f}    |    Score: {score:.5f}")