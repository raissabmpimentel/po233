# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:45:50 2020

@author: raiss
"""

import pandas as pd
import Orange
import matplotlib.pyplot as plt

def plotCLA(title):
    n_splits = 10
    df_score = df_metrics.rank(axis=1, ascending=False)
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
    
def plotREG(title):
    n_splits = 10
    df_score = df_metrics.rank(axis=1, ascending=False)
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
    
df_metrics = pd.read_csv('output-classification/df_metrics_class_from_G1_G2.csv', index_col=0)
plotCLA("[G1, G2]")
df_metrics = pd.read_csv('output-classification/df_metrics_class_from_G2.csv', index_col=0)
plotCLA("[G2]")
df_metrics = pd.read_csv('output-classification/df_metrics_class_from_G1.csv', index_col=0)
plotCLA("[G1]")

df_metrics = pd.read_csv('output-regression/df_metrics_G3_from_G1_G2.csv', index_col=0)
plotREG("[G1, G2]")
df_metrics = pd.read_csv('output-regression/df_metrics_G3_from_G2.csv', index_col=0)
plotREG("[G2]")
df_metrics = pd.read_csv('output-regression/df_metrics_G3_from_G1.csv', index_col=0)
plotREG("[G1]")