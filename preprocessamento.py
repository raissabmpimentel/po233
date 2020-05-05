# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
        

base = pd.read_csv('dados/student-por.csv', sep = ";")

base.drop(['school'], axis=1, inplace=True)

base.replace(['LE3', 'GT3'], [0,1], inplace=True)

# onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,4,7,8,9,10,14,15,16,17,18,19,20,21])],remainder='passthrough')
# base_np = onehotencorder.fit_transform(base)

items = ['sex', 'address','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher',
         'internet','romantic']

for item in items:
    base = pd.concat([base,pd.get_dummies(base[item], prefix=item)],axis=1)
    base.drop([item],axis=1, inplace=True)
    

scaler = StandardScaler()

columns = base.columns.tolist()

for column in columns:
    value = base[column].values.reshape(-1, 1)
    scaled_array = scaler.fit_transform(value)
    base[column] = scaled_array

base.to_csv('dados/pre_processed_data.csv', index=False)
