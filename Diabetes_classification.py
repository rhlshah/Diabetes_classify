# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:45:40 2019

@author: Rahil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfx=pd.read_csv('./Dataset/Diabetes_XTrain.csv')
dfy=pd.read_csv('./Dataset/Diabetes_YTrain.csv')

x=dfx.values
y=dfy.values

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x,y)

X_test=pd.read_csv('./Dataset/Diabetes_XTest.csv')
x_t=X_test.values

output=classifier.predict(x_t)