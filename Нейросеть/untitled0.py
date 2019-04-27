# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 02:22:38 2019

@author: Anton
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn import cross_validation, datasets, metrics, tree

df = pd.read_csv('C:/Users/Anton/Documents/4_курс/ВКР/Данные.csv', sep=';', engine='python')
df['nameOrig'] = df['nameOrig'].map(lambda x: x.lstrip('CM'))
df['nameDest'] = df['nameDest'].map(lambda x: x.lstrip('CM'))

X = df.drop(['isFraud','isFlaggedFraud'], axis=1)
y = df['isFraud']

# Делим выборку на train и test, все метрики будем оценивать на тестовом датасете

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, stratify=y,  test_size=0.33, random_state=42)

from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout

mod = Sequential()
mod.add(Dense(4, input_dim=9, activation='relu'))
mod.add(Dense(4, activation='relu'))
mod.add(Dense(1, activation='relu'))
mod.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

mod.fit(X_train, y_train, batch_size=16, epochs=2, verbose=2, validation_data=(X_test, y_test))