# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 12:32:38 2020

@author: Digvijay Singh
"""
from numpy import array
from numpy import linalg as LA
from keras.models import Sequential
from keras.layers import LSTM, Input, TimeDistributed, RepeatVector
from keras.models import Model
from keras.layers import Dense
from Parameters import pickle_read, D, T, split_sequence
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout
from keras import regularizers
import keras
import matplotlib.pyplot as plt
import numpy as np
optimizer = keras.optimizers.Adam(lr=0.05)

N = int((D*24)/(T))
n_features = 1
"""
N = Number of observations on which model to be trained
"""
raw = pickle_read('Ozone.pickle')
scaler = max(raw)
Data = [float(i/scaler) for i in raw]

train, test = Data[:N], Data[N:]
n = int(N/10)
history = train[-n:]
predictions = list()
observations = list()  

X, Y = split_sequence(train, n)
X = X.reshape((X.shape[0], X.shape[1], n_features))
inputs = Input(shape = (X.shape[1], X.shape[2]))
L1 = LSTM(16, activation = 'relu',  return_sequences = True)(inputs)
L2 = LSTM(4, activation = 'relu', return_sequences = True)(L1)
L3 = RepeatVector(X.shape[1])(L2)
L4 = LSTM(4, activation = 'tanh', return_sequences = True)(L3)
L5 = LSTM(16, activation = 'tanh', return_sequences = True)(L4)
output = TimeDistributed(Dense(X.shape[2]))(L5)
model = Model(inputs = inputs, outputs = output)
model.compile(optimizer='adam', loss='mse')
loss_history = model.fit(X, Y, epochs=100, verbose=1)
plt.plot(loss_history.history['loss'])
plt.show(block = False)
plt.pause(1)
plt.clf()

for t in range(len(test)):
    pd = np.array(history)
    pd = pd.reshape((1, n, n_features))
    out = model.predict(pd, verbose=0)[0][0][0]
    
    if t >= n:
        pred = out
        obs = test[t - n]
        predictions.append(pred)
        plt.plot(range(t + 1 - n), predictions, c = 'g', linewidth = 6)
        observations.append(obs)
        print("Expected: ",obs, " Predicted: ",pred)
        print("MAE = ", abs(scaler*(obs-pred)))
        plt.plot(range(t + 1 - n), observations, c = 'r', linewidth = 3)
        plt.show(block = False)
        plt.pause(0.1)
        plt.clf()
    history.append(test[t])
    if (len(history) > n):
        history.pop(0)

L = len(observations)
error = mean_squared_error(observations, predictions[-L:])
print('Test MSE: %.3f' % error)
