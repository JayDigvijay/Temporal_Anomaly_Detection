# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 12:32:38 2020

@author: Digvijay Singh
"""
from numpy import array
from numpy import linalg as LA
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from Parameters import pickle_read, D, T, split_sequence
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout
from keras.regularizers import l2
import matplotlib.pyplot as plt
from keras import backend as K
import keras
optimizer = keras.optimizers.Adam(lr=0.05)


N = int((D*24)/(T))
n_features = 1
n_steps = N
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
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences = True, input_shape=(n, n_features)))
"""
model.add(LSTM(50, activation='relu', return_sequences = True))
model.add(Dropout(0.2))
"""
model.add(Dense(1, activation='tanh'))

model.compile(optimizer=optimizer, loss='mse')
loss_history = model.fit(X, Y, epochs=100, verbose=1)
plt.plot(loss_history.history['loss'])
plt.show(block = False)
plt.pause(0.1)
plt.clf()



for t in range(len(test)): 
    pd = np.array(history)
    pd = pd.reshape((1, n, n_features))
    out = model.predict(pd, verbose=0)[0][0][0]
    pred = out
    if t >= n:
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
#shifting = N

