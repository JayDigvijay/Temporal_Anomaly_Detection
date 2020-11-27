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

train, test = Data[:10*N], Data[10*N:]
history = train[-N:]
predictions = list()
observations = list()   

X, Y = split_sequence(train, n_steps)
X = X.reshape((X.shape[0], X.shape[1], n_features))
model = Sequential()
model.add(LSTM(50, activation='tanh', return_sequences = True, input_shape=(n_steps, n_features)))
#model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer=optimizer, loss='mse')
loss_history = model.fit(X, Y, epochs=100, verbose=1)
plt.plot(loss_history.history['loss'])
plt.show(block = False)
plt.pause(0.1)
plt.clf()



for t in range(len(test)): 
    pd = np.array(history)
    pd = pd.reshape((1, N, n_features))
    out = model.predict(pd, verbose=0)[0][0][0]
    pred = 1.725*out - 0.295
    if t >= N:
        obs = test[t - N]
        predictions.append(pred)
        plt.plot(range(t + 1 - N), predictions, c = 'g', linewidth = 6)
        observations.append(obs)
        print("Expected: ",obs, " Predicted: ",pred)
        print("MAE = ", abs(scaler*(obs-pred)))
        plt.plot(range(t + 1 - N), observations, c = 'r', linewidth = 3)
        plt.show(block = False)
        plt.pause(0.1)
        plt.clf()
    history.append(test[t])
    if (len(history) > N):
        history.pop(0)

L = len(observations)
error = mean_squared_error(observations, predictions[-L:])
print('Test MSE: %.3f' % error)
#shifting = N

