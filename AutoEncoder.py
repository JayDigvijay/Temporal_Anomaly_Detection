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
from Parameters import pickle_read, D, T
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout
from keras import regularizers
N = int((D*24)/(T))
n_features = 1
"""
N = Number of observations on which model to be trained
"""
raw = pickle_read('Ozone.pickle')
scaler = max(raw)
Data = [float(i/scaler) for i in raw]

train, test = Data[0:N], Data[N:2*N]
history = [x for x in train]
predictions = list()
observations = list()   

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

for t in range(len(test)):
    obs = scaler * test[t]
    observations.append(obs)
    X, Y = split_sequence(history, N-1)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    inputs = Input(shape = (X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation = 'relu',  return_sequences = True, kernel_regularizer = regularizers.l2(0.01))(inputs)
    L2 = LSTM(4, activation = 'relu', return_sequences = False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation = 'tanh', return_sequences = True)(L3)
    L5 = LSTM(16, activation = 'tanh', return_sequences = True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)
    model = Model(inputs = inputs, outputs = output)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=200, verbose=1)
    out = model.predict(X, verbose=0)
    yhat = scaler * out[0][0][0]
    print("Expected: ",obs, " Predicted: ",yhat)
    predictions.append(yhat)
    history.append(test[t])
    if (len(history) > N):
        history.pop(0)

error = mean_squared_error(observations, predictions)
print('Test MSE: %.3f' % error)
