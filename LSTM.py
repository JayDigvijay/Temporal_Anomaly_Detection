# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 12:32:38 2020

@author: Digvijay Singh
"""
from numpy import array
from numpy import linalg as LA
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from Parameters import pickle_read, D, T
N = int((D*24)/(T))
n_features = 1

"""
N = Number of observations on which model to be trained
"""
raw = pickle_read('Ozone.pickle')
Sequence = [float(i)/max(raw) for i in raw]

train, test = Sequence[0:2*N], Sequence[2*N:len(Sequence)]
history = Sequence[0:N]
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
X, Y = split_sequence(train, 5)
X = X.reshape((X.shape[0], X.shape[1], n_features))
model = Sequential()
model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(5, n_features)))
model.add(LSTM(50, activation='tanh'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, Y, epochs=200, verbose=1)
for t in range(len(test)):
    obs = max(raw) * test[t]
    hist_arr = array(history)
    hist_arr = hist_arr.reshape((1, N, n_features))
    out = model.predict(hist_arr, verbose=0)
    yhat = max(raw) * out[0][0]
    print("Expected: ",obs, " Observed: ",yhat)
    predictions.append(yhat)
    history.append(obs)
    if (len(history) > N):
        history.pop(0)

error = mean_squared_error(observations, predictions)
print('Test MSE: %.3f' % error)

