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
from Parameters import pickle_read, split_sequence, N, F
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout
from keras.regularizers import l2
import matplotlib.pyplot as plt
from keras import backend as K
import keras
import time


def LSTM_Predict(filename):
    start = time.time()
    n_features = 1
    raw = pickle_read(filename)
    scaler = max(raw)
    Data = [(i/scaler) for i in raw]

    train, test = Data[:N], Data[N:]
    n = int(N/120) # Number of data points to use for prediction
    history = train[-n:]
    predictions = list()
    observations = list()   

    X, Y = split_sequence(train, n)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = Sequential()
    model.add(LSTM(5, activation='tanh', return_sequences = True, input_shape=(n, n_features)))
    #model.add(LSTM(50, activation='relu', return_sequences = True))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='tanh'))
    optimizer = keras.optimizers.Adam(lr=0.05)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X, Y, epochs=100, verbose=1, validation_split = 0.1)

    for t in range(F): 
        history.append(test[t])
        if (len(history) > n):
            history.pop(0)
        pd = np.array(history)
        pd = pd.reshape((1, n, n_features))
        out = model.predict(pd, verbose=0)[0][n-1][0]
        pred = scaler * out
        #print(out.shape)
        
        obs = scaler * test[t]
        predictions.append(pred)
        plt.plot(range(t + 1), predictions, c = 'g', linewidth = 6)
        observations.append(obs)
        print("Expected: ",obs, " Predicted: ",pred)
        print("MAE = ", abs((obs-pred)))
        plt.plot(range(t + 1), observations, c = 'r', linewidth = 3)
        plt.show(block = False)
        plt.pause(0.1)
        plt.clf()
        

    #L = len(observations)
    error = mean_squared_error(observations, predictions)
    print('Test MSE: %.3f' % error)
    #shifting = N
    print("Time = ", time.time() - start)


if __name__ == "__main__":
    LSTM_Predict('Dataset/CarbMonox1.pickle')