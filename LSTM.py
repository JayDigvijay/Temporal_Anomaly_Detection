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
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

def LSTM_Predict(filename):
    #start = time.time()
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
    model.add(LSTM(4, activation='tanh', return_sequences = True, input_shape=(n, n_features)))
    #model.add(LSTM(50, activation='relu', return_sequences = True))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='tanh'))
    optimizer = keras.optimizers.Adam(lr=0.05)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X, Y, epochs=50, verbose=1, validation_split = 0.1)

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
        """
        plt.plot(range(t + 1), observations, c = 'r', linewidth = 3)
        plt.show(block = False)
        plt.pause(0.1)
        plt.clf()
        """
        
    error = mean_squared_error(observations, predictions)
    print('Prediction MSE: %.3f' % error)
    """
    x = np.linspace(1, t+1, t+1)
    sns.lineplot(x, predictions, color = 'g', linewidth = 6, label = "Predicted")
    hist = sns.lineplot(x, observations, color = 'r', linewidth = 3, label = "Observed")
    plt.rc('grid', linestyle="-", color='black')
    hist.set_yticklabels(hist.get_yticks(), size = 20, weight = 'bold')
    hist.set_xticklabels(hist.get_xticks(), size = 20,  weight = 'bold')
    hist.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    hist.set_xlabel("Time (hours)",fontsize=30, weight = 'bold')
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    hist.set_ylabel("O3 Level".translate(SUB),fontsize=30, weight = 'bold')
    plt.legend(fontsize = 30, bbox_to_anchor=(0, -0.45, 1, 0), loc='lower left',ncol = 3, mode="expand", borderaxespad=0.)
    plt.grid(True)
    plt.show()
    """
    return error
    #print("Time = ", time.time() - start)
    

if __name__ == "__main__":
    MSE = LSTM_Predict('Dataset/Ozone1.pickle')