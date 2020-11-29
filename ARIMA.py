# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:22:56 2020

@author: Digvijay Singh
"""

import matplotlib.pyplot as plt
#from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from Parameters import pickle_read, D, T, N, F
import time


def ARIMA_Predict(filename):
    start = time.time()
    X = pickle_read(filename)
    train, test = X[:N], X[N:]
    history = train
    print(len(history))
    predictions = list()

    for t in range(F):
        obs = test[t]
        history.append(obs)
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        if (len(history) > N):
            history.pop(0)
        plt.plot(range(t + 1), predictions, c = 'b', linewidth = 6)
        plt.plot(range(t + 1), test[:t+1], c = 'r', linewidth = 3)
        plt.show(block = False)
        plt.pause(0.1)
        plt.clf()
        print("MAE = ", abs((obs-yhat)))
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test[:F], predictions)
    print('Test MSE: %.3f' % error)
    print("Time = ", time.time() - start)

if __name__ == "__main__":
    ARIMA_Predict('Dataset/CarbMonox1.pickle')



