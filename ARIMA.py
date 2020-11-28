# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:22:56 2020

@author: Digvijay Singh
"""

import matplotlib.pyplot as plt
#from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from Parameters import pickle_read, D, T

N = int((D*24)/(T)) 
"""
N = Number of observations on which model to be trained
"""
X = pickle_read('Ozone.pickle')

train, test = X[0:N], X[N:len(X)]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
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
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
'''
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
'''



