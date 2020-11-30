# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:22:56 2020

@author: Digvijay Singh
"""
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from Parameters import pickle_read, D, T, N, F
import time
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter

def ARIMA_Predict(filename):
    #start = time.time()
    X = pickle_read(filename)
    train, test = X[:N], X[N:]
    history = train
    #print(len(history))
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
        """
        plt.plot(range(t + 1), predictions, c = 'b', linewidth = 6)
        plt.plot(range(t + 1), test[:t+1], c = 'r', linewidth = 3)
        plt.show(block = False)
        plt.pause(0.1)
        plt.clf()
        """
        print("MAE = ", abs((obs-yhat)))
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test[:F], predictions)
    print('Prediction MSE: %.3f' % error)
    return error
    """
    x = np.linspace(1, t+1, t+1)
    sns.lineplot(x, predictions, color = 'b', linewidth = 6, label = "Predicted")
    hist = sns.lineplot(x, test[:F], color = 'r', linewidth = 3, label = "Observed")
    plt.rc('grid', linestyle="-", color='black')
    hist.set_yticklabels(hist.get_yticks(), size = 20, weight = 'bold')
    hist.set_xticklabels(hist.get_xticks(), size = 20,  weight = 'bold')
    hist.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    hist.set_xlabel("Time (hours)",fontsize=30, weight = 'bold')
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    hist.set_ylabel("NO2 Level".translate(SUB),fontsize=30, weight = 'bold')
    plt.legend(fontsize = 30, bbox_to_anchor=(0, -0.45, 1, 0), loc='lower left',ncol = 3, mode="expand", borderaxespad=0.)
    plt.grid(True)
    plt.show()
    """
    #print("Time = ", time.time() - start)

if __name__ == "__main__":
    MSE = ARIMA_Predict('Dataset/NitDiox1.pickle')
    


