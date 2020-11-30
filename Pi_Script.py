from ARIMA import ARIMA_Predict
from LSTM import LSTM_Predict
import os
import time
import csv
start = time.time()

directory = r'Dataset'
ARIMA_Dict = {}
LSTM_Dict = {}
for filename in os.listdir(directory):
    if filename.endswith(".pickle"):
        print("Currently processing the file ", os.path.join(filename))
        filepath = 'Dataset/' + filename
        file = filename[:-7]
        MSE1 = ARIMA_Predict(filepath)
        ARIMA_Dict[file] = MSE1 
        MSE2 = LSTM_Predict(filepath)
        LSTM_Dict[file] = MSE2


with open('LSTM_MSE.csv', 'w') as f:
    for key in LSTM_Dict.keys():
        f.write("%s,%s\n"%(key,LSTM_Dict[key]))  
with open('ARIMA_MSE.csv', 'w') as f:
    for key in ARIMA_Dict.keys():
        f.write("%s,%s\n"%(key,ARIMA_Dict[key]))    
print("Total time = ", (time.time() - start) , "seconds")