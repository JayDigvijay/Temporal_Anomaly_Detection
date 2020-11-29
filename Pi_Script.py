from ARIMA import ARIMA_Predict
from LSTM import LSTM_Predict
import os
import time

start = time.time()

directory = r'Dataset'

for filename in os.listdir(directory):
    if filename.endswith(".pickle"):
        print("Currently processing the file ", os.path.join(filename))
        filepath = 'Dataset/' + filename
        ARIMA_Predict(filepath)
        LSTM_Predict(filepath)
       
print("Total time = ", (time.time() - start) , "seconds")