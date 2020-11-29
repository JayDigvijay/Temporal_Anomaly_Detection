# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:57:51 2020

@author: Digvijay Singh
"""
import pickle
from numpy import array
##################### PARAMETERS ##############################################

T = 1
''' 
T = Time period in hours between model training. The readings taken in this 
period (at 15 minute intervals) are aggregated to increase reliability.
'''

D = 10
'''
D = Number of days whose data is considered while generating the model
'''

N = int((D*24)/(T))
"""
N = Number of observations on which model to be trained
"""
n_predict = 100
F = 24 + n_predict
"""
n_predict = Number of observations to predict
"""
###############################################################################


###################### FUNCTIONS ##############################################

############# 1.PICKLING#######################################################
def pickle_read (file_name):
    pickfile = open(file_name, 'rb')
    var = pickle.load(pickfile)
    return var
    pickfile.close()
    
def pickle_write (var_name, file_name):
    pickfile = open(file_name, 'wb')
    pickle.dump(var_name, pickfile, protocol=pickle.HIGHEST_PROTOCOL)
    pickfile.close()
###############################################################################


################# 2. SEQUENCE GENERATOR FOR LSTM ##############################
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


