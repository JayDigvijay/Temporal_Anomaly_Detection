# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:57:51 2020

@author: Digvijay Singh
"""
import pickle

##################### PARAMETERS ##############################################

T = 1
''' 
T = Time period in hours between model training. The readings taken in this 
period (at 15 minute intervals) are aggregated to increase reliability.
'''

D = 30
'''
D = Number of days whose data is considered while generating the model
'''

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