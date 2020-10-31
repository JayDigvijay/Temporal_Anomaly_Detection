# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 19:07:25 2020

@author: Digvijay Singh
"""
import pandas as pd
from Parameters import T , pickle_write


data = pd.read_csv('Dataset/pollutionData180926.csv')
Gas = list(data['ozone'])
#Gas = list(data['particullate_matter'])
#Gas = input(data['carbon_monoxide'])
#Gas = input(data['sulfure_dioxide'])
#Gas = input(data['nitrogen_dioxide'])

gas_name = 'Ozone'
Aggregate_Values = list()
k = 4*T
i = 0
while (i < len(Gas)):
    agg = 0
    for j in range(k):
        agg +=  Gas[i + j]
    Aggregate_Values.append(agg/4)
    i += k
    
pickle_write(Aggregate_Values, (gas_name + '.pickle'))