import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import numpy as np
import math
from math import exp as exp
from math import factorial as fact
from scipy import stats 
from matplotlib.ticker import FormatStrFormatter

d1 = pd.read_csv('CPU_Usage.csv')
y1 = d1['CPU']
d2 = pd.read_csv('Memory_Usage.csv')
y2 = d2['Memory']


#y = y1 + y2 + y3
#print(np.mean(y), np.var(y))

x1 = np.linspace(1, len(y1), len(y1))
x2 = np.linspace(1, len(y2), len(y2))

fig, ax = plt.subplots()
plt.grid(True)

plt.rc('grid', linestyle="-", color='black')
ax.set_axisbelow(True)
plt.hist(y1, bins = 20, density = False, color = 'green', edgecolor='black', linewidth=1.2)
#plt.hist(y2, bins = 10, density = False, color = 'orange', edgecolor='black', linewidth=1.2)


ax.set_xticklabels(ax.get_xticks(), fontsize=25, weight = 'bold')
ax.set_yticklabels(ax.get_yticks(),fontsize=25, weight = 'bold')
ax.set_xlabel('CPU Usage (%)', fontsize=35, weight = 'bold')
#ax.set_xlabel('Memory Usage (%)', fontsize=35, weight = 'bold')
ax.set_ylabel('Frequency', fontsize=35, weight = 'bold')
plt.show()