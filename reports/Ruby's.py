# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:50:35 2016

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:44:54 2016

@author: Opstrup
"""
#import xlrd
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr
from pydoc import help

# importing the dataset
temp = []
wind = []
rain = []
FFMC = []
DMC = []
DC = []
ISI = []
area = []
with open('dataset/forestfires.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    count = 2
    for row in reader:
#        print('Row num.' + str(count) + ' - ' +  str(row['temp']))
        temp.append(row['temp'])
        wind.append(row['wind'])
        rain.append(row['rain'])
        FFMC.append(row['FFMC'])
        DMC.append(row['DMC'])
        DC.append(row['DC'])
        ISI.append(row['ISI'])
        area.append(row['area'])
        
print "temp"
print pearsonr(np.array(temp).astype(np.float),np.array(wind).astype(np.float))
print pearsonr(np.array(temp).astype(np.float),np.array(rain).astype(np.float))
print pearsonr(np.array(temp).astype(np.float),np.array(FFMC).astype(np.float))
print pearsonr(np.array(temp).astype(np.float),np.array(DMC).astype(np.float))
print pearsonr(np.array(temp).astype(np.float),np.array(DC).astype(np.float))
print pearsonr(np.array(temp).astype(np.float),np.array(ISI).astype(np.float))
print pearsonr(np.array(temp).astype(np.float),np.array(area).astype(np.float))

print "wind"
print pearsonr(np.array(wind).astype(np.float),np.array(rain).astype(np.float))
print pearsonr(np.array(wind).astype(np.float),np.array(FFMC).astype(np.float))
print pearsonr(np.array(wind).astype(np.float),np.array(DMC).astype(np.float))
print pearsonr(np.array(wind).astype(np.float),np.array(DC).astype(np.float))
print pearsonr(np.array(wind).astype(np.float),np.array(ISI).astype(np.float))
print pearsonr(np.array(wind).astype(np.float),np.array(area).astype(np.float))

print "rain"
print pearsonr(np.array(rain).astype(np.float),np.array(FFMC).astype(np.float))
print pearsonr(np.array(rain).astype(np.float),np.array(DMC).astype(np.float))
print pearsonr(np.array(rain).astype(np.float),np.array(DC).astype(np.float))
print pearsonr(np.array(rain).astype(np.float),np.array(ISI).astype(np.float))
print pearsonr(np.array(rain).astype(np.float),np.array(area).astype(np.float))

print "FFMC"
print pearsonr(np.array(FFMC).astype(np.float),np.array(DMC).astype(np.float))
print pearsonr(np.array(FFMC).astype(np.float),np.array(DC).astype(np.float))
print pearsonr(np.array(FFMC).astype(np.float),np.array(ISI).astype(np.float))
print pearsonr(np.array(FFMC).astype(np.float),np.array(area).astype(np.float))

print "DMC"
print pearsonr(np.array(DMC).astype(np.float),np.array(DC).astype(np.float))
print pearsonr(np.array(DMC).astype(np.float),np.array(ISI).astype(np.float))
print pearsonr(np.array(DMC).astype(np.float),np.array(area).astype(np.float))

print "DC"
print pearsonr(np.array(DC).astype(np.float),np.array(ISI).astype(np.float))
print pearsonr(np.array(DC).astype(np.float),np.array(area).astype(np.float))

print "ISI"
print pearsonr(np.array(ISI).astype(np.float),np.array(area).astype(np.float))