# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:24:02 2016

@author: Opstrup
"""

from pylab import *
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import cross_validation
from toolbox_02450 import feature_selector_lr, bmplot

# importing the dataset
temp = []
wind = []
rain = []
x = []
y = []
FFMC = []
DMC = []
DC = []
ISI = []
RH = []
area = []

with open('../../dataset/forestfires.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    count = 2
    for row in reader:
        temp.append(row['temp'])
        wind.append(row['wind'])
        rain.append(row['rain'])
        x.append(row['X'])
        y.append(row['Y'])
        FFMC.append(row['FFMC'])
        DMC.append(row['DMC'])
        DC.append(row['DC'])
        ISI.append(row['ISI'])
        RH.append(row['RH'])
        area.append(row['area'])
     
all_data = { 'temp' : temp, 'wind' : wind, 'rain' : rain, 
            'x' : x, 'y' : y, 'FFMC' : FFMC, 'DMC' : DMC, 
            'DC' : DC, 'ISI' : ISI, 'RH' : RH, 'area' : area}