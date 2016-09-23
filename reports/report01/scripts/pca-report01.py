# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:44:54 2016

@author: Opstrup
"""
#import xlrd
import csv
import matplotlib.pyplot as plt
import numpy as np

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

key = 1
for key, data in all_data.items():
    plot_data = np.array(data).astype(np.float)
    fig = plt.figure(key, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(plot_data)
    fig.savefig(key +'.png', bbox_inches='tight')
    key =+ 1
