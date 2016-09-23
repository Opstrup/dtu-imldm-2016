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
with open('dataset/forestfires.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    count = 2
    for row in reader:
#        print('Row num.' + str(count) + ' - ' +  str(row['temp']))
        temp.append(row['temp'])
        wind.append(row['wind'])
        rain.append(row['rain'])
        
        
data_to_plot = [temp]
plt.boxplot(np.array(temp).astype(np.float))
