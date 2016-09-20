# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:44:54 2016

@author: Opstrup
"""
#import xlrd
import csv

# importing the dataset
with open('dataset/forestfires.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    count = 2
    for row in reader:
        print('Row num.' + str(count) + ' - ' +  str(row['temp']))
        count += 1