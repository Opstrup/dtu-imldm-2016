# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from pylab import *
import xlrd

a = 3
b = [1, 2, 3]
print('{0} times list {1} is {2}'.format(a, b, a * b))

def hello(name, n):
    M = np.random.rand(n, n)
    M = np.asmatrix(M)
    print('\nHello {0}!! This is your matrix: \n{1}'.format(name, M))
    
name = 'anders'
matrix_size = 3
hello(name, matrix_size)

# multiply arrays
a = np.random.rand(2,2)
b = np.array([[2, 0], [0, 2]])
print('here are some arrays multiplyed:\n {0}'.format(a * b))

# convert to matrices and multiply
a = np.asmatrix(a)
b = np.asmatrix(b)

np.dot(a,b)         # matrix multiplication
np.multiply(a,b)    # element-wise multiplication
print('here are some matrices multiplyed:\n {0}'.format(a * b))


x = np.linspace(0,4*np.pi,100)
noise = np.random.normal(0,0.2,100)
y = np.sin(x) + noise
plot(x, y, '.-r')
title('Sine with gaussian noise')
show()


print('Loading xls sheet')
# Load xls sheet with data
doc = xlrd.open_workbook('../02450Toolbox_Python/Data/nanonose.xls').sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(0,3,11)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(0,2,92)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(5)))

# Extract vector y, convert to NumPy matrix and transpose
y = np.mat([classDict[value] for value in classLabels]).T

# Preallocate memory, then extract excel data to matrix X
X = np.mat(np.empty((90,8)))
for i, col_id in enumerate(range(3,11)):
    X[:,i] = np.mat(doc.col_values(col_id,2,92)).T

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)