
# exercise 2.1.1
import numpy as np
import xlrd
from pylab import *

# Load xls sheet with data
doc = xlrd.open_workbook('../dataset/forestfires.xls').sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(0,0,13)

# Preallocate memory, then extract excel data to matrix X
XY = np.mat(np.empty((517,9)))
for i, col_id in enumerate(range(4,13)):
    XY[:,i] = np.mat(doc.col_values(col_id,1,518)).T
    
X = XY[:,:-1]
area = []
# Make "area" into concrete variable: 0 - no fire;  0 - 20 small fire;  >20 - big fire
for i in XY[:,-1]:
    if i==0:
        area.append(0)
    else:
        if i<=20:
            area.append(1)
        else:
            area.append(2)
y = np.mat(area).T
classNames = ['No fire', 'Small fire', 'Big fire']
    
#print X[0,:]

'''for i in range(0,9):
    figure()
    hist(np.array(X[:,i]),bins=80)
    xlabel(attributeNames[i+4])
'''
    