
# exercise 2.1.1
import numpy as np
import xlrd
from pylab import *

# Load xls sheet with data
doc = xlrd.open_workbook('dataset/forestfires.xls').sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(0,0,13)

# Preallocate memory, then extract excel data to matrix X
X = np.mat(np.empty((517,13)))
for i, col_id in enumerate(range(4,13)):
    X[:,i] = np.mat(doc.col_values(col_id,1,518)).T
    
#print X[:,0]

for i in range(0,9):
    figure()
    hist(np.array(X[:,i]),bins=80)
    xlabel(attributeNames[i+4])
    
    