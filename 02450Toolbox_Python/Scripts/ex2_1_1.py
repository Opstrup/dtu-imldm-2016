
# exercise 2.1.1
import numpy as np
import xlrd

# Load xls sheet with data
doc = xlrd.open_workbook('../Data/nanonose.xls').sheet_by_index(0)
forestfires = xlrd.open_workbook('../../reports/dataset/forestfires.xls').sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(0,3,11)
fireAttributeNames = forestfires.row_values(0,4,13)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(0,2,92)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(5)))

fireClassLabels = forestfires.col_values(2,1,518)
fireClassNames = sorted(set(fireClassLabels))
fireDict = dict(zip(fireClassNames,range(12)))

# Extract vector y, convert to NumPy matrix and transpose
y = np.mat([classDict[value] for value in classLabels]).T

firey = y = np.mat([fireDict[value] for value in fireClassLabels]).T

# Preallocate memory, then extract excel data to matrix X
X = np.mat(np.empty((90,8)))
for i, col_id in enumerate(range(3,11)):
    X[:,i] = np.mat(doc.col_values(col_id,2,92)).T

fireX = np.mat(np.empty((517, 9)))
for i, firecol_id in enumerate(range(4,13)):
    fireX[:,i] = np.mat(forestfires.col_values(firecol_id,1,518)).T

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

fireN = len(firey)
fireM = len(fireAttributeNames)
fireC = len(fireClassNames)

print(y)