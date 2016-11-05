import numpy as np
from scipy.stats import ttest_rel

knn = [ 8.42995169, 8.42791596,  8.71770104,  8.86259358,  8.40376721,  8.62110601,  8.79014731,  8.33132094,  8.28302342,  8.54865974]
ann = [ 0.07375271, 0.05869565,  0.08043478,  0.07608696,  0.05      ,  0.07826087,  0.07608696,  0.05652174,  0.07173913,  0.07173913]

print "t-value and p-value"
print (ttest_rel(np.asarray(knn),np.asarray(ann)*100))

print("KNN mean: {0}",np.asarray(knn).mean())
print("ANN mean: {0}",(np.asarray(ann)*100).mean())
