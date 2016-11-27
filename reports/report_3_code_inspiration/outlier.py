# exercise 11.3.1
from pylab import *
from scipy.io import loadmat
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import sys

X = np.loadtxt('../dataset/spam.data')
attributeNames = np.loadtxt("../dataset/spambase.names",dtype="string").tolist()
classNames = ["Valid", "Spam"]
C = len(classNames)
N, M = X.shape
y = np.asmatrix(X[:,57].astype(int)).T
#X = np.delete(X, M-1,1)
N, M = X.shape
#X = stats.zscore(X);


magic = 4601

### Gausian Kernel density estimator
# cross-validate kernel width by leave-one-out-cross-validation
# (efficient implementation in gausKernelDensity function)
# evaluate for range of kernel widths
widths = X.var(axis=0).max() * (2.0**np.arange(-10,3))
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
   print str(i+1)+"/"+str(len(widths))+""
   density, log_density = gausKernelDensity(X,w)
   logP[i] = log_density.sum()
val = logP.max()
ind = logP.argmax()

width=widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# evaluate density for estimated width
density, log_density = gausKernelDensity(X,width)

density = np.concatenate((density, y), axis=1)

# Sort the densities
i = (density[:,0].argsort(axis=0)).ravel()
density = density[i][0]

# Fix scaling for CDF
gauss_cdf = np.cumsum(density[:,1]).T
gauss_cdf = gauss_cdf*(max(density[:,0])/max(density[:,1]))
gauss_cdf = gauss_cdf*(max(density[:,0])/max(gauss_cdf))

# Plot density estimate of outlier score
figure(1)
plot(range(magic),density[:magic,0], label="Density")
#plot(range(magic),gauss_cdf[:magic],'r--', label="CDF")
legend(loc="upper left")
title('Density estimate')

### K-neighbors density estimator
# Neighbor to use:
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
x = np.matrix(1./(D.sum(axis=1)/K)).T
density = np.concatenate((x, y), axis=1)
# Sort the scores
i = (density[:,0].argsort(axis=0)).ravel()
density = density[i][0]

d = density[:,0]
d_max = np.nanmax(d[d != inf])
# Fix scaling for CDF
cdf = np.cumsum(density[:,1]).T
cdf = cdf*(d_max/max(cdf))

# Plot k-neighbor estimate of outlier score (distances)
figure(3)
plot(range(magic),density[:magic,0], label="Density")
plot(range(magic),cdf[:magic],'r--', label="CDF")
legend(loc="upper left")
title('KNN density: Outlier score')

### K-nearest neigbor average relative density
# Compute the average relative density

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
x = 1./(D.sum(axis=1)/K)
avg_rel_density = np.matrix(x/(x[i[:,1:]].sum(axis=1)/K)).T

density = np.concatenate((avg_rel_density, y), axis=1)

# Sort the avg.rel.densities
i = (density[:,0].argsort(axis=0)).ravel()
density = density[i][0]

cdf = np.cumsum(density[:,1]).T
cdf = cdf*(max(density[:,0])/max(cdf))

# Plot k-neighbor estimate of outlier score (distances)
figure(5)
plot(range(magic),density[:magic,0], label="Density")
plot(range(magic),cdf[:magic],'r--', label="CDF")
legend(loc="upper left")
title('KNN average relative density: Outlier score')

### Distance to 5'th nearest neighbor outlier score
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

# Outlier score
score = D[:,K-1]
density = np.concatenate((np.matrix(score).T,y), axis=1)
# Sort the scores
i = (density[:,0].argsort(axis=0)).ravel()
density = np.flipud(density[i[::-1]][0])

cdf = np.cumsum(density[:,1]).T
cdf = cdf*(max(density[:,0])/max(cdf))

# Plot k-neighbor estimate of outlier score (distances)
figure(7)
plot(range(magic),density[:magic,0], label="Density")
plot(range(magic),cdf[:magic],'r--', label="CDF")
legend(loc="upper left")
title('5th neighbor distance: Outlier score')
show()

