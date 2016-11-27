from pylab import *
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import numpy as np

# Load spam data
X = np.loadtxt('../../dataset/spam.data')
attributeNames = np.loadtxt("../../dataset/spambase.names",dtype="string").tolist()
classNames = ["Valid", "Spam"]
C = len(classNames)
N, M = X.shape
y = np.asmatrix(X[:,57].astype(int)).T
X = np.delete(X, M-1,1)
N, M = X.shape

# Perform hierarchical/agglomerative clustering on data matrix
Method = 'single'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 5
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
np.savetxt("cls_hierarchical.txt", cls)
figure(1)
clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

# Display dendrogram
max_display_levels=5
figure(2)
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()
