from pylab import *
from toolbox_02450 import clusterplot
from sklearn.mixture import GMM
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn import cross_validation

# Load spam data
X = np.loadtxt('../../data/spam.data')
attributeNames = np.loadtxt("../../data/spambase.names",dtype="string").tolist()
classNames = ["Valid", "Spam"]
C = len(classNames)
N, M = X.shape
y = np.asmatrix(X[:,57].astype(int)).T
X = np.delete(X, M-1,1)
N, M = X.shape
print str(X.shape)

# Number of Clusters
K = 15
# Hierarchical
Method = 'single'
Metric = 'seuclidean'
Z = linkage(X, method=Method, metric=Metric)
cls_h = fcluster(Z, criterion='maxclust', t=K)

# Gausian Mixture Model
cov_type = 'full'
reps = 10
gmm = GMM(n_components=K, covariance_type=cov_type, n_init=reps, params='wmc').fit(X)
cls_gmm = gmm.predict(X)
cds = gmm.means_
covs = gmm.covars_

# Evaluating GMM
pairs_gmm = map(lambda i: (cls_gmm[i],y.item(i)), range(N))
clusters_spam_gmm = dict()
for pair in pairs_gmm:
    if pair[0] not in clusters_spam_gmm:
        clusters_spam_gmm[pair[0]] = dict()
        clusters_spam_gmm[pair[0]]['spam'] = 0
        clusters_spam_gmm[pair[0]]['valid'] = 0
    clusters_spam_gmm[pair[0]]['spam' if pair[1] == 1 else 'valid'] += 1
    clusters_spam_gmm[pair[0]]['purity'] = float(max(clusters_spam_gmm[pair[0]]['spam'], clusters_spam_gmm[pair[0]]['valid']))/float(clusters_spam_gmm[pair[0]]['spam'] +  clusters_spam_gmm[pair[0]]['valid'])

# Evaluating Hierarchical
pairs_h = map(lambda i: (cls_h[i],y.item(i)), range(N))
clusters_spam_h = dict()
for pair in pairs_h:
    if pair[0] not in clusters_spam_h:
        clusters_spam_h[pair[0]] = dict()
        clusters_spam_h[pair[0]]['spam'] = 0
        clusters_spam_h[pair[0]]['valid'] = 0
    clusters_spam_h[pair[0]]['spam' if pair[1] == 1 else 'valid'] += 1
    clusters_spam_h[pair[0]]['purity'] = float(max(clusters_spam_h[pair[0]]['spam'], clusters_spam_h[pair[0]]['valid']))/float(clusters_spam_h[pair[0]]['spam'] +  clusters_spam_h[pair[0]]['valid'])

eval_file = open('eval_gmm_h_15', 'w+')
eval_file.write('GAUSIAN MIXTURE MODEL CLUSTERS\n')
eval_file.write(str(clusters_spam_gmm))
eval_file.write('\n\n')
eval_file.write('HIEARCHICAL CLUSTERS\n')
eval_file.write(str(clusters_spam_h))
eval_file.write('\n\n')
eval_file.write('GAUSIAN MIXTURE MODEL CLUSTER CENTERS\n')
eval_file.write(np.array_str(cds))
eval_file.write('\n\n')
eval_file.close()
