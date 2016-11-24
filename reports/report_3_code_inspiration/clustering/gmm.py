from pylab import *
from toolbox_02450 import clusterplot
from sklearn.mixture import GMM
import numpy as np

# Load spam data
X = np.loadtxt('../../dataset/spam.data')
attributeNames = np.loadtxt("../../dataset/spambase.names",dtype="string").tolist()
classNames = ["Valid", "Spam"]
C = len(classNames)
N, M = X.shape
y = np.asmatrix(X[:,57].astype(int)).T
#X = stats.zscore(X);

# mat_data = loadmat('../Data/synth1.mat')
# X = np.matrix(mat_data['X'])
# y = np.matrix(mat_data['y'])
# attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
# classNames = [name[0][0] for name in mat_data['classNames']]
# N, M = X.shape
# C = len(classNames)

# PCA
c1 = X[:,57].T
c1 = c1.astype(bool)
c2 = X[:,57].T
c2 = (c2 + 1) % 2
c2 = c2.astype(bool)
X = np.delete(X, M-1,1)
Y = X - np.ones((N,1))*X.mean(0)
U,S,V = linalg.svd(Y,full_matrices=False)
V = mat(V).T
rho = (S*S) / (S*S).sum()
Z = Y * V
z = np.asarray(Z[:,0:2])

# Number of clusters
K = 5
cov_type = 'diag'       # type of covariance, you can try out 'diag' as well
reps = 10                # number of fits with different initalizations, best result will be kept

# Fit Gaussian mixture model
gmm = GMM(n_components=K, covariance_type=cov_type, n_init=reps, params='wmc').fit(X)
cls = gmm.predict(X)    # extract cluster labels
cds = gmm.means_        # extract cluster centroids (means of gaussians)
print cds
covs = gmm.covars_      # extract cluster shapes (covariances of gaussians)

print "Z Shape: " + str(Z.shape)
print "z Shape: " + str(z.shape)
print "Y Shape: " + str(Y.shape)
print "V Shape: " + str(V.shape)
print "X Shape: " + str(X.shape)
print "y Shape: " + str(y.shape)
print "cls Shape: " + str(cls.shape)
print "cds Shape: " + str(cds.shape)
print "covs Shape: " + str(covs.shape)

if cov_type == 'diag':
    new_covs = np.zeros([K,M,M])
    count = 0
    for elem in covs:
        temp_m = np.zeros([M,M])
        for i in range(len(elem)):
            temp_m[i][i] = elem[i]
        new_covs[count] = temp_m
        count += 1
    covs = new_covs

np.savetxt("cls.txt",cls)
result = [ abs(cls[i] - y[i]) for i in range(len(cls))]
np.savetxt("result.txt",result)
print "Result mean: " + str(np.mean(result))

#Plot results:
figure(figsize=(10,6))
clusterplot(z, clusterid=cls, centroids=cds, y=y, covars=5)


show()
