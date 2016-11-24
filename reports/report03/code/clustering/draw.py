# exercise 10.1.1

from pylab import *

from scipy.io import loadmat

from toolbox_02450 import clusterplot

from sklearn.mixture import GMM

# Load spam data
X = np.loadtxt('D:/github-Machine-Learning/reports/dataset/spam.data')
attributeNames = np.loadtxt("D:/github-Machine-Learning/reports/dataset/spambase.names",dtype="string").tolist()
classNames = ["Valid", "Spam"]
C = len(classNames)
N, M = X.shape
y = np.asmatrix(X[:,57].astype(int)).T
print X.shape
print "str "+str(X.shape)

# Load Matlab data file and extract variables of interest







# Number of clusters

K = 5

cov_type = 'full'       # type of covariance, you can try out 'diag' as well

reps = 10                # number of fits with different initalizations, best result will be kept



# Fit Gaussian mixture model

gmm = GMM(n_components=K, covariance_type=cov_type, n_init=reps, params='wmc').fit(X)

cls = gmm.predict(X)    # extract cluster labels

cds = gmm.means_        # extract cluster centroids (means of gaussians)
print cds
covs = gmm.covars_      # extract cluster shapes (covariances of gaussians)



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



# Plot results:

figure(figsize=(14,9))

clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)

show()