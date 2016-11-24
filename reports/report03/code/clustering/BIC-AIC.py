# exercise 10.1.5
from pylab import *
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from sklearn.mixture import GMM
from sklearn import cross_validation
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


# Range of K's to try
KRange = range(1,20)
T = len(KRange)

covar_type = 'full'     # you can try out 'diag' as well
reps = 3                # number of fits with different initalizations, best result will be kept

# Allocate variables
BIC = np.zeros((T,1))
AIC = np.zeros((T,1))
CVE = np.zeros((T,1))

# K-fold crossvalidation
CV = cross_validation.KFold(N,10,shuffle=True)

for t,K in enumerate(KRange):
        print('Fitting model for K={0}\n'.format(K))

        # Fit Gaussian mixture model
        gmm = GMM(n_components=K, covariance_type=covar_type, n_init=reps, params='wmc').fit(X)

        # Get BIC and AIC
        BIC[t,0] = gmm.bic(X)
        AIC[t,0] = gmm.aic(X)

        # For each crossvalidation fold
        for train_index, test_index in CV:

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GMM(n_components=K, covariance_type=covar_type, n_init=reps, params='wmc').fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score(X_test).sum()
            

# Plot results

figure(1); hold(True)
plot(KRange, BIC)
plot(KRange, AIC)
plot(KRange, 2*CVE)
legend(['BIC', 'AIC', 'Crossvalidation'])
xlabel('K')
show()