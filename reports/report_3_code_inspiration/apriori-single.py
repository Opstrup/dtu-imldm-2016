# exercise 6.2.1


from pylab import *
from scipy.io import loadmat
from scipy import stats
from writeapriorifile import *

#### Load data from file(s)
X = np.loadtxt('../data/spam.data')
attributeNames = np.loadtxt("../data/spambase.names",dtype="string").tolist()
classNames = ["Valid", "Spam"]
C = len(classNames)
N, M = X.shape
y = np.matrix(X[:,57].astype(int)).T
X = np.delete(X, M-1,1)
N, M = X.shape
X = stats.zscore(X);

'''
#### PCA stuff to remove attributes
# Subtract mean from data
Y = X - np.ones((N,1))*X.mean(0)

# Create the PCA by finding the svd of Y (not X)
U,S,V = linalg.svd(Y,full_matrices=False)
V = mat(V).T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

# Project te centered data onto principal component space
Z = Y * V

Xorig = X
origN, origM = X.shape
X = np.asarray(Z[:,0:10])

N, M = X.shape
'''
#### Binarize the data into two columns (less than median and not less than)
Xmedians = np.median(X,0)

Xnew = np.zeros((N,M))
attrNames = []
for i in range(M):
    r = X[:,i]
    rFlags = r>Xmedians[i]
    r[rFlags] = 1
    r[~rFlags] = 0
    Xnew[:,i] = r
    attrNames.append(attributeNames[i])
    '''
    for j in range(origM):
        if (Xorig[:,j] == r).all():
            attrNames.append(attributeNames[j] + "_high")
            attrNames.append(attributeNames[j] + "_low")
    '''
attrNames.append("spam_indicator")
Xnew = np.concatenate((Xnew,y), axis=1)

WriteAprioriFile(Xnew, titles = attrNames)
#WriteAprioriFile(Xnew)
