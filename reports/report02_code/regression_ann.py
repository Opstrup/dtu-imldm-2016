from pylab import *
from scipy.io import loadmat
import neurolab as nl
from sklearn import cross_validation
import scipy.linalg as linalg
from scipy import stats


X = np.matrix(np.loadtxt("../dataset/spam.data"))

attributeNames = np.loadtxt("../dataset/spambase.names", dtype='str').T
attributeNames = np.delete(attributeNames, len(attributeNames)-1, 0) # Deleting the attributename, of prediction
# Normalize data
X = stats.zscore(X);
N, M = X.shape
spam_class = X[:,57].T
#y = X[:,56]
#X = X[:,1:55]
y = X[:,19]
X = np.concatenate((X[:,1:18], X[:,20:55]), axis=1)
# X = np.delete(X, M-1,1) # Deleting spam class column
# X = np.delete(X, M-2, 1) # Removing the column to be predicted
N, M = X.shape

# Load Matlab data file and extract variables of interest
# mat_data = loadmat('..\\Data\\wine2.mat')
# attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
# X = np.matrix(mat_data['X'])
# y = X[:,10]             # alcohol contents (target)
# X = X[:,1:10]           # the rest of features
# N, M = X.shape
# C = 2




# Normalize and compute PCA (UNCOMMENT to experiment with PCA preprocessing)
#Y = stats.zscore(X,0);
#U,S,V = linalg.svd(Y,full_matrices=False)
#V = mat(V).T
# Components to be included as features
#k_pca = 3
#X = X*V[:,0:k_pca]
#N, M = X.shape


# Parameters for neural network classifier
n_hidden_units = 2      # number of hidden units
n_train = 5             # number of networks trained in each k-fold
learning_goal = 10    # stop criterion 1 (train mse to be reached)
max_epochs = 64         # stop criterion 2 (max epochs in training)
show_error_freq = 3     # frequency of training status updates

# K-fold crossvalidation
K = 10                   # only five folds to speed up this example
CV = cross_validation.KFold(N,K,shuffle=True)

# Variable for classification error
errors = np.zeros(K)
error_hist = np.zeros((max_epochs,K))
bestnet = list()
k=0
for train_index, test_index in CV:
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    # X_train = X[train_index]
    # y_train = y[train_index]
    # X_test = X[test_index]
    # y_test = y[test_index]

    best_train_error = 1e100
    for i in range(n_train):
        print('Training network {0}/{1}...'.format(i+1,n_train))
        # Create randomly initialized network with 2 layers
        ann = nl.net.newff([[-3, 3]]*M, [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
        if i==0:
            bestnet.append(ann)
        # train network
        train_error = ann.train(X_train, y_train, goal=learning_goal, epochs=max_epochs, show=show_error_freq)
        if train_error[-1]<best_train_error:
            bestnet[k]=ann
            best_train_error = train_error[-1]
            error_hist[range(len(train_error)),k] = train_error

    print('Best train error: {0}...'.format(best_train_error))
    y_est = bestnet[k].sim(X_test)
    errors[k] = np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]
    print('Best Net ' + str(k) + ': ' + str(bestnet[k]))
    k+=1

# Print the average least squares error
print('Mean-square error: {0}'.format(mean(errors)))
# Print the average classification error rate
print('Error rate: {0}%'.format(100*mean(errors)))


figure();
subplot(2,1,1); bar(range(0,K),errors); title('Mean-square errors');
subplot(2,1,2); plot(error_hist); title('Training error as function of BP iterations');
figure();
subplot(2,1,1); plot(y_est); plot(y_test.A); title('Last CV-fold: est_y vs. test_y');
subplot(2,1,2); plot((y_est-y_test).A); title('Last CV-fold: prediction error (est_y-test_y)'); show()
show()
