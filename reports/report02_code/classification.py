# exercise 6.2.1


from pylab import *
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import cross_validation,tree
from sklearn.neighbors import KNeighborsClassifier
from toolbox_02450 import feature_selector_lr, bmplot
from scipy import stats

# Load data from file(s)
X = np.loadtxt('../dataset/spam.data')
attributeNames = np.loadtxt("../dataset/spambase.names",dtype="string").tolist()
classNames = ["Valid", "Spam"]
C = len(classNames)
N, M = X.shape
y = X[:,57].astype(int)
X = np.delete(X, M-1,1)
N, M = X.shape

# Add offset attribute
#X = np.concatenate((np.ones((X.shape[0],1)),X),1)
#attributeNames = [u'Offset']+attributeNames
#M = M+1

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)

X = stats.zscore(X);

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = cross_validation.KFold(N,K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

internal_cross_validation = 10
dtc_runs = 20
dtc_error_test = np.empty((K,internal_cross_validation))
dtc_error_train = np.empty((K,internal_cross_validation))
dtc_test = np.empty(K)
dtc_train = np.empty(K)

max_neigh = 20
knn_errors = np.zeros((K,max_neigh))

k=0
fig = figure(17)
ax = fig.add_subplot(111)
ax.set_title('Decision tree folds (green: training, red: test)')
ax.set_xlabel('Tree depth')
ax.set_ylabel('Classification error rate (%)')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

for train_index, test_index in CV:
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Getting shapes for inner loop(s)
    N_train, M_train = X_train.shape
    inner_CV = cross_validation.KFold(N_train,internal_cross_validation,shuffle=True)
    j = 0


    # Innter cross-validation loop
    for train_train_index, test_test_index in inner_CV:
        print('Outer: {0}/{1} - Inner: {2}/{3}'.format(k+1,K,j+1,internal_cross_validation))


        # Extract variables
        X_train_train = X_train[train_train_index,:]
        y_train_train = y_train[train_train_index]
        X_test_test   = X_train[test_test_index,:]
        y_test_test   = y_train[test_test_index]

        N_train_train, M_train_train = X_train_train.shape
        N_test_test, M_test_test = X_test_test.shape

        # Decision Tree
        # Test pruning leves between 1 and 20 (included)
        de_test  = np.empty((dtc_runs,1))
        de_train = np.empty((dtc_runs,1))
        for i in range(dtc_runs):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=100, max_depth=(i+1))
            #dtc = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=2)
            dtc = dtc.fit(X_train_train,y_train_train)
            y_est_test = dtc.predict(X_test_test)
            y_est_train = dtc.predict(X_train_train)
            # Evaluate misclassification rate over train/test data (in this CV fold)
            misclass_rate_test = sum(np.abs(np.mat(y_est_test).T - y_test_test)) / float(len(y_est_test))
            misclass_rate_train = sum(np.abs(np.mat(y_est_train).T - y_train_train)) / float(len(y_est_train))
            de_test[i], de_train[i] = 100*misclass_rate_test/N_test_test, 100*misclass_rate_train/N_train_train

        # Calculate best pruning level
        dtc_error_test[k,j]  = de_test.argmin() +1
        dtc_error_train[k,j] = de_train.argmin() +1

        if j == internal_cross_validation - 1 :
            t=fig.add_subplot(5,2,k+1)
            t.plot(np.arange(1,dtc_runs+1),de_test, 'r',np.arange(1,dtc_runs+1),de_train,'g')
            for item in (t.get_yticklabels()):
                item.set_fontsize(9)



        j+=1


    # Decision Tree
    max_depth = dtc_error_test[k].mean()
    dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    dtc = dtc.fit(X_train,y_train)
    out = tree.export_graphviz(dtc, out_file='tree_deviance'+str(k)+'.gvz', feature_names=attributeNames)
    y_est_test = dtc.predict(X_test)
    y_est_train = dtc.predict(X_train)
    # Evaluate misclassification rate over train/test data (in this CV fold)
    misclass_rate_test = sum(np.abs(np.mat(y_est_test).T - y_test)) / float(len(y_est_test))
    misclass_rate_train = sum(np.abs(np.mat(y_est_train).T - y_train)) / float(len(y_est_train))
    dtc_test[k], dtc_train[k] = misclass_rate_test, misclass_rate_train


    # KNN
    # Cross-validation not necessary. Instead, compute matrix of nearest neighbor
    # distances between each pair of data points ..
    knclassifier = KNeighborsClassifier(n_neighbors=max_neigh+1).fit(X_train, ravel(y_train))
    neighbors = knclassifier.kneighbors(X_train)
    # .. and extract matrix where each row contains class labels of subsequent neighbours
    # (sorted by distance)
    ndist, nid = neighbors[0], neighbors[1]
    nclass = y_train[nid].flatten().reshape(N_train,max_neigh+1)

    # Use the above matrix to compute the class labels of majority of neighbors
    # (for each number of neighbors l), and estimate the test errors.
    errors = np.zeros(max_neigh)
    nclass_count = np.zeros((N_train,C))
    for l in range(1,max_neigh+1):
        for c in range(C):
            nclass_count[:,c] = sum(nclass[:,1:l+1]==c,1).ravel()
        y_est = np.argmax(nclass_count,1);
        knn_errors[k,l-1] = 100.0*(y_est!=y_train.ravel()).sum()/float(N_train)

    k+=1

# Display results
'''
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))
'''
#fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

figure(2)
title('Optimal DTC levels for folds')
boxplot(dtc_error_test.T)
xlabel('Folds')
ylabel('Optimal pruning level')


fig_knn = figure(3)
ax_knn = fig_knn.add_subplot(111)
ax_knn.set_title('K-nearest neighbour test folds')
ax_knn.set_xlabel('Neighbours')
ax_knn.set_ylabel('Classification error rate (%)')
ax_knn.spines['top'].set_color('none')
ax_knn.spines['bottom'].set_color('none')
ax_knn.spines['left'].set_color('none')
ax_knn.spines['right'].set_color('none')
ax_knn.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

knn_compare = np.zeros(K)

for i in range(K):
    knn_compare[i] = min(knn_errors[i])
    t=fig_knn.add_subplot(5,2,i+1)
    t.plot(np.arange(1,max_neigh+1),knn_errors[i])
    for item in (t.get_yticklabels()):
        item.set_fontsize(9)

print "KNN best for folds:"
print knn_compare
print('KNN Error rate: {0}%'.format(mean(knn_compare)))

show()
