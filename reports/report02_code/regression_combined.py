from pylab import *
from scipy.io import loadmat
import neurolab as nl
from sklearn import cross_validation
import scipy.linalg as linalg
from scipy import stats
import numpy as np
from scipy.stats import zscore
import sklearn.linear_model as lm
from toolbox_02450 import feature_selector_lr, bmplot
from scipy.stats import ttest_rel


X = np.matrix(np.loadtxt("../dataset/spam.data"))
attributeNames = np.loadtxt("../dataset/spambase.names", dtype='str').T
attributeNames = attributeNames[1:55]

# Normalize data
X = stats.zscore(X);
spam_class = X[:,57].T
y = X[:,56]
X = X[:,1:55]
N, M = X.shape

# K-fold crossvalidation
K = 10
CV = cross_validation.KFold(N,K,shuffle=True)

#---------------------------ANN PARAMETERS------------------------------------
# Parameters for ANN classifier
ANN_HIDDEN_UNITS = 2      # number of hidden units
ANN_TRAIN = 5             # number of networks trained in each k-fold
ANN_LEARNING_GOAL = 10    # stop criterion 1 (train mse to be reached)
ANN_MAX_EPOCHS = 64         # stop criterion 2 (max epochs in training)
ANN_SHOW_ERR_FREQ = 3     # frequency of training status updates
ANN_ERRORS = np.zeros(K)
ANN_ERROR_HIST = np.zeros((ANN_MAX_EPOCHS,K))
ANN_BESTNET = list()
#---------------------------ANN PARAMETERS------------------------------------

#---------------------------LINEAR PARAMETERS---------------------------------
# Initialize variables
LINEAR_FEATURES = np.zeros((M,K))
LINEAR_ERROR_TRAIN = np.empty((K,1))
LINEAR_ERROR_TEST = np.empty((K,1))
LINEAR_ERROR_TRAIN_FS = np.empty((K,1))
LINEAR_ERROR_TEST_FS = np.empty((K,1))
LINEAR_ERROR_TRAIN_NOFEATURES = np.empty((K,1))
LINEAR_ERROR_TEST_NOFEATURES = np.empty((K,1))
#---------------------------LINEAR PARAMETERS---------------------------------

k=0
for train_index, test_index in CV:
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))

    # Training and test data
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    print('--------------START LINEAR ON FOLD--------------')
    LINEAR_INTERNAL_CROSS_VALIDATION = 10
    # Compute squared error without using the input data at all
    LINEAR_ERROR_TRAIN_NOFEATURES[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    LINEAR_ERROR_TEST_NOFEATURES[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    model = lm.LinearRegression().fit(X_train, y_train)
    LINEAR_ERROR_TRAIN[k] = np.square(y_train-model.predict(X_train)).sum()/y_train.shape[0]
    LINEAR_ERROR_TEST[k] = np.square(y_test-model.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, LINEAR_INTERNAL_CROSS_VALIDATION)
    LINEAR_FEATURES[selected_features,k]=1

    model = lm.LinearRegression().fit(X_train[:,selected_features], y_train)
    LINEAR_ERROR_TRAIN_FS[k] = np.square(y_train-model.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
    LINEAR_ERROR_TEST_FS[k] = np.square(y_test-model.predict(X_test[:,selected_features])).sum()/y_test.shape[0]

    print('MODEL COEFFICENTS: ')
    print('Selected Features: ' + str(selected_features))
    params = attributeNames[selected_features]
    for ind in range(len(selected_features)):
        print params[ind] + ": " + str(model.coef_[:,ind])

    figure(k)
    subplot(1,2,1)
    plot(range(1,len(loss_record)), loss_record[1:])
    xlabel('Iteration')
    ylabel('Squared error (crossvalidation)')

    # subplot(1,3,3)
    # bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
    # clim(-1.5,0)
    # xlabel('Iteration')

    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))
    print('--------------STOP LINEAR ON FOLD--------------')

    print('--------------START ANN ON FOLD--------------')
    best_train_error = 1e100
    for i in range(ANN_TRAIN):
        print('Training network {0}/{1}...'.format(i+1,ANN_TRAIN))
        # Create randomly initialized network with 2 layers
        ann = nl.net.newff([[-3, 3]]*M, [ANN_HIDDEN_UNITS, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
        if i==0:
            ANN_BESTNET.append(ann)
        # train network
        train_error = ann.train(X_train, y_train, goal=ANN_LEARNING_GOAL, epochs=ANN_MAX_EPOCHS, show=ANN_SHOW_ERR_FREQ)
        if train_error[-1]<best_train_error:
            ANN_BESTNET[k]=ann
            best_train_error = train_error[-1]
            ANN_ERROR_HIST[range(len(train_error)),k] = train_error

    print('Best train error: {0}...'.format(best_train_error))
    ANN_Y_EST = ANN_BESTNET[k].sim(X_test)
    ANN_ERRORS[k] = np.power(ANN_Y_EST-y_test,2).sum().astype(float)/y_test.shape[0]
    print('Best Net ' + str(k) + ': ' + str(ANN_BESTNET[k]))
    print('--------------STOP ANN ON FOLD--------------')
    k+=1

print('-------------- START LINEAR CONCLUSION --------------')
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(LINEAR_ERROR_TRAIN.mean()))
print('- Test error:     {0}'.format(LINEAR_ERROR_TEST.mean()))
print('- R^2 train:     {0}'.format((LINEAR_ERROR_TRAIN_NOFEATURES.sum()-LINEAR_ERROR_TRAIN.sum())/LINEAR_ERROR_TRAIN_NOFEATURES.sum()))
print('- R^2 test:     {0}'.format((LINEAR_ERROR_TEST_NOFEATURES.sum()-LINEAR_ERROR_TEST.sum())/LINEAR_ERROR_TEST_NOFEATURES.sum()))
# print('- Error rate train: {0}%'.format(100*mean(LINEAR_ERROR_TRAIN)))
# print('- Error rate test: {0}%'.format(100*mean(LINEAR_ERROR_TEST)))

print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(LINEAR_ERROR_TRAIN_FS.mean()))
print('- Test error:     {0}'.format(LINEAR_ERROR_TEST_FS.mean()))
print('- R^2 train:     {0}'.format((LINEAR_ERROR_TRAIN_NOFEATURES.sum()-LINEAR_ERROR_TRAIN_FS.sum())/LINEAR_ERROR_TRAIN_NOFEATURES.sum()))
print('- R^2 test:     {0}'.format((LINEAR_ERROR_TEST_NOFEATURES.sum()-LINEAR_ERROR_TEST_FS.sum())/LINEAR_ERROR_TEST_NOFEATURES.sum()))
# print('- Error rate train: {0}%'.format(100*mean(LINEAR_ERROR_TRAIN_FS)))
# print('- Error rate test: {0}%'.format(100*mean(LINEAR_ERROR_TEST_FS)))

figure(k)
subplot(1,3,2)
bmplot(attributeNames, range(1,LINEAR_FEATURES.shape[1]+1), -LINEAR_FEATURES)
clim(-1.5,0)
xlabel('Crossvalidation fold')
ylabel('Attribute')

f=2 # cross-validation fold to inspect
ff=LINEAR_FEATURES[:,f-1].nonzero()[0]
m = lm.LinearRegression().fit(X[:,ff], y)

# print "ff: " + str(ff)
# params = attributeNames[ff]
# coefficients = m.coef_
#
# for ind in range(len(ff)):
#     print params[ind] + ": " + str(coefficients[ind])

print "Linear Model Parameters: " + str(m.get_params())

y_est= m.predict(X[:,ff])
residual=y-y_est

figure(k+1)
title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
for i in range(0,len(ff)):
   subplot(2,ceil(len(ff)/2.0),i+1)
   # plot(X[:,ff[i]].A,residual.A,'.')
   plot(X[:,ff[i]],residual,'.')
   xlabel(attributeNames[ff[i]])
   ylabel('residual error')

show()

print('-------------- STOP LINEAR CONCLUSION --------------')


print('-------------- START ANN CONCLUSION --------------')
# Print the average least squares error
print('ANN Mean-square error: {0}'.format(mean(ANN_ERRORS)))
# Print the average classification error rate
# print('ANN Error rate: {0}%'.format(100*mean(ANN_ERRORS)))

figure();
subplot(2,1,1); bar(range(0,K),ANN_ERRORS); title('ANN Mean-square errors');
subplot(2,1,2); plot(ANN_ERROR_HIST); title('ANN Training error as function of BP iterations');
figure();
subplot(2,1,1); plot(ANN_Y_EST); plot(y_test.A); title('ANN Last CV-fold: est_y vs. test_y');
subplot(2,1,2); plot((ANN_Y_EST-y_test).A); title('ANN Last CV-fold: prediction error (est_y-test_y)'); show()
show()
print('-------------- STOP ANN CONCLUSION --------------')

print('-------------- START AVERAGE TEST ----------------')

print('-------------- STOP AVERAGE TEST ----------------')

print('-------------- START PAIRED T-TEST ----------------')
print "t-value and p-value"
print 'Linear Error Test with FS: ' + str(LINEAR_ERROR_TEST_FS)
print 'ANN Errors: ' + str(ANN_ERRORS)
print str(ttest_rel(np.asarray(LINEAR_ERROR_TEST_FS),np.asarray(ANN_ERRORS)))
print('-------------- STOP PAIRED T-TEST ----------------')
