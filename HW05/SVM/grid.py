from libsvm.svmutil import *
import numpy as np

'''
svm_train return:
the returned svm_model instance. See svm.h for details of this
structure. If '-v' is specified, cross validation is
conducted and the returned model is just a scalar: cross-validation
accuracy for classification and mean-squared error for regression.
'''

def grid_search(log2c,log2g,X_train,y_train,X_test,y_test):
    confusion_matrix=np.zeros((len(log2c),len(log2g)))
    for i in range(len(log2c)):
        for j in range(len(log2g)):
            param='-q -t 2 -v 3 -c {} -g {}'.format(2**log2c[i],2**log2g[j])
            acc=svm_train(y_train,X_train,param)
            confusion_matrix[i,j]=acc
    return confusion_matrix