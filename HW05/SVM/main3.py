import numpy as np
from libsvm.svmutil import *
from scipy.spatial.distance import pdist,squareform
from load import load_x,load_y

def precomputed_kernel(X,gamma):
    kernel_linear=X @ X.T
    kernel_RBF=squareform(np.exp(-gamma*pdist(X,'sqeuclidean')))
    kernel=kernel_linear+kernel_RBF
    kernel=np.hstack((np.arange(1,len(X)+1).reshape(-1,1),kernel))
    return kernel

if __name__=='__main__':
    X_train=load_x('X_train.csv')
    y_train=load_y('Y_train.csv')
    X_test=load_x('X_test.csv')
    y_test=load_y('Y_test.csv')


    kernel_train=precomputed_kernel(X_train,2**-4)
    prob=svm_problem(y_train,kernel_train,isKernel=True)
    param=svm_parameter('-q -t 4')
    model=svm_train(prob,param)

    kernel_test=precomputed_kernel(X_test,2**-4)
    p_label,p_acc,p_vals=svm_predict(y_test,kernel_test,model,'-q')
    print('linear kernel + RBF kernel accuracy: {:.2f}%'.format(p_acc[0]))

