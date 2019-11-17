import numpy as np

eps=1e-30

def update_posterior(X_train,Lambda,Distribution):
    '''
    update posterior using log likelihood
    :param X_train: (60000,784) 0-1 uint8 matrix
    :param Lambda: (10,1)
    :param Distribution: (10,784)
    :return: (60000,10)
    '''
    Distribution[Distribution < eps] = eps
    Distribution_log=np.log(Distribution)
    Distribution_complement_log=np.log(1-Distribution)
    W=np.zeros((60000,10))
    for i in range(60000):
        for j in range(10):
            W[i,j]=X_train[i]@Distribution_log[j].T+(1-X_train[i])@Distribution_complement_log[j].T
    #add prior
    Lambda_log=np.log(Lambda).reshape(1,-1)
    W=W+Lambda_log

    #normalized each row negative values to [0,1] & sum=1
    mins=np.min(W,axis=1).reshape(-1,1)
    W=W-mins
    sums=np.sum(W,axis=1).reshape(-1,1)
    W=W/sums

    return W