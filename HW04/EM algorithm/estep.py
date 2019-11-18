import numpy as np


def update_posterior(X_train,Lambda,Distribution):
    '''
    update posterior using log likelihood
    :param X_train: (60000,784) 0-1 uint8 matrix
    :param Lambda: (10,1)
    :param Distribution: (10,784)
    :return: (60000,10)
    '''
    Distribution_complement=1-Distribution
    W=np.zeros((60000,10))
    for i in range(60000):
        for j in range(10):
            W[i,j]=np.prod(X_train[i]*Distribution[j]+(1-X_train[i])*Distribution_complement[j])
    #add prior
    W = W*Lambda.reshape(1,-1)

    #normalized each row to [0,1] & sum=1
    sums = np.sum(W,axis=1).reshape(-1,1)
    sums[sums==0] = 1
    W = W/sums

    return W