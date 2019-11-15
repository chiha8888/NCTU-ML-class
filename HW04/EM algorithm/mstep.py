import numpy as np

def update_lambda(W):
    '''
    :param W: (60000,10)
    :return: (10,1)
    '''
    L=np.sum(W,axis=0)
    L/=np.sum(L)
    return L.T


def update_distribution(A,W):
    '''
    A.T@W -> normalized,transpose -> concate with 1-complement
    :param A: (60000,784)
    :param W: (60000,10)
    :return: (10,784,2)
    '''
    p=A.T@W
    weighted_sum=np.sum(W,axis=0)
    p/=weighted_sum
    p=p.T
    P=np.stack((1-p,p),axis=-1)
    return P