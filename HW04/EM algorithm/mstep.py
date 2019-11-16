import numpy as np

def update_lambda(W):
    '''
    :param W: (60000,10)
    :return: (10,1)
    '''
    L = np.sum(W,axis=0)
    L = L/60000
    return L.T


def update_distribution(A,W):
    '''
    A.T@W -> normalized,transpose -> concate with 1-complement
    :param A: (60000,784)
    :param W: (60000,10)
    :return: (10,784)
    '''
    #normalized W
    sums = np.sum(W,axis=0)
    W_normalized = W/sums
    P=A.T@W_normalized

    return P.T