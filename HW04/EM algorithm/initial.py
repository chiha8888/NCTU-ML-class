import numpy as np

def init_lambda():
    '''
    lambda[k]= prior of class k
    sum(lambda)=1
    :return: (10,1) matrix
    '''
    re=np.random.rand(10,1)
    re=re/np.sum(re)
    return re

def init_P(A,b):
    '''
    P[i,j]: pixel value==1 prob in class i's jth feature distribution
    :return: (10,784) matrix
    '''
    re=np.random.rand(10,784)

    return re