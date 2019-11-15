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

def init_P():
    '''
    P[i,j,0]= pixel value==0 prob in class i's jth feature distribution
    P[i,j,1]= pixel value==1 prob in class i's jth feature distribution
    P[i,j,0]+P[i,j,1]=1 (for every i,j)
    :return: (10,784,2) matrix
    '''
    re=np.empty((10,784,2))
    re[:,:,0]=np.random.rand(10,784)
    re[:,:,1]=1-re[:,:,0]
    return re