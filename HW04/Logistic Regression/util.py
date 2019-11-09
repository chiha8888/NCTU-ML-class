import numpy as np
import random
import math

def default(case):
    return (50,1,1,10,10,2,2,2,2) if case==1 else (50,1,1,3,3,2,2,4,4)

def gaussian_distribution(m,s):
    U=random.random()
    V=random.random()
    z=math.sqrt(-2*math.log(U))*math.cos(2*math.pi*V)
    sample=z*s+m
    return sample

def sampling(mx,my,vx,vy,N):
    '''
    re=np.random.multivariate_normal(mean=[mx,my],cov=[[vx,0],[0,vy]],size=N)
    :param mx: x mean
    :param my: y mean
    :param vx: x variance
    :param vy: y variance
    :param N: sampling N data points
    :return: (N,2) shape matrix
    '''
    re=np.empty((N,2))
    for i in range(N):
        re[i,0]=gaussian_distribution(mx,vx)
        re[i,1]=gaussian_distribution(my,vy)
    return re

def predict(A,w):
    '''
    predict whether is class0 or class1
    :param A: (2N,3) shape matrix
    :param w: (3,1) shape matrix
    :return: (2N,1) shape matrix
    '''
    N=len(A)
    b_predict=np.empty((N,1))
    for i in range(N):
        b_predict[i]=0 if A[i]@w<0 else 1

    return b_predict

def get_A(C0,C1):
    A = np.ones((2 * len(C0), 3))
    A[:, 1:] = np.vstack((C0, C1))
    return A

def get_b(N):
    b = np.zeros((2 * N, 1))
    b[N:] = np.ones((N, 1))
    return b