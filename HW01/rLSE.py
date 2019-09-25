import numpy as np
from linearalgo.linalg import inverse

def rlse(A,LAMBDA,b):
    m,n=A.shape
    x=inverse(A.T@A+LAMBDA*np.identity(n))@A.T@b

    loss_value=get_loss_value(A,x,b)
    return x,loss_value

def get_loss_value(A,x,b):
    return np.sum(np.square(A@x-b))