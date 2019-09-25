import numpy as np
from linearalgo.linalg import inverse

def newtonmethod(A,b):
    m,n=A.shape
    #initial parameters x
    x0=np.random.rand(n,1)
    eps=100
    while eps>1e-6:
        x1=x0-inverse(2*A.T@A)@(2*A.T@A@x0-2*A.T@b)
        eps=abs(np.sum(np.square(x1-x0))/n)
        x0=x1

    loss_value=get_loss_value(A,x0,b)
    return x0,loss_value

def get_loss_value(A,x,b):
    return np.sum(np.square(A@x-b))

