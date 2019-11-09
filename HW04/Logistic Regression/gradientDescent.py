import numpy as np

eps=1e-2

def run_gradient(A,w,b,lr=0.01):
    g=100
    while np.sqrt(np.sum(g**2))>eps:
        g=A.T@(b-1/(1+np.exp(-A@w)))
        w=w+lr*g
        #print('w={}'.format(w.reshape(1,-1)))
        #print('g={}'.format(g.reshape(1,-1)))

    return w
