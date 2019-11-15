import numpy as np
from MNIST import load
from initial import *
from estep import *
from mstep import *


A,b=load()

#init, lambda represent by L
L=init_lambda()
P=init_P()

diff=100
while diff>1:
    #E-step (calculate posterior)
    W=update_posterior(A,L,P)
    '''
    #print(W)
    #M-step (update L,P)
    L_new=update_lambda(W)
    P_new=update_distribution(A,W)
    diff=np.sum(np.abs(L-L_new))+np.sum(np.abs(P[:,:,0]-P_new[:,:,1]))
    print('diff: ',diff)
    L=L_new
    P=P_new
    '''