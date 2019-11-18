import numpy as np
from MNIST import load
from initial import init_lambda,init_P
from estep import update_posterior
from mstep import update_lambda,update_distribution
from discrete import get_pixvalueProb_discrete,plot_discrete
from match import perfect_matching
from utilPlot import *

eps=1

A,b=load()

#init, lambda represent by L
L=init_lambda()
P=init_P(A,b) # Distribution(784-dim) of each class

last_diff,diff,count=1000,100,0
while abs(last_diff-diff)>eps and diff>eps:
    #E-step (calculate posterior)
    W=update_posterior(A,L,P)

    #M-step (update L,P)
    L_new=update_lambda(W)
    P_new=update_distribution(A,W)
    #calculate diff
    last_diff=diff
    diff=np.sum(np.abs(L-L_new))+np.sum(np.abs(P-P_new))
    print('diff: ',diff)
    print('Lambda:',L_new.reshape(1,-1)[0])
    L=L_new
    P=P_new
    count+=1


#take a view of classes belonging (but not exactly class)
maxs=np.argmax(W,axis=1)
unique,counts=np.unique(maxs,return_counts=True)
print(dict(zip(unique,counts)))
print('Lambda:',L.reshape(1,-1))

#plot classes predict & confusion matrix
GT_distribution=get_pixvalueProb_discrete(A,b)
'''
plot_discrete(distribution)
'''
class_order=perfect_matching(GT_distribution,P)

plot(P,class_order,threshold=0.35)
confusion_matrix(b,maxs,class_order)
print_error_rate(count,b,maxs,class_order)
