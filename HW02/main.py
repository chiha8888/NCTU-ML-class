import sys
from MNIST import load
from mode.discrete import *
from mode.continuous import *
from mode.util import *
import numpy as np
import math

if __name__=='__main__':
    (train_x,train_y),(test_x,test_y)=load()
    #train_x=train_x.reshape((train_x.shape[0],train_x.shape[1]*train_x.shape[2]))
    #test_x=test_x.reshape((test_x.shape[0],test_x.shape[1]*test_x.shape[2]))
    print(train_x.dtype)
    print(train_y.dtype)

    toggle_bar=input('Toggle option (0:discrete / 1:continuous): ')
    #discrete mode or continuous mode(Gaussian)
    if toggle_bar=='0':
        pixvalueProb = get_pixvalueProb_discrete(train_x, train_y)
        prior=get_prior(train_y)
        test_discrete(len(test_y), pixvalueProb,prior, test_x, test_y)
        print_imagination_numbers(pixvalueProb,16)
    else:
        pixvalueProb=get_pixvalueProb_continuous(train_x,train_y)
        prior=get_prior(train_y)
        test_continuous(len(test_y),pixvalueProb,prior,test_x,test_y)
        print_imagination_numbers(pixvalueProb,128)


