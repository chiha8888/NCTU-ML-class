import numpy as np

def get_prior(train_y):
    '''
    prior probability of each class
    :param train_y: (60000,) ndarray
    :return: (10,)
    '''
    re=np.zeros(10)
    for c in range(10):
        re[c]=np.sum(train_y==c)/len(train_y)
    return re

def print_imagination_numbers(pixvalueProb,threshold):
    print('Imagination of numbers in Bayesian classifier:')
    for c in range(10):
        print('{}:'.format(c))
        for i in range(28):
            for j in range(28):
                print('1' if np.argmax(pixvalueProb[c,i*28+j])>=threshold else '0',end=' ')
            print()
        print()
    print()