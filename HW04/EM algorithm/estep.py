import numpy as np
import math
import traceback
from multiprocessing import Pool
from multiprocessing import shar


def handle_error(e):
    traceback.print_exception(type(e),e,e.__context__)

def subprocess(start,size,x_train,Lambda,Posterior):
    '''
    :param start:
    :param size:
    :param W:
    :param x_train:
    :param Lambda:
    :param Posterior:
    :return: (size,10) an subset of W
    '''
    w = np.zeros((size,10))
    for i in range(size):
        for k in range(10):
            tmp = 0
            for j in range(784):
                tmp += np.log(max(1e-3,Posterior[k,j,x_train[i][j]]))
            w[i,k] = tmp
            if math.isnan(tmp):
                print('i={},k={}'.format(i,k))

    w += np.log(Lambda.reshape(1,-1)) # multiply align axis 0 (data points)
    w = w/(np.sum(w,axis=1).reshape(-1,1)) # normalized align axis 1 (classes)
    #W[start:start+size]=w
    print('subprocess {} finish'.format(start//size))
    return w

def update_posterior(X_train,Lambda,Posterior):
    '''
    update posterior using log likelihood
    :param X_train: (60000,784) 0-1matrix
    :param Lambda: (10,1)
    :param Posterior: (10,784,2)
    :return: (60000,10)
    '''
    W = np.zeros((60000, 10))
    size=100
    with Pool(processes=8) as p:
        # each subprocess dealing 100 data points
        for i in range(60000//size):
            start = i*size
            result = p.apply_async(subprocess,args=(start,size,X_train[start:start+size],Lambda,Posterior),error_callback=handle_error)
            print(result)
        p.close()
        p.join()
        print('waiting for join......')
    '''
    W=np.zeros((60000,10))
    for k in range(10):
        tmp_list=np.zeros(10)
        for i in range(60000):
            tmp=0
            for j in range(784):
                tmp += np.log(max(1e-3,Posterior[k][j][X_train[i][j]]))
            tmp += np.log(max(1e-3,Lambda[k]))
        tmp_list[k]=tmp
        tmp_list /= np.sum(tmp_list) #normalized align classes
        W[i]=tmp_list
        print('W {} row finish'.format(i))
    '''

    print('W')
    print(W[:2])
    return W

