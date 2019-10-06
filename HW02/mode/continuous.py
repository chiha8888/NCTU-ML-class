import math
from mode.util import *

def get_pixvalueProb_continuous(train_x,train_y):
    '''
    get pixvalue_prob conditional on class&dim
    :param train_x: (60000,784) ndarray
    :param train_y: (60000,) ndarray
    :return: (#class,#dim,#pix_value):(10,784,256) ndarray
    '''
    re =np.zeros((10,28*28,256))
    for c in range(10):  # 10 classes
        '''A:(#class c pics,784)'''
        A = train_x[train_y == c]
        for i in range(28*28):
            mu = get_mu(A[:, i])
            var = get_variance(A[:, i])
            for j in range(256):
                re[c,i,j] =gaussain_prob(j,mu,var)
    return re

def test_continuous(pics,pixvalueProb,prior,test_x,test_y):
    error = 0
    for i in range(pics):
        probs = np.zeros(10)
        for c in range(10):
            # posterior probability
            for d in range(28 * 28):  # tally likelihood(assume naive Baye's)
                probs[c] += np.log(max(1e-4,pixvalueProb[c, d, int(test_x[i, d])]))
                #print(test_x[i, d],' ',pixvalueProb[c, d, int(test_x[i, d])],' ',np.log(max(1e-4,pixvalueProb[c, d, int(test_x[i, d])])))
            probs[c] += np.log(prior[c])
        # normalized
        probs /= np.sum(probs)
        print('Posterior (in log scale):')
        for c in range(10):
            print('{}: {}'.format(c, probs[c]))
        predict=np.argmin(probs)
        print('Prediction: {}, Ans: {}'.format(predict, test_y[i]))
        print()
        if predict != test_y[i]:
            error += 1
    print('Error rate: {:.4f}'.format(error / pics))
    print()

def get_mu(array):
    return np.mean(array)

def get_variance(array):
    var=np.var(array)
    return var if var!=0 else 1e-4 #avoid Gaussian formula divided by zero

def gaussain_prob(x,mu,var):
    return ((1/math.sqrt(2*math.pi*var))*math.exp((-(x-mu)**2)/(2*var)))
