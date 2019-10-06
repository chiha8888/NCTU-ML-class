from mode.util import *

def get_pixvalueProb_discrete(train_x,train_y):
    '''
    get pixvalue_prob conditional on class&dim
    :param train_x: (60000,784) ndarray
    :param train_y: (60000,) ndarray
    :return: (#class,#dim,#pix_value):(10,784,32) ndarray
    '''
    re=np.zeros((10,28*28,32))
    for i in range(len(train_x)):
        c=train_y[i]
        for d in range(28*28):
            re[c][d][int(train_x[i,d])//8]+=1
    for c in range(10):
        for d in range(28*28):
            count=0
            for p in range(32):
                count+=re[c][d][p]
            re[c][d][:]/=count
    return re

def test_discrete(pics,pixvalueProb,prior,test_x,test_y):
    error=0
    for i in range(pics):
        probs = np.zeros(10)
        for c in range(10):
            # posterior probability
            for d in range(28 * 28):  # tally likelihood(assume naive Baye's)
                    probs[c] += np.log(max(1e-4,pixvalueProb[c, d, int(test_x[i, d])//8]))
            probs[c] += np.log(prior[c])
        # normalized
        '''
        probs is negative!!!
        So after normalization,it will be positive,that changes argmax to argmin.
        '''
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
