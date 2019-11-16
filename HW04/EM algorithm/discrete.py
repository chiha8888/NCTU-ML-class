import numpy as np

def get_pixvalueProb_discrete(train_x,train_y):
    '''
    get pixvalue_prob conditional on class & dim
    :param train_x: (60000,784) 0-1 matrix
    :param train_y: (60000,)
    :return: (10,784) probability matrix of pixelValue==1
    '''
    labels = np.zeros(10)
    for label in train_y:
        labels[label] += 1

    distribution=np.zeros((10,784))
    for i in range(60000):
        c=train_y[i]
        for j in range(784):
            if train_x[i,j]==1:
                distribution[c,j]+=1

    #normalized
    distribution = distribution / labels.reshape(-1,1)

    return distribution

def plot_discrete(Distribution):
    '''
    :param Distribution: (10,784)
    :return:
    '''
    for c in range(10):
        print('class',c)
        for i in range(28):
            for j in range(28):
                print(1 if Distribution[c,i*28+j]>0.5 else 0,end=' ')
            print()
        print()
        print()