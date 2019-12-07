import numpy as np

def load_data(path='input.data'):
    x=[]
    y=[]
    f = open(path, 'r')
    for line in f.readlines():
        datapoint = line.split(' ')
        x.append(float(datapoint[0]))
        y.append(float(datapoint[1]))
    f.close()
    x=np.asarray(x)
    y=np.asarray(y)
    return x,y

def kernel(X1,X2,alpha=1,length_scale=1):
    '''
    using rational quadratic kernel function: k(x_i, x_j) = (1 + (x_i-x_j)^2 / (2*alpha * length_scale^2))^-alpha
    :param X1: (m,d) ndarray
    :param X2: (n,d) ndarray
    return: (m,n) kernel ndarray
    '''
    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    kernel = (1+sqdist**2/(2*alpha*length_scale**2))**-alpha

    return kernel


def get_mus(x_line,x,y,kernel):
    '''
    :param x_line: sampling by linspace(-60,60)
    :param x: datapoints x
    :param y: datapoints y
    :param kernel: (n,n) matrix
    return: (len(x_line)) array
    '''
    n=len(x_line)
    means=np.zeros(n)
    for i in range(n):
        k_x_xstar=kernel_function(x_line[i],x)
        means[i]=k_x_xstar @ np.linalg.inv(kernel) @ y.reshape(-1,1)
    return means

def get_vars(x_line,x,kernel,beta):
    '''
    :param x_line: sampling by linspace(-60,60)
    :param x: datapoints x
    :param kernel: (n,n) matrix
    :param beta: noise scalar
    :return: (len(x_line)) array
    '''
    n=len(x_line)
    variances=np.zeros(n)
    for i in range(n):
        k_x_xstar=kernel_function(x_line[i],x)
        variances[i]=(kernel_function(x_line[i],x_line[i]) + 1/beta) + k_x_xstar @ np.linalg.inv(kernel) @ k_x_xstar.T
    return variances
